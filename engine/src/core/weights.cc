#include "core/weights.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <sstream>

#include "core/log.h"

namespace q38 {

static std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> r;
  std::string x;
  std::istringstream is(s);
  while (std::getline(is, x, d)) r.push_back(x);
  if (!s.empty() && s.back() == d) r.emplace_back();
  return r;
}

static size_t itemsize(const std::string& dt) {
  if (dt == "BF16" || dt == "F16" || dt == "I16") return 2;
  if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
  if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
  return 1;  // U8, I8, F8_E4M3, BOOL
}

Weights::Weights(const std::string& model_dir, const std::string& table_tsv, int rank) : dir_(model_dir), rank_(rank) {
  std::ifstream f(table_tsv);
  if (!f) qlog::fail("weights", "no tensor table: %s", table_tsv.c_str());
  std::string line;
  std::getline(f, line);
  auto hdr = split(line, '\t');
  auto col = [&](const char* n) {
    for (size_t i = 0; i < hdr.size(); i++) if (hdr[i] == n) return int(i);
    qlog::fail("weights", "tensor table has no %s column", n);
  };
  int c_name = col("name"), c_file = col("file"), c_off = col("offset"), c_nb = col("nbytes"), c_dt = col("dtype"),
      c_sh = col("shape"), c_tp = col("tp"), c_bl = col("block"), c_pa = col("part");
  while (std::getline(f, line)) {
    auto v = split(line, '\t');
    if (v[c_tp] == "skip") continue;
    TensorInfo ti{v[c_file], v[c_dt], v[c_tp], v[c_bl], v[c_pa], std::stoull(v[c_off]), std::stoull(v[c_nb]), {}};
    if (v[c_sh] != "-") for (auto& d : split(v[c_sh], 'x')) ti.shape.push_back(std::stoll(d));
    if (ti.tp == "-") qlog::fail("weights", "tp column is empty — rebuild the table with tools/tp_plan.py");
    t_.emplace(v[c_name], std::move(ti));
  }
  drop_cache();  // start from an empty page cache for the shard files (left by earlier readers)
}

void Weights::drop_cache() const {
  std::unordered_map<std::string, int> seen;
  for (auto& [n, ti] : t_) seen.emplace(ti.file, 0);
  for (auto& [f, _] : seen) {
    int fd = open((dir_ + "/" + f).c_str(), O_RDONLY);
    if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), close(fd);
  }
}

Weights::~Weights() {
  for (auto& [k, fd] : fds_) close(fd);
}

const TensorInfo& Weights::info(const std::string& name) const {
  auto it = t_.find(name);
  if (it == t_.end()) qlog::fail("weights", "no such tensor: %s", name.c_str());
  return it->second;
}

void Weights::pread_all(const TensorInfo& ti, uint64_t off, uint64_t len, uint8_t* dst) const {
  auto it = fds_.find(ti.file);
  int fd;
  if (it == fds_.end()) {
    fd = open((dir_ + "/" + ti.file).c_str(), O_RDONLY);
    if (fd < 0) qlog::fail("weights", "cannot open: %s", ti.file.c_str());
    fds_[ti.file] = fd;
  } else {
    fd = it->second;
  }
  uint64_t done = 0;
  while (done < len) {
    ssize_t r = pread(fd, dst + done, len - done, ti.offset + off + done);
    if (r <= 0) qlog::fail("weights", "read failed: %s @%llu", ti.file.c_str(), (unsigned long long)(ti.offset + off + done));
    done += r;
  }
  // drop what we just read from the page cache (no root needed): on GB10 the GPU driver needs free pages, and ~50 GB of
  // cached shard pages can stall cudaMalloc (a note in the reference deployment's recipe)
  posix_fadvise(fd, ti.offset + off, len, POSIX_FADV_DONTNEED);
}

std::vector<uint8_t> Weights::raw(const std::string& name) const {
  const auto& ti = info(name);
  std::vector<uint8_t> b(ti.nbytes);
  pread_all(ti, 0, ti.nbytes, b.data());
  return b;
}

std::vector<uint8_t> csf_decode(const uint8_t* fixed, int64_t groups, int64_t width, const uint32_t* exc, int64_t n_exc) {
  const int64_t rows = 16 * groups, cols = (width - 16) / 8, half = cols / 2;
  std::vector<uint8_t> out(rows * cols);
  for (int64_t g = 0; g < groups; g++) {
    const uint8_t* fg = fixed + g * width;
    for (int i = 0; i < 16; i++) {
      const uint8_t base = fg[i];
      const uint8_t* p = fg + 16 + i * half;
      uint8_t* o = out.data() + (16 * g + i) * cols;
      for (int64_t j = 0; j < half; j++) {
        o[2 * j] = uint8_t(base + (p[j] & 15));
        o[2 * j + 1] = uint8_t(base + (p[j] >> 4));
      }
    }
  }
  for (int64_t e = 0; e < n_exc; e++) out[exc[e] & 0xFFFFFF] = uint8_t(exc[e] >> 24);
  return out;
}

// dim0 rows [r0, r1) of a row-major tensor with `rb` bytes per row
static void take_rows(const std::vector<uint8_t>& src, uint64_t rb, int64_t r0, int64_t r1, std::vector<uint8_t>& dst) {
  dst.insert(dst.end(), src.begin() + r0 * rb, src.begin() + r1 * rb);
}

std::vector<uint8_t> Weights::slice(const std::string& name, std::vector<int64_t>* shape_out) const {
  const auto& ti = info(name);
  std::vector<int64_t> shape = ti.shape;
  const std::string& tp = ti.tp;
  std::vector<uint8_t> out;
  const int r = rank_;

  if (tp == "rep") {
    out = raw(name);
  } else if (tp.rfind("own:", 0) == 0) {
    if (std::stoi(tp.substr(4)) == r) out = raw(name);
    else shape.assign(1, 0);
  } else if (tp == "col") {
    uint64_t h = ti.nbytes / 2;
    out.resize(h);
    pread_all(ti, r * h, h, out.data());
    shape[0] /= 2;
  } else if (tp == "row") {
    auto all = raw(name);
    uint64_t rb = ti.nbytes / shape[0], hb = rb / 2;
    out.resize(shape[0] * hb);
    for (int64_t i = 0; i < shape[0]; i++) std::memcpy(out.data() + i * hb, all.data() + i * rb + r * hb, hb);
    shape[1] /= 2;
  } else if (tp.rfind("seg:", 0) == 0) {
    auto all = raw(name);
    uint64_t rb = ti.nbytes / shape[0];
    int64_t base = 0;
    for (auto& s : split(tp.substr(4), ',')) {
      int64_t n = std::stoll(s);
      take_rows(all, rb, base + r * n / 2, base + (r + 1) * n / 2, out);
      base += n;
    }
    if (base != shape[0]) qlog::fail("weights", "%s: seg sum %lld ≠ dim0 %lld", name.c_str(), (long long)base, (long long)shape[0]);
    shape[0] /= 2;
  } else if (tp.rfind("csf:", 0) == 0) {
    // name is the `...weight_scale.nvfp4_csf_fixed` tensor; its exceptions sit next to it
    const std::string pre = name.substr(0, name.rfind('.'));
    auto fx = raw(pre + ".nvfp4_csf_fixed");
    auto ex = raw(pre + ".nvfp4_csf_exceptions");
    const auto& fi = info(pre + ".nvfp4_csf_fixed");
    auto dec = csf_decode(fx.data(), fi.shape[0], fi.shape[1], reinterpret_cast<const uint32_t*>(ex.data()),
                          int64_t(ex.size() / 4));
    int64_t rows = 16 * fi.shape[0], cols = (fi.shape[1] - 16) / 8;
    if (tp == "csf:col") {
      take_rows(dec, cols, r * rows / 2, (r + 1) * rows / 2, out);
      shape = {rows / 2, cols};
    } else {
      out.resize(rows * cols / 2);
      for (int64_t i = 0; i < rows; i++) std::memcpy(out.data() + i * cols / 2, dec.data() + i * cols + r * cols / 2, cols / 2);
      shape = {rows, cols / 2};
    }
  } else {
    qlog::fail("weights", "%s: unknown tp rule %s", name.c_str(), tp.c_str());
  }
  if (shape_out) *shape_out = shape;
  return out;
}

}  // namespace q38
