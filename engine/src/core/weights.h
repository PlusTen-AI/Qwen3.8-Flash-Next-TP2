// weights.h — tensor table + this rank's slice of each tensor (host bytes), same rules as tools/tp_plan.py:
//   rep | col (dim0 halves) | row (dim1 halves) | seg:a,b,c (dim0 segments, each halved) | own:r | csf:col/row | skip
// CSF expert scales are decoded to standard E4M3 [N, K/16] (tools/csf_ref.py) before splitting.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace q38 {

struct TensorInfo {
  std::string file, dtype, tp, block, part;
  uint64_t offset = 0, nbytes = 0;
  std::vector<int64_t> shape;
};

class Weights {
 public:
  Weights(const std::string& model_dir, const std::string& table_tsv, int rank);
  ~Weights();
  bool has(const std::string& name) const { return t_.count(name) != 0; }
  const TensorInfo& info(const std::string& name) const;
  // This rank's bytes of `name` (row-major; for csf the decoded E4M3 scales). `shape` gets the local shape.
  std::vector<uint8_t> slice(const std::string& name, std::vector<int64_t>* shape = nullptr) const;
  // Whole tensor as stored (no split, no decode).
  std::vector<uint8_t> raw(const std::string& name) const;
  int rank() const { return rank_; }
  void drop_cache() const;  // posix_fadvise(DONTNEED) on every shard file (no root): GB10's driver wants free pages
  size_t count() const { return t_.size(); }

 private:
  void pread_all(const TensorInfo& ti, uint64_t off, uint64_t len, uint8_t* dst) const;
  std::string dir_;
  int rank_;
  std::unordered_map<std::string, TensorInfo> t_;
  mutable std::unordered_map<std::string, int> fds_;
};

// CSF decode: fixed u8 [G, 16+8C] + exceptions u32 [E] -> E4M3 bytes [16G, C]
std::vector<uint8_t> csf_decode(const uint8_t* fixed, int64_t groups, int64_t width, const uint32_t* exc, int64_t n_exc);

}  // namespace q38
