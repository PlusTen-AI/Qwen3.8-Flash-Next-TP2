// log.h — same contract as tools/qlog.py: one line per step `[tag] point`; green OK, red fail (exits), yellow warn
// (printed once per key, counted at exit). Long output goes to the detail file; the screen shows only its path.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace qlog {
inline bool color() { static bool c = !std::getenv("NO_COLOR"); return c; }
inline std::mutex& mu() { static std::mutex m; return m; }
inline FILE*& detail_fp() { static FILE* f = nullptr; return f; }

inline void line(const char* col, const char* tag, const char* fmt, va_list ap) {
  char buf[2048];
  vsnprintf(buf, sizeof buf, fmt, ap);
  std::lock_guard<std::mutex> g(mu());
  if (color()) std::printf("\033[36m[%s]\033[0m %s%s%s\n", tag, col, buf, *col ? "\033[0m" : "");
  else std::printf("[%s] %s\n", tag, buf);
  std::fflush(stdout);
}
#define QLOG_FN(name, col)                                   \
  inline void name(const char* tag, const char* fmt, ...) {  \
    va_list ap; va_start(ap, fmt);                           \
    line(color() ? col : "", tag, fmt, ap); va_end(ap);      \
  }
QLOG_FN(info, "")
QLOG_FN(ok, "\033[32m")
#undef QLOG_FN

[[noreturn]] inline void fail(const char* tag, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  line(color() ? "\033[31m" : "", tag, fmt, ap); va_end(ap);
  std::exit(1);
}

struct WarnCounts {
  std::map<std::string, int> n;
  ~WarnCounts() {
    for (auto& [k, c] : n)
      if (c > 1) std::printf(color() ? "\033[33m%s ×%d\033[0m\n" : "%s ×%d\n", k.c_str(), c);
  }
};
inline WarnCounts& warns() { static WarnCounts w; return w; }

// Printed once per (tag, message); repeats are counted and summarised at exit.
inline void warn(const char* tag, const char* fmt, ...) {
  char buf[2048];
  va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  std::string key = std::string("[") + tag + "] " + buf;
  int c;
  { std::lock_guard<std::mutex> g(mu()); c = ++warns().n[key]; }
  if (c == 1) {
    std::lock_guard<std::mutex> g(mu());
    if (color()) std::printf("\033[36m[%s]\033[0m \033[33m%s\033[0m\n", tag, buf);
    else std::printf("[%s] %s\n", tag, buf);
    std::fflush(stdout);
  }
}

// Breadcrumb: the engine's last finished op group and layer (Engine::mark, one store each). A CUDA fault surfaces at the
// first API call after the faulting kernel, so the CUDA error line names this point to narrow down where it happened.
inline const char*& where() { static const char* w = "-"; return w; }
inline int& where_layer() { static int l = -1; return l; }

inline void open_detail(const char* path) { detail_fp() = std::fopen(path, "a"); }
inline void detail(const char* fmt, ...) {
  FILE* f = detail_fp();
  if (!f) return;
  va_list ap; va_start(ap, fmt); std::vfprintf(f, fmt, ap); va_end(ap);
  std::fputc('\n', f); std::fflush(f);
}
}  // namespace qlog

#define CUDA_CHECK(x)                                                                          \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess)                                                                     \
      qlog::fail("cuda", "%s · %s:%d · last %s (layer %d)", cudaGetErrorString(e_), __FILE__, __LINE__, qlog::where(), \
                 qlog::where_layer());                                                         \
  } while (0)
