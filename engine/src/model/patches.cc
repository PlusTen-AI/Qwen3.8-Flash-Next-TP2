#include "model/patches.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

#include "core/log.h"

namespace q38::patch {

namespace {
struct Entry {
  int num;
  const char* name;
  bool on;
};

std::map<int, Entry>& table() {
  static std::map<int, Entry> t = [] {
    std::map<int, Entry> m;
#define Q38_PATCH(num, name, def, what) m[num] = Entry{num, #name, def};
#include "model/patches.def"
#undef Q38_PATCH
    if (const char* env = std::getenv("Q38_PATCHES")) {
      std::istringstream is(env);
      std::string tok;
      while (std::getline(is, tok, ',')) {
        if (tok.empty()) continue;
        bool v = tok[0] != '-';
        std::string n = (tok[0] == '+' || tok[0] == '-') ? tok.substr(1) : tok;
        bool found = false;
        for (auto& [k, e] : m)
          if (n == e.name) e.on = v, found = true;
        if (!found) qlog::fail("patch", "unknown patch: %s (Q38_PATCHES)", n.c_str());
      }
    }
    return m;
  }();
  return t;
}
}  // namespace

bool on(Id id) {
  auto& t = table();
  auto it = t.find(int(id));
  return it != t.end() && it->second.on;
}

void set(Id id, bool v) {
  auto& t = table();
  auto it = t.find(int(id));
  if (it == t.end()) qlog::fail("patch", "unknown patch number %d", int(id));
  it->second.on = v;
}

void set(const std::string& name, bool v) {
  for (auto& [k, e] : table())
    if (name == e.name) return void(e.on = v);
  qlog::fail("patch", "unknown patch: %s", name.c_str());
}

bool on(const std::string& name) {
  for (auto& [k, e] : table())
    if (name == e.name) return e.on;
  qlog::fail("patch", "unknown patch: %s", name.c_str());
  return false;
}

std::string active() {
  std::string s;
  char buf[96];
  for (auto& [k, e] : table())
    if (e.on) {
      std::snprintf(buf, sizeof buf, "%sP%03d %s", s.empty() ? "" : ", ", e.num, e.name);
      s += buf;
    }
  return s.empty() ? "none" : s;
}

}  // namespace q38::patch
