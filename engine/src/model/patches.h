// patches.h — runtime switches for the numbered patches listed in patches.def.
//   if (q38::patch::on(q38::patch::rdma_ar)) { … }
#pragma once
#include <string>

namespace q38::patch {

enum Id : int {
#define Q38_PATCH(num, name, def, what) name = num,
#include "model/patches.def"
#undef Q38_PATCH
  kNone = 0
};

bool on(Id id);           // patches.def default, overridden once by env Q38_PATCHES ("+name,-name")
std::string active();     // "P001 rdma_ar, …" or "none" — logged once at engine start
void set(Id id, bool on); // tests: switch a patch inside one process (base vs patch comparisons)
void set(const std::string& name, bool on);
bool on(const std::string& name);  // by name (bindings); unknown name → fail

}  // namespace q38::patch
