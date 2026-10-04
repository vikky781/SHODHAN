#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

void register_selectors(PluginRegistry&);
void register_branching_simple(PluginRegistry&);
void register_branching_reliability(PluginRegistry&);

namespace {

void register_builtin(PluginRegistry& reg) {
  register_selectors(reg);
  register_branching_simple(reg);
  register_branching_reliability(reg);
}

}  // namespace

PluginRegistry& plugin_registry() {
  static PluginRegistry registry = [] {
    PluginRegistry r;
    register_builtin(r);
    return r;
  }();
  return registry;
}

}  // namespace shodhan::mip
