#include "shodhan/mip/plugins.hpp"

namespace shodhan::mip {

namespace {

void register_builtin(PluginRegistry&) {}

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
