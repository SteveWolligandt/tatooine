#ifndef TATOOINE_DEMANGLING_H
#define TATOOINE_DEMANGLING_H
//==============================================================================
#include <boost/core/demangle.hpp>
//==============================================================================
namespace tatooine{
//==============================================================================
inline auto type_name(std::type_info const& t) -> std::string {
  return boost::core::demangle(t.name());
}
//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline auto type_name(T && /*t*/) -> std::string {
  return boost::core::demangle(typeid(T).name());
}
//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline auto type_name() -> std::string {
  return boost::core::demangle(typeid(T).name());
}
//------------------------------------------------------------------------------
/// returns demangled typename
template <typename T>
inline auto type_name(std::string const& name) -> std::string {
  return boost::core::demangle(name.c_str());
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
