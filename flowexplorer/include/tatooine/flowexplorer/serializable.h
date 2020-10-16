#ifndef TATOOINE_FLOWEXPLORER_SERIALIZABLE_H
#define TATOOINE_FLOWEXPLORER_SERIALIZABLE_H
//==============================================================================
#include <toml++/toml.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct serializable {
  virtual auto serialize() const -> toml::table = 0;
  virtual void deserialize(toml::table const&) = 0;
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
