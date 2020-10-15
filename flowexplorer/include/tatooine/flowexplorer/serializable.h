#ifndef TATOOINE_FLOWEXPLORER_SERIALIZABLE_H
#define TATOOINE_FLOWEXPLORER_SERIALIZABLE_H
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct serializable {
  virtual void serialize()   = 0;
  virtual void deserialize() = 0;
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
