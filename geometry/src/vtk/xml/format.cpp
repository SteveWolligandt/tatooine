#include <tatooine/vtk/xml/format.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
auto parse_format(char const* str) -> format {
  if (std::strcmp(str, "ascii") == 0) {
    return format::ascii;
  }
  if (std::strcmp(str, "binary") == 0) {
    return format::binary;
  }
  if (std::strcmp(str, "appended") == 0) {
    return format::appended;
  }
  return format::unknown;
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
