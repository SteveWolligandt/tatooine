#include <tatooine/vtk/xml/byte_order.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
auto to_byte_order(char const* str) -> byte_order {
  if (std::strcmp(str, "LittleEndian") == 0) {
    return byte_order::little_endian;
  }
  if (std::strcmp(str, "BigEndian") == 0) {
    return byte_order::big_endian;
  }
  return byte_order::unknown;
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
