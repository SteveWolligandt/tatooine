#ifndef TATOOINE_VTK_XML_BYTE_ORDER_H
#define TATOOINE_VTK_XML_BYTE_ORDER_H
//==============================================================================
#include <cstring>
//==============================================================================
namespace tatooine::vtk::xml{
//==============================================================================
enum class byte_order { little_endian, big_endian, unknown };
auto to_byte_order(char const* str) -> byte_order;
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
