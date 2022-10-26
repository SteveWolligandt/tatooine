#include <tatooine/vtk/xml/data_type.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
auto parse_data_type(char const* str) -> data_type {
  if (std::strcmp(str, "Int8") == 0) {
    return data_type::int8;
  }
  if (std::strcmp(str, "UInt8") == 0) {
    return data_type::uint8;
  }
  if (std::strcmp(str, "Int16") == 0) {
    return data_type::int16;
  }
  if (std::strcmp(str, "UInt16") == 0) {
    return data_type::uint16;
  }
  if (std::strcmp(str, "Int32") == 0) {
    return data_type::int32;
  }
  if (std::strcmp(str, "UInt32") == 0) {
    return data_type::uint32;
  }
  if (std::strcmp(str, "Int64") == 0) {
    return data_type::int64;
  }
  if (std::strcmp(str, "UInt64") == 0) {
    return data_type::uint64;
  }
  if (std::strcmp(str, "Float32") == 0) {
    return data_type::float32;
  }
  if (std::strcmp(str, "Float64") == 0) {
    return data_type::float64;
  }
  return data_type::unknown;
}
//------------------------------------------------------------------------------
auto size(data_type const dt) -> std::size_t {
  switch (dt) {
    case data_type::int8:
      return sizeof(std::int8_t);
    case data_type::uint8:
      return sizeof(std::uint8_t);
    case data_type::int16:
      return sizeof(std::int16_t);
    case data_type::uint16:
      return sizeof(std::uint16_t);
    case data_type::int32:
      return sizeof(std::int32_t);
    case data_type::uint32:
      return sizeof(std::uint32_t);
    case data_type::int64:
      return sizeof(std::int64_t);
    case data_type::uint64:
      return sizeof(std::uint64_t);
    case data_type::float32:
      return sizeof(float);
    case data_type::float64:
      return sizeof(double);
    case data_type::unknown:
    default:
      return 0;
  }
}
//------------------------------------------------------------------------------
auto to_string(data_type const t) -> std::string_view {
  switch (t) {
    case data_type::int8:
      return "Int8";
    case data_type::uint8:
      return "UInt8";
    case data_type::int16:
      return "Int16";
    case data_type::uint16:
      return "UInt16";
    case data_type::int32:
      return "Int32";
    case data_type::uint32:
      return "UInt32";
    case data_type::int64:
      return "Int64";
    case data_type::uint64:
      return "UInt64";
    case data_type::float32:
      return "Float32";
    case data_type::float64:
      return "Float64";
    default:
      return "UnknownType";
  }
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
