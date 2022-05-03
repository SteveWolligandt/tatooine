#ifndef TATOOINE_VTK_XML_DATA_ARRAY_H
#define TATOOINE_VTK_XML_DATA_ARRAY_H
//==============================================================================
#include <cstring>
#include <cstdint>
#include <tatooine/type_traits.h>
#include <limits>
#include <rapidxml.hpp>
#include <string>
//==============================================================================
namespace tatooine::vtk::xml {
struct data_array {
  //==============================================================================
  enum class type_t {
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
    unknown
  };
  //------------------------------------------------------------------------------
  static auto to_type(char const* str) {
    if (std::strcmp(str, "Int8") == 0) {
      return type_t::int8;
    }
    if (std::strcmp(str, "UInt8") == 0) {
      return type_t::uint8;
    }
    if (std::strcmp(str, "Int16") == 0) {
      return type_t::int16;
    }
    if (std::strcmp(str, "UInt16") == 0) {
      return type_t::uint16;
    }
    if (std::strcmp(str, "Int32") == 0) {
      return type_t::int32;
    }
    if (std::strcmp(str, "UInt32") == 0) {
      return type_t::uint32;
    }
    if (std::strcmp(str, "Int64") == 0) {
      return type_t::int64;
    }
    if (std::strcmp(str, "UInt64") == 0) {
      return type_t::uint64;
    }
    if (std::strcmp(str, "Float32") == 0) {
      return type_t::float32;
    }
    if (std::strcmp(str, "Float64") == 0) {
      return type_t::float64;
    }
    return type_t::unknown;
  }
  //------------------------------------------------------------------------------
  template <typename T>
  static auto constexpr to_type() {
    if constexpr (is_same<std::int8_t, T>) {
      return type_t::int8;
    } else if constexpr (is_same<T, std::uint8_t>) {
      return type_t::uint8;
    } else if constexpr (is_same<T, std::int16_t>) {
      return type_t::int16;
    } else if constexpr (is_same<T, std::uint16_t>) {
      return type_t::uint16;
    } else if constexpr (is_same<T, std::int32_t>) {
      return type_t::int32;
    } else if constexpr (is_same<T, std::uint32_t>) {
      return type_t::uint32;
    } else if constexpr (is_same<T, std::int64_t>) {
      return type_t::int64;
    } else if constexpr (is_same<T, std::uint64_t>) {
      return type_t::uint64;
    } else if constexpr (is_same<T, float>) {
      return type_t::float32;
    } else if constexpr (is_same<T, double>) {
      return type_t::float64;
    } else {
      return type_t::unknown;
    }
  }
  //------------------------------------------------------------------------------
  static auto constexpr to_string(type_t const t) -> std::string_view {
    switch (t) {
      case type_t::int8:
        return "Int8";
      case type_t::uint8:
        return "UInt8";
      case type_t::int16:
        return "Int16";
      case type_t::uint16:
        return "UInt16";
      case type_t::int32:
        return "Int32";
      case type_t::uint32:
        return "UInt32";
      case type_t::int64:
        return "Int64";
      case type_t::uint64:
        return "UInt64";
      case type_t::float32:
        return "Float32";
      case type_t::float64:
        return "Float64";
      default:
        return "UnknownType";
    }
  }
  //==============================================================================
  enum class format_t { ascii, binary, appended, unknown };
  //------------------------------------------------------------------------------
  static auto to_format(char const* str) {
    if (std::strcmp(str, "ascii") == 0) {
      return format_t::ascii;
    }
    if (std::strcmp(str, "binary") == 0) {
      return format_t::binary;
    }
    if (std::strcmp(str, "appended") == 0) {
      return format_t::appended;
    }
    return format_t::unknown;
  }
  //==============================================================================
 private:
  type_t      m_type = type_t::unknown;
  std::string m_name;
  std::size_t m_num_components = 1;
  format_t    m_format;
  std::size_t m_offset = std::numeric_limits<std::size_t>::max();
  //==============================================================================
 public:
  data_array(rapidxml::xml_node<>* node)
      : m_type{to_type(node->first_attribute("type")->value())},
        m_name{node->first_attribute("Name")->value()},
        m_format{to_format(node->first_attribute("format")->value())} {
    if (m_format == format_t::appended) {
      m_offset = std::stoul(node->first_attribute("offset")->value());
    }
    if (auto* n = node->first_attribute("NumberOfComponents"); n != nullptr) {
      m_num_components = std::stoul(n->value());
    }
  }
  //----------------------------------------------------------------------------
  auto type() const { return m_type; }
  auto name() const { return m_name; }
  auto num_components() const { return m_num_components; }
  auto format() const { return m_format; }
  auto offset() const {return m_offset;}
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
