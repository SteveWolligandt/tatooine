#ifndef TATOOINE_GEOMETRY_VTK_XML_DATA_TYPE_H
#define TATOOINE_GEOMETRY_VTK_XML_DATA_TYPE_H
//==============================================================================
#include <tatooine/concepts.h>

#include <cstring>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
enum class data_type {
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
auto visit(data_type dt, auto&& f) {
  switch (dt) {
    case data_type::int8:
      if constexpr (std::invocable<decltype(f), std::int8_t>) {
        f(std::int8_t{});
      }
      break;
    case data_type::uint8:
      if constexpr (std::invocable<decltype(f), std::uint8_t>) {
        f(std::uint8_t{});
      }
      break;
    case data_type::int16:
      if constexpr (std::invocable<decltype(f), std::int16_t>) {
        f(std::int16_t{});
      }
      break;
    case data_type::uint16:
      if constexpr (std::invocable<decltype(f), std::uint16_t>) {
        f(std::uint16_t{});
      }
      break;
    case data_type::int32:
      if constexpr (std::invocable<decltype(f), std::int32_t>) {
        f(std::int32_t{});
      }
      break;
    case data_type::uint32:
      if constexpr (std::invocable<decltype(f), std::uint32_t>) {
        f(std::uint32_t{});
      }
      break;
    case data_type::int64:
      if constexpr (std::invocable<decltype(f), std::int64_t>) {
        f(std::int64_t{});
      }
      break;
    case data_type::uint64:
      if constexpr (std::invocable<decltype(f), std::uint64_t>) {
        f(std::uint64_t{});
      }
      break;
    case data_type::float32:
      if constexpr (std::invocable<decltype(f), float>) {
        f(float{});
      }
      break;
    case data_type::float64:
      if constexpr (std::invocable<decltype(f), double>) {
        f(double{});
      }
      break;
    case data_type::unknown:
    default:
      break;
  }
}
auto parse_data_type(char const* str) -> data_type;
auto to_string(data_type const t) -> std::string_view;
auto size(data_type const dt) -> std::size_t;
//------------------------------------------------------------------------------
template <typename T>
static auto constexpr to_type() {
  if constexpr (is_same<std::int8_t, T>) {
    return data_type::int8;
  } else if constexpr (is_same<T, std::uint8_t>) {
    return data_type::uint8;
  } else if constexpr (is_same<T, std::int16_t>) {
    return data_type::int16;
  } else if constexpr (is_same<T, std::uint16_t>) {
    return data_type::uint16;
  } else if constexpr (is_same<T, std::int32_t>) {
    return data_type::int32;
  } else if constexpr (is_same<T, std::uint32_t>) {
    return data_type::uint32;
  } else if constexpr (is_same<T, std::int64_t>) {
    return data_type::int64;
  } else if constexpr (is_same<T, std::uint64_t>) {
    return data_type::uint64;
  } else if constexpr (is_same<T, float>) {
    return data_type::float32;
  } else if constexpr (is_same<T, double>) {
    return data_type::float64;
  } else {
    return data_type::unknown;
  }
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
