#ifndef TATOOINE_GEOMETRY_VTK_XML_DATA_ARRAY_H
#define TATOOINE_GEOMETRY_VTK_XML_DATA_ARRAY_H
//==============================================================================
#include <tatooine/type_traits.h>
#include <tatooine/vtk/xml/data_type.h>
#include <tatooine/vtk/xml/format.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <rapidxml.hpp>
#include <string>
#include <vector>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader;
struct data_array {
  //==============================================================================
 private:
  data_type                  m_type           = data_type::unknown;
  std::optional<std::string> m_name           = {};
  std::size_t                m_num_components = 1;
  xml::format                m_format         = format::unknown;
  std::size_t                m_offset = std::numeric_limits<std::size_t>::max();
  reader*                    m_reader = nullptr;
  rapidxml::xml_node<>*      m_node   = nullptr;
  //==============================================================================
 public:
  data_array() = default;
  data_array(reader& r, rapidxml::xml_node<>* node);
  data_array(data_array const&)                    = default;
  auto operator=(data_array const&) -> data_array& = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto type() const { return m_type; }
  [[nodiscard]] auto name() const { return m_name; }
  [[nodiscard]] auto num_components() const { return m_num_components; }
  [[nodiscard]] auto format() const { return m_format; }
  [[nodiscard]] auto offset() const { return m_offset; }
  //----------------------------------------------------------------------------
  auto visit_data(auto&& f) const {
    switch (m_type) {
      case data_type::int8:
        if constexpr (std::invocable<decltype(f), std::vector<std::int8_t>>) {
          f(read<std::int8_t>());
        }
        break;
      case data_type::uint8:
        if constexpr (std::invocable<decltype(f), std::vector<std::uint8_t>>) {
          f(read<std::uint8_t>());
        }
        break;
      case data_type::int16:
        if constexpr (std::invocable<decltype(f), std::vector<std::int16_t>>) {
          f(read<std::int16_t>());
        }
        break;
      case data_type::uint16:
        if constexpr (std::invocable<decltype(f), std::vector<std::uint16_t>>) {
          f(read<std::uint16_t>());
        }
        break;
      case data_type::int32:
        if constexpr (std::invocable<decltype(f), std::vector<std::int32_t>>) {
          f(read<std::int32_t>());
        }
        break;
      case data_type::uint32:
        if constexpr (std::invocable<decltype(f), std::vector<std::uint32_t>>) {
          f(read<std::uint32_t>());
        }
        break;
      case data_type::int64:
        if constexpr (std::invocable<decltype(f), std::vector<std::int64_t>>) {
          f(read<std::int64_t>());
        }
        break;
      case data_type::uint64:
        if constexpr (std::invocable<decltype(f), std::vector<std::uint64_t>>) {
          f(read<std::uint64_t>());
        }
        break;
      case data_type::float32:
        if constexpr (std::invocable<decltype(f), std::vector<float>>) {
          f(read<float>());
        }
        break;
      case data_type::float64:
        if constexpr (std::invocable<decltype(f), std::vector<double>>) {
          f(read<double>());
        }
        break;
      case data_type::unknown:
        throw std::runtime_error {
          "[vtk::xml::data_array] could not visit data because data_type is "
          "unknown."
        };
      default:
        throw std::runtime_error {
          "[vtk::xml::data_array] could not visit data because no function "
          "matches."
        };
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto read() const -> std::vector<T> {
    switch (format()) {
      case xml::format::ascii:
        return read_data_ascii<T>();
      case xml::format::binary:
        return read_data_binary<T>();
      case xml::format::appended:
        return read_data_appended<T>();
      case xml::format::unknown:
      default:
        return {};
    }
  }
  //----------------------------------------------------------------------------
 private:
  template <typename T>
  auto read_data_ascii() const {
    throw std::runtime_error{"[vtk::xml::data_array] cannot read ascii"};
    auto data = std::vector<T>{};
    return data;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto read_data_binary() const {
    throw std::runtime_error{"[vtk::xml::data_array] cannot read binary"};
    auto data = std::vector<T>{};
    return data;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto read_data_appended() const {
    auto const num_bytes = read_appended_data_size();
    auto       data      = std::vector<T>(num_bytes / sizeof(T));
    read_appended_data(reinterpret_cast<char*>(data.data()), num_bytes);
    return data;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto read_appended_data() const {
    auto const num_bytes = read_appended_data_size();
    auto       data      = std::vector<T>(num_bytes / sizeof(T));
    read_appended_data(reinterpret_cast<char*>(data.data()), num_bytes);
    return data;
  }
  auto read_appended_data(char* data, std::size_t num_bytes) const -> void;
  auto read_appended_data_size() const -> std::size_t;
  ////------------------------------------------------------------------------------
  // template <typename T, std::size_t N, typename F>
  // auto read_appended_data(std::uint8_t const* data_begin, F&& f) {
  //   if constexpr (N > 1) {
  //     f(reinterpret_cast<std::array<T, N> const*>(data_begin));
  //   } else {
  //     f(reinterpret_cast<T const*>(data_begin));
  //   }
  // }
  ////------------------------------------------------------------------------------
  // template <typename T, typename F>
  // auto read_appended_data(std::uint8_t const* data_begin,
  //                         std::size_t const num_components, F&& f) {
  //   switch (num_components) {
  //     case 1:
  //       read_appended_data<T, 1>(data_begin, std::forward<F>(f));
  //       break;
  //     case 2:
  //       read_appended_data<T, 2>(data_begin, std::forward<F>(f));
  //       break;
  //     case 3:
  //       read_appended_data<T, 3>(data_begin, std::forward<F>(f));
  //       break;
  //     case 4:
  //       read_appended_data<T, 4>(data_begin, std::forward<F>(f));
  //       break;
  //   }
  // }
  ////------------------------------------------------------------------------------
  // template <typename F>
  // auto read_appended_data(std::uint8_t const* data_begin,
  //                         data_array const& meta, F&& f) {
  //   switch (meta.type()) {
  //     case data_array::type_t::int8:
  //       read_appended_data<std::int8_t>(data_begin, meta.num_components(),
  //                                       std::forward<F>(f));
  //       break;
  //     case data_array::type_t::uint8:
  //       read_appended_data<std::uint8_t>(data_begin, meta.num_components(),
  //                                        std::forward<F>(f));
  //       break;
  //     case data_array::type_t::int16:
  //       read_appended_data<std::int16_t>(data_begin, meta.num_components(),
  //                                        std::forward<F>(f));
  //       break;
  //     case data_array::type_t::uint16:
  //       read_appended_data<std::uint16_t>(data_begin, meta.num_components(),
  //                                         std::forward<F>(f));
  //       break;
  //     case data_array::type_t::int32:
  //       read_appended_data<std::int32_t>(data_begin, meta.num_components(),
  //                                        std::forward<F>(f));
  //       break;
  //     case data_array::type_t::uint32:
  //       read_appended_data<std::uint32_t>(data_begin, meta.num_components(),
  //                                         std::forward<F>(f));
  //       break;
  //     case data_array::type_t::int64:
  //       read_appended_data<std::int64_t>(data_begin, meta.num_components(),
  //                                        std::forward<F>(f));
  //       break;
  //     case data_array::type_t::uint64:
  //       read_appended_data<std::uint64_t>(data_begin, meta.num_components(),
  //                                         std::forward<F>(f));
  //       break;
  //     case data_array::type_t::float32:
  //       read_appended_data<float>(data_begin, meta.num_components(),
  //                                 std::forward<F>(f));
  //       break;
  //     case data_array::type_t::float64:
  //       read_appended_data<double>(data_begin, meta.num_components(),
  //                                  std::forward<F>(f));
  //       break;
  //     case data_array::type_t::unknown:
  //     default:
  //       break;
  //   }
  // }
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
