#ifndef TATOOINE_VTK_XML_READER_H
#define TATOOINE_VTK_XML_READER_H
//==============================================================================
#include <tatooine/filesystem.h>
#include <tatooine/vtk/xml/byte_order.h>
#include <tatooine/vtk/xml/listener.h>
#include <tatooine/vtk/xml/data_attribute.h>
#include <tatooine/vtk/xml/data_array.h>

#include <array>
#include <cstring>
#include <fstream>
#include <rapidxml.hpp>
#include <sstream>
#include <string>
#include <vector>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader {
  //==============================================================================
  // MEMBERS
  //==============================================================================
  std::vector<listener*> m_listeners;
  std::vector<std::uint8_t> m_appended_data;
  filesystem::path          m_path;
  //==============================================================================
  // CTORS
  //==============================================================================
  reader(filesystem::path const& path) : m_path{path} {}
  //==============================================================================
  // STATIC METHODS
  //==============================================================================
  static auto extract_appended_data(std::string& content)
      -> std::vector<std::uint8_t>;
  //------------------------------------------------------------------------------
  static auto extract_extents(rapidxml::xml_attribute<> const* attr)
      -> std::array<std::pair<size_t, size_t>, 3>;
  //==============================================================================
  // METHODS
  //==============================================================================
  auto listen(listener& l) -> void { m_listeners.push_back(&l); }
  //------------------------------------------------------------------------------
  auto read() -> void;
  //------------------------------------------------------------------------------
  auto read_points(rapidxml::xml_node<>* node)->void;
  auto read_point_data(rapidxml::xml_node<>* node)->void;
  auto read_cell_data(rapidxml::xml_node<>* node)->void;
  //------------------------------------------------------------------------------
  template <typename T, std::size_t N, typename F>
  auto read_appended_data(std::uint8_t const* data_begin, F&& f) {
    if constexpr (N > 1) {
      f(reinterpret_cast<std::array<T, N> const*>(data_begin));
    } else {
      f(reinterpret_cast<T const*>(data_begin));
    }
  }
  //------------------------------------------------------------------------------
  template <typename T, typename F>
  auto read_appended_data(std::uint8_t const* data_begin,
                          std::size_t const num_components, F&& f) {
    switch (num_components) {
      case 1:
        read_appended_data<T, 1>(data_begin, std::forward<F>(f));
        break;
      case 2:
        read_appended_data<T, 2>(data_begin, std::forward<F>(f));
        break;
      case 3:
        read_appended_data<T, 3>(data_begin, std::forward<F>(f));
        break;
      case 4:
        read_appended_data<T, 4>(data_begin, std::forward<F>(f));
        break;
    }
  }
  //------------------------------------------------------------------------------
  template <typename F>
  auto read_appended_data(std::uint8_t const* data_begin,
                          data_array const& meta, F&& f) {
    switch (meta.type()) {
      case data_array::type_t::int8:
        read_appended_data<std::int8_t>(data_begin, meta.num_components(),
                                        std::forward<F>(f));
        break;
      case data_array::type_t::uint8:
        read_appended_data<std::uint8_t>(data_begin, meta.num_components(),
                                         std::forward<F>(f));
        break;
      case data_array::type_t::int16:
        read_appended_data<std::int16_t>(data_begin, meta.num_components(),
                                         std::forward<F>(f));
        break;
      case data_array::type_t::uint16:
        read_appended_data<std::uint16_t>(data_begin, meta.num_components(),
                                          std::forward<F>(f));
        break;
      case data_array::type_t::int32:
        read_appended_data<std::int32_t>(data_begin, meta.num_components(),
                                         std::forward<F>(f));
        break;
      case data_array::type_t::uint32:
        read_appended_data<std::uint32_t>(data_begin, meta.num_components(),
                                          std::forward<F>(f));
        break;
      case data_array::type_t::int64:
        read_appended_data<std::int64_t>(data_begin, meta.num_components(),
                                         std::forward<F>(f));
        break;
      case data_array::type_t::uint64:
        read_appended_data<std::uint64_t>(data_begin, meta.num_components(),
                                          std::forward<F>(f));
        break;
      case data_array::type_t::float32:
        read_appended_data<float>(data_begin, meta.num_components(),
                                  std::forward<F>(f));
        break;
      case data_array::type_t::float64:
        read_appended_data<double>(data_begin, meta.num_components(),
                                   std::forward<F>(f));
        break;
      case data_array::type_t::unknown:
      default:
        break;
    }
  }
  //------------------------------------------------------------------------------
  auto read_structured_grid(rapidxml::xml_node<> const* node) -> void;
  //------------------------------------------------------------------------------
  auto read_data_array_header(rapidxml::xml_node<> const* node) -> void;
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
