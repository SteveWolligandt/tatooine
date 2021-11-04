#ifndef TATOOINE_VTK_XML_LISTENER_H
#define TATOOINE_VTK_XML_LISTENER_H
//==============================================================================
#include <tatooine/vtk/xml/byte_order.h>

#include <array>
#include <cstdint>
#include <string>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct listener {
  virtual auto on_vtk_type(char const*) -> void{};
  virtual auto on_vtk_version(char const*) -> void{};
  virtual auto on_vtk_byte_order(byte_order) -> void{};

  virtual auto on_points(std::int8_t const*) -> void {}
  virtual auto on_points(std::uint8_t const*) -> void {}
  virtual auto on_points(std::int16_t const*) -> void {}
  virtual auto on_points(std::uint16_t const*) -> void {}
  virtual auto on_points(std::int32_t const*) -> void {}
  virtual auto on_points(std::uint32_t const*) -> void {}
  virtual auto on_points(std::int64_t const*) -> void {}
  virtual auto on_points(std::uint64_t const*) -> void {}
  virtual auto on_points(float const*) -> void {}
  virtual auto on_points(double const*) -> void {}

  virtual auto on_points(std::array<std::int8_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::uint8_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::int16_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::uint16_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::int32_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::uint32_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::int64_t, 2> const*) -> void {}
  virtual auto on_points(std::array<std::uint64_t, 2> const*) -> void {}
  virtual auto on_points(std::array<float, 2> const*) -> void {}
  virtual auto on_points(std::array<double, 2> const*) -> void {}

  virtual auto on_points(std::array<std::int8_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::uint8_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::int16_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::uint16_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::int32_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::uint32_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::int64_t, 3> const*) -> void {}
  virtual auto on_points(std::array<std::uint64_t, 3> const*) -> void {}
  virtual auto on_points(std::array<float, 3> const*) -> void {}
  virtual auto on_points(std::array<double, 3> const*) -> void {}

  virtual auto on_points(std::array<std::int8_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::uint8_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::int16_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::uint16_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::int32_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::uint32_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::int64_t, 4> const*) -> void {}
  virtual auto on_points(std::array<std::uint64_t, 4> const*) -> void {}
  virtual auto on_points(std::array<float, 4> const*) -> void {}
  virtual auto on_points(std::array<double, 4> const*) -> void {}

  virtual auto on_point_data(std::string const& name, std::int8_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::uint8_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::int16_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::uint16_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::int32_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::uint32_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::int64_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::uint64_t const*) -> void {}
  virtual auto on_point_data(std::string const& name, float const*) -> void {}
  virtual auto on_point_data(std::string const& name, double const*) -> void {}

  virtual auto on_point_data(std::string const& name, std::array<std::int8_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint8_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int16_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint16_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int32_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint32_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int64_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint64_t, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<float, 2> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<double, 2> const*) -> void {}

  virtual auto on_point_data(std::string const& name, std::array<std::int8_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint8_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int16_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint16_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int32_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint32_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int64_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint64_t, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<float, 3> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<double, 3> const*) -> void {}

  virtual auto on_point_data(std::string const& name, std::array<std::int8_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint8_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int16_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint16_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int32_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint32_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::int64_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<std::uint64_t, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<float, 4> const*) -> void {}
  virtual auto on_point_data(std::string const& name, std::array<double, 4> const*) -> void {}

  virtual auto on_structured_grid(
      std::array<std::pair<std::size_t, std::size_t>, 3> const&) -> void {}
  virtual auto on_structured_grid_piece(
      std::array<std::pair<std::size_t, std::size_t>, 3> const&) -> void {}
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
