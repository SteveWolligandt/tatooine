#include <tatooine/line.h>
#include <tatooine/vtk/xml.h>

#include <catch2/catch_test_macros.hpp>
#include <iostream>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("vtk_xml_read_structured_grid", "[vtk][xml][read]") {
  SECTION("poly mesh with embedded ascii data") {
    auto reader = vtk::xml::reader{"poly_data.vtp"};
    REQUIRE(reader.type() == vtk::xml::vtk_type::poly_data);
    REQUIRE(reader.version() == "0.1");
    REQUIRE(reader.byte_order() == vtk::xml::byte_order::little_endian);
    REQUIRE(reader.poly_data().has_value());
    auto const& poly_data = *reader.poly_data();
    REQUIRE(poly_data.pieces.size() == 1);
    auto const& piece = poly_data.pieces.front();
    REQUIRE(piece.num_points == 8);
    REQUIRE(piece.num_polygons == 6);
    REQUIRE(piece.num_vertices == 0);
    REQUIRE(piece.num_lines == 0);
    REQUIRE(piece.num_strips == 0);
    REQUIRE(piece.polygons.size() == 2);
    REQUIRE(piece.point_data.size() == 1);
    REQUIRE(piece.cell_data.size() == 2);
    REQUIRE(piece.points.type() == vtk::xml::data_type::float32);
  }
  SECTION("poly line with appended data") {
    SECTION("write line") {
      auto l = line3{};

      auto  v0   = l.push_back(1, 1, 1);
      auto  v1   = l.push_back(2, 2, 2);
      auto  v2   = l.push_back(3, 3, 3);
      auto& prop = l.scalar_vertex_property("prop");
      auto& prop2 = l.vec2_vertex_property("prop2");
      prop[v0]   = 5;
      prop[v1]   = 4;
      prop[v2]   = 3;
      prop2[v1]   = vec2{1,2};
      l.write("line.vtp");
    }
    SECTION("read line") {
      auto reader = vtk::xml::reader{"line.vtp"};
      REQUIRE(reader.type() == vtk::xml::vtk_type::poly_data);
      REQUIRE(reader.version() == "1.0");
      REQUIRE(reader.byte_order() == vtk::xml::byte_order::little_endian);
      REQUIRE(reader.poly_data().has_value());
      auto const& poly_data = *reader.poly_data();
      REQUIRE(poly_data.pieces.size() == 1);
      auto const& piece = poly_data.pieces.front();
      REQUIRE(piece.num_points == 3);
      REQUIRE(piece.num_polygons == 0);
      REQUIRE(piece.num_vertices == 0);
      REQUIRE(piece.num_lines == 2);
      REQUIRE(piece.num_strips == 0);
      REQUIRE(piece.polygons.size() == 0);
      REQUIRE(piece.point_data.size() == 2);
      REQUIRE(piece.cell_data.size() == 0);
      REQUIRE(piece.points.type() == vtk::xml::data_type::float64);
      piece.points.visit_data([](std::vector<double> const& points_data) {
        REQUIRE(points_data.size() == 9);
        CHECK(points_data[0] == 1);
        CHECK(points_data[1] == 1);
        CHECK(points_data[2] == 1);
        CHECK(points_data[3] == 2);
        CHECK(points_data[4] == 2);
        CHECK(points_data[5] == 2);
        CHECK(points_data[6] == 3);
        CHECK(points_data[7] == 3);
        CHECK(points_data[8] == 3);
      });

      piece.point_data.at("prop").visit_data([](std::vector<double> const& data) {
        REQUIRE(data.size() == 3);
        REQUIRE(data[0] == 5);
        REQUIRE(data[1] == 4);
        REQUIRE(data[2] == 3);
      });
      piece.point_data.at("prop2").visit_data([](std::vector<double> const& data) {
        REQUIRE(data.size() == 6);
        REQUIRE(data[0] == 0);
        REQUIRE(data[1] == 0);
        REQUIRE(data[2] == 1);
        REQUIRE(data[3] == 2);
        REQUIRE(data[4] == 0);
        REQUIRE(data[5] == 0);
      });

      auto read_line = line3::read_vtp("line.vtp");
      REQUIRE(read_line.num_vertices() == 3);
      REQUIRE(read_line.vertex_properties().size() == 2);
      REQUIRE(read_line.vertex_properties().contains("prop"));
      auto& p = read_line.scalar_vertex_property("prop");
      REQUIRE(p[line3::vertex_handle{0}] == 5);
      REQUIRE(p[line3::vertex_handle{1}] == 4);
      REQUIRE(p[line3::vertex_handle{2}] == 3);
      auto& p2 = read_line.vec2_vertex_property("prop2");
      REQUIRE(p2[line3::vertex_handle{0}](0) == 0);
      REQUIRE(p2[line3::vertex_handle{0}](1) == 0);
      REQUIRE(p2[line3::vertex_handle{1}](0) == 1);
      REQUIRE(p2[line3::vertex_handle{1}](1) == 2);
      REQUIRE(p2[line3::vertex_handle{2}](0) == 0);
      REQUIRE(p2[line3::vertex_handle{2}](1) == 0);
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
