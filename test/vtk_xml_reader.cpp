#include <tatooine/vtk/xml.h>
#include <catch2/catch.hpp>
#include <iostream>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("vtk_xml_read_structured_grid", "[vtk][xml][read][structured_grid]"){
  struct listener_t : vtk::xml::listener {
    auto on_vtk_version(char const* v) -> void override {
      std::cout << v << '\n';
    }
    auto on_points(std::array<double, 3> const* v) -> void override {
      std::cout << v->at(0) << '\n';
      std::cout << v->at(1) << '\n';
      std::cout << v->at(2) << '\n';
    }
    auto on_point_data(std::string const& name, float const* v) -> void override {
      std::cout << name << ' ' << *v << '\n';
    }
  } listener;
  auto reader =
      vtk::xml::reader{"/home/steve/firetec/valley_losAlamos/output.1000.vts"};
  reader.listen(listener);
  reader.read();
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
