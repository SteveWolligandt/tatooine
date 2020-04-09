#include <catch2/catch.hpp>
#include <tatooine/html.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("htmltable", "[html][table]") {
  html::doc doc{"table_test.html"};
  doc.add(html::table{{{"a", "b"}, {"1", "2"}}, true, false});
  doc.write();
}
//==============================================================================
TEST_CASE("htmlchart", "[html][chart]") {
  html::doc doc{"chart_test.html"};
  doc.add(html::chart{std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                      std::vector<std::string>{"a", "b", "c"}});
  doc.write();
}
//==============================================================================
TEST_CASE("htmltablechart", "[html][table][chart]") {
  html::doc doc{"table_chart_test.html"};
  doc.add(html::table{{{"a", "b"}, {"1", "2"}}, true, false});
  doc.add(html::chart{std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                std::vector<std::string>{"a", "b", "c"}});
  doc.add(html::table{{{"c", "d"}, {"2", "3"}}});
  doc.write();
}
//==============================================================================
TEST_CASE("htmlimage", "[html][image]") {
  html::doc doc{"image_test.html"};
  doc.add(
      html::image{"https://upload.wikimedia.org/wikipedia/commons/c/c6/"
                  "Amadeo_king_of_Spain.jpg"});
  doc.write();
}
//==============================================================================
TEST_CASE("htmltableimage", "[html][table][image]") {
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  html::doc doc{"table_image_test.html"};
  doc.add(html::image{path}, html::image{path});
  doc.write();
}
//==============================================================================
TEST_CASE("htmltableimagechart", "[html][table][image][chart]") {
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  html::doc doc{"table_image_chart_test.html"};
  doc.add(html::image{path},
          html::chart{std::vector{0.0, 1.0, 3.0}, "test", "#F00FAB",
                      std::vector<std::string>{"a", "b", "c"}});
  doc.write();
}
//==============================================================================
TEST_CASE("htmlslider", "[html][slider][image][chart]") {
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  const std::string path2 =
      "https://upload.wikimedia.org/wikipedia/commons/0/0c/"
      "Maria_Anna_Austria_1770_1809_young.jpg";
  html::doc doc{"slider_image_test.html"};
  doc.add_slider(
      html::horizontal_container{
          "Fürst von Meerrettich",
          html::vertical_container{html::image{path}, html::image{path2}}},
      html::horizontal_container{"Wie ein Fisch... also wie ein Opfer...",
                                 html::image{path2}});
  doc.add("Foo");
  doc.add("Bar");
  doc.add(html::horizontal_container{
      "Bloob", html::chart{std::vector{0.0, 1.0, 3.0}, "test", "#F00FAB",
                           std::vector<std::string>{"a", "b", "c"}}});
  doc.write();
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
