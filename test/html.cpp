#include <catch2/catch.hpp>
#include <tatooine/html.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("htmltable", "[html][table]") {
  html doc{"table_test.html"};
  doc.add_table({{"a", "b"}, {"1", "2"}}, true, false);
  doc.write();
}
//==============================================================================
TEST_CASE("htmlchart", "[html][chart]") {
  html doc{"chart_test.html"};
  doc.add_chart(std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                std::vector<std::string>{"a", "b", "c"});
  doc.write();
}
//==============================================================================
TEST_CASE("htmltablechart", "[html][table][chart]") {
  html doc{"table_chart_test.html"};
  doc.add_table({{"a", "b"}, {"1", "2"}}, true, false);
  doc.add_chart(std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                std::vector<std::string>{"a", "b", "c"});
  doc.add_table({{"c", "d"}, {"2", "3"}});
  doc.write();
}
//==============================================================================
TEST_CASE("htmlimage", "[html][image]") {
  html doc{"image_test.html"};
  doc.add_image("https://upload.wikimedia.org/wikipedia/commons/c/c6/Amadeo_king_of_Spain.jpg");
  doc.write();
}
//==============================================================================
TEST_CASE("htmltableimage", "[html][table][image]") {
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  html doc{"table_image_test.html"};
  doc.add_side_by_side(html::image(path), html::image(path));
  doc.write();
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
