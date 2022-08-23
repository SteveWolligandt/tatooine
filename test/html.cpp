#include <catch2/catch_test_macros.hpp>
#include <tatooine/html.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("htmltable", "[html][table]") {
  const std::string outpath = "table_test.html";
  html::doc         doc;
  doc.add(html::table{std::vector{"a", "b"}, std::vector{"1", "2"}});
  doc.write(outpath);
}
//==============================================================================
TEST_CASE("htmlchart", "[html][chart]") {
  const std::string outpath = "chart_test.html";
  html::doc doc;
  doc.add(html::chart{std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                      std::vector<std::string>{"a", "b", "c"}});
  doc.write(outpath);
}
//==============================================================================
TEST_CASE("htmltablechart", "[html][table][chart]") {
  const std::string outpath = "table_chart_test.html";
  html::doc doc;
  doc.add(html::table{std::vector{"a", "b"}, std::vector{"1", "2"}});
  doc.add(html::chart{std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                std::vector<std::string>{"a", "b", "c"}});
  doc.add(html::table{std::vector{"c", "d"}, std::vector{"2", "3"}});
  doc.write(outpath);
}
//==============================================================================
TEST_CASE("htmlimage", "[html][image]") {
  const std::string outpath = "image_test.html";
  html::doc doc;
  doc.add(
      html::image{"https://upload.wikimedia.org/wikipedia/commons/c/c6/"
                  "Amadeo_king_of_Spain.jpg"});
  doc.write(outpath);
}
//==============================================================================
TEST_CASE("htmltableimage", "[html][table][image]") {
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  const std::string outpath = "table_image_test.html";
  html::doc doc;
  doc.add(html::image{path}, html::image{path});
  doc.write(outpath);
}
//==============================================================================
TEST_CASE("htmltableimagechart", "[html][table][image][chart]") {
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  const std::string outpath = "table_image_chart_test.html";
  html::doc         doc;
  doc.add(html::image{path},
          html::chart{std::vector{0.0, 1.0, 3.0}, "test", "#F00FAB",
                      std::vector<std::string>{"a", "b", "c"}});
  doc.write(outpath);
}
//==============================================================================
TEST_CASE("htmlslider", "[html][slider][image][chart]") {
  using namespace html;
  const std::string path =
      "https://upload.wikimedia.org/wikipedia/commons/c/c6/"
      "Amadeo_king_of_Spain.jpg";
  const std::string path2 =
      "https://upload.wikimedia.org/wikipedia/commons/0/0c/"
      "Maria_Anna_Austria_1770_1809_young.jpg";
  const std::string outpath = "slider_image_test.html";

  doc doc;
  doc.add_slider(
      vbox{heading{"Fürst von Meerrettich"},
                 hbox{image{path}, image{path2}}},
      vbox{heading{"Wie ein Fisch... also wie ein Opfer..."},
                 image{path2}});
  doc.add(heading{"Foo"}, "fifafoo", "lirum larum Löffelstiel");
  doc.add("Bar", "bliblablo");
  doc.add(heading{"Bloob"},
          chart{std::vector{0.0, 1.0, 3.0}, "test", "#F00FAB",
                      std::vector{1, 2, 3}});
  doc.add(heading{"important data"},
          table{std::vector{"a", "b"}, std::vector{1, 2}});
  doc.write(outpath);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
