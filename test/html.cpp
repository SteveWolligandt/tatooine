#include <catch2/catch.hpp>
#include <tatooine/html.h>
namespace tatooine::test {
TEST_CASE("htmltable", "[html][table]") {
  html doc{"table_test.html"};
  doc.add_table({{"a", "b"}, {"1", "2"}}, true, false);
  doc.write();
}
TEST_CASE("htmlchart", "[html][chart]") {
  html doc{"chart_test.html"};
  doc.add_chart(std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                std::vector<std::string>{"a", "b", "c"});
  doc.write();
}
TEST_CASE("htmltablechart", "[html][table][chart]") {
  html doc{"table_chart_test.html"};
  doc.add_table({{"a", "b"}, {"1", "2"}}, true, false);
  doc.add_chart(std::vector<double>{0.0, 1.0, 3.0}, "test", "#F00FAB",
                std::vector<std::string>{"a", "b", "c"});
  doc.add_table({{"c", "d"}, {"2", "3"}});
  doc.write();
}
}  // namespace tatooine::test
