#include <tatooine/analytical/numerical/doublegyre.h>
#include <catch2/catch_test_macros.hpp>
//═════════════════════════════════════════════════════════════════════════════╗
namespace tatooine::test {
//═══════════════════════════════════════════════════════════════════════════╗
TEST_CASE("field_list0", "[field_operations]") {
  vectorfield_list<double, 2> vs;
  vs.push_back(
      std::make_unique<analytical::numerical::doublegyre<double>>());
  vs.push_back(
      std::make_unique<analytical::numerical::doublegyre<double>>());

  for (const auto& v : vs) {
    std::cerr << v->evaluate({0.1, 0.1}, 1) << '\n';
  }
}
//═══════════════════════════════════════════════════════════════════════════╝
}  // namespace tatooine::test
//═════════════════════════════════════════════════════════════════════════════╝
