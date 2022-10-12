#include <tatooine/linspace.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <catch2/matchers/catch_matchers_all.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace Catch;
TEST_CASE("linspace", "[linspace]") {
  auto lin = linspace{0.0, 1.0, 11};
  using linspace_type = decltype(lin);
  SECTION("indexing") {
    REQUIRE(lin[0] == 0.0);
    REQUIRE(lin[1] == Approx(0.1));
    REQUIRE(lin[10] == 1.0);
  }
  SECTION("iterators") {
    SECTION("iterator_traits"){
      REQUIRE(is_same<
              std::iterator_traits<linspace_type::iterator>::iterator_category,
              std::random_access_iterator_tag>);
    }
    SECTION("navigation") {
      auto it = begin(lin);
      REQUIRE(*it == 0.0);
      REQUIRE(*next(it) == 0.1);
      ++it;
      REQUIRE(*it == 0.1);
      REQUIRE(*prev(it) == 0.0);
      advance(it, 2);
      REQUIRE(*it == Approx(0.3));
      --it;
      REQUIRE(*it == Approx(0.2));
    }

    SECTION("distance") {
      auto lin_it           = begin(lin);
      auto other_range      = std::vector<double>{0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                  0.6, 0.7, 0.8, 0.9, 1.0};
      auto other_range_it   = begin(other_range);

      REQUIRE(lin_it - next(lin_it) == other_range_it - next(other_range_it));
      REQUIRE(next(lin_it) - lin_it == next(other_range_it) - other_range_it);
      REQUIRE(distance(end(lin), begin(lin)) ==
              distance(end(other_range), begin(other_range)));
      REQUIRE(end(lin) - begin(lin) == end(other_range) - begin(other_range));
      REQUIRE(distance(begin(lin), end(lin)) ==
              distance(begin(other_range), end(other_range)));
      REQUIRE(begin(lin) - end(lin) == begin(other_range) - end(other_range));
      REQUIRE(distance(lin_it, next(lin_it)) ==
              distance(other_range_it, next(other_range_it)));
    }

    SECTION("for loop") {
      auto as_vector = std::vector<linspace_type::value_type>{};
      for (auto it = begin(lin); it != end(lin); ++it) {
        as_vector.push_back(*it);
      }
      REQUIRE_THAT(as_vector,
                   Catch::Matchers::Approx(std::vector{
                       0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}));
    }
  }
  SECTION("range") {
    SECTION("for loop") {
      auto as_vector = std::vector<linspace_type::value_type>{};
      for (auto t : lin) {
        as_vector.push_back(t);
      }
      REQUIRE_THAT(as_vector,
                   Catch::Matchers::Approx(std::vector{
                       0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}));
    }
    SECTION("C++20 ranges library") {
      auto as_vector = std::vector<linspace_type::value_type>{};
      std::ranges::copy(lin, std::back_inserter(as_vector));
      REQUIRE_THAT(as_vector,
                   Catch::Matchers::Approx(std::vector{
                       0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}));
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
