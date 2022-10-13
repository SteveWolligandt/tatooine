#ifndef TATOOINE_TEST_APPROX_RANGE_H
#define TATOOINE_TEST_APPROX_RANGE_H
//==============================================================================
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
//==============================================================================
template <std::ranges::range Range>
struct EqualRangeMatcher : Catch::Matchers::MatcherGenericBase {
 private:
  Range range;

 public:
  template <std::convertible_to<Range> OtherRange>
  EqualRangeMatcher(OtherRange&& range) : range{std::forward<Range>(range)} {}

  auto match(std::ranges::range auto const& other) const -> bool {
    return std::ranges::all_of(
        range, [it = std::ranges::begin(other)](auto const& elem) mutable {
          return elem == *(it++);
        });
  }

  auto describe() const -> std::string override {
    return "Equal: " + Catch::rangeToString(range);
  }
};
template <typename T>
EqualRangeMatcher(T&&) -> EqualRangeMatcher<T>;
template <typename T>
EqualRangeMatcher(T const&) -> EqualRangeMatcher<T const&>;
template <typename T>
EqualRangeMatcher(T&) -> EqualRangeMatcher<T const&>;

auto EqualRange(std::ranges::range auto && range) {
  return EqualRangeMatcher{range};
}
//==============================================================================
#endif
