#ifndef TATOOINE_TEST_APPROX_RANGE_H
#define TATOOINE_TEST_APPROX_RANGE_H
//==============================================================================
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
//==============================================================================
template <std::ranges::range Range>
struct ApproxRangeMatcher : Catch::Matchers::MatcherGenericBase {
 private:
  Range range;

 public:
  template <std::convertible_to<Range> OtherRange>
  ApproxRangeMatcher(OtherRange&& range) : range{std::forward<Range>(range)} {}

  auto match(std::ranges::range auto const& other) const -> bool {
    return std::ranges::all_of(
        range, [it = std::ranges::begin(other)](auto const& elem) mutable {
          return elem == Catch::Approx(*(it++));
        });
  }

  auto describe() const -> std::string override {
    return "Approx: " + Catch::rangeToString(range);
  }
};
template <typename T>
ApproxRangeMatcher(T&&) -> ApproxRangeMatcher<T>;
template <typename T>
ApproxRangeMatcher(T const&) -> ApproxRangeMatcher<T const&>;
template <typename T>
ApproxRangeMatcher(T&) -> ApproxRangeMatcher<T&>;

auto ApproxRange(std::ranges::range auto && range) {
  return ApproxRangeMatcher{range};
}
//==============================================================================
#endif
