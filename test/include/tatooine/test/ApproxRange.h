#ifndef TATOOINE_TEST_APPROX_RANGE_H
#define TATOOINE_TEST_APPROX_RANGE_H
//==============================================================================
#include <catch2/catch_approx.hpp>
#include <optional>
#include <catch2/matchers/catch_matchers_templated.hpp>
//==============================================================================
template <std::ranges::range Range>
struct ApproxRangeMatcher : Catch::Matchers::MatcherGenericBase {
  using range_value_type = std::ranges::range_value_t<Range>;

  //----------------------------------------------------------------------------
 private:
  Range                           m_range;
  std::optional<range_value_type> m_margin;

  //----------------------------------------------------------------------------
 public:
  template <std::convertible_to<Range> OtherRange>
  ApproxRangeMatcher(OtherRange&& range)
      : m_range{std::forward<Range>(range)} {}
  //----------------------------------------------------------------------------
  auto match(std::ranges::range auto const& other) const -> bool {
    return std::ranges::all_of(m_range, [this, it = std::ranges::begin(other)](
                                            auto const& elem) mutable {
      if (m_margin.has_value()) {
        return elem == Catch::Approx(*(it++)).margin(*m_margin);
      } else {
        return elem == Catch::Approx(*(it++));
      }
    });
  }
  //----------------------------------------------------------------------------
  auto describe() const -> std::string override {
    return "Approx: " + Catch::rangeToString(m_range);
  }
  //----------------------------------------------------------------------------
  auto margin(range_value_type const m) -> auto& {
    m_margin = m;
    return *this;
  }
};
//==============================================================================
template <typename T>
ApproxRangeMatcher(T &&) -> ApproxRangeMatcher<T>;
template <typename T>
ApproxRangeMatcher(T const&) -> ApproxRangeMatcher<T const&>;
template <typename T>
ApproxRangeMatcher(T&) -> ApproxRangeMatcher<T const&>;
//==============================================================================
auto ApproxRange(std::ranges::range auto && range) {
  return ApproxRangeMatcher{range};
}
//==============================================================================
#endif
