#ifndef TATOOINE_VISIT_H
#define TATOOINE_VISIT_H
//==============================================================================
#include <variant>
//==============================================================================
namespace tatooine {
//==============================================================================
using std::visit;
//==============================================================================
/// Visit implementation that wraps stored value of std::variants to function
/// parameters.
template <typename Visitor, typename Variant0, typename Variant1,
          typename... Variants>
constexpr auto visit(Visitor&& visitor, Variant0&& variant0,
                     Variant1&& variant1, Variants&&... variants) -> void {
  auto nested_visitor = [&](auto&& value0) {
    visit(
        [&](auto&&... rest_of_values) {
          visitor(std::forward<decltype(value0)>(value0),
                  std::forward<decltype(rest_of_values)>(rest_of_values)...);
        },
        std::forward<Variant1>(variant1), std::forward<Variants>(variants)...);
  };
  visit(nested_visitor, std::forward<Variant0>(variant0));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
