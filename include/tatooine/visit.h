#ifndef TATOOINE_VISIT_H
#define TATOOINE_VISIT_H
//==============================================================================
#include <variant>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, typename Variant>
constexpr auto visit(F&& f, Variant&& variant) -> decltype(auto) {
  std::visit(std::forward<F>(f), std::forward<Variant>(variant));
}
//==============================================================================
template <typename F, typename Variant0, typename Variant1,
          typename... RestVariants>
constexpr auto visit(F&& f, Variant0&& variant0, Variant1&& variant1,
                     RestVariants&&... rest) -> void {
  visit(
      [&](auto&& visitor0) {
        visit(
            [&](auto&&... visitors) {
              f(std::forward<decltype(visitor0)>(visitor0),
                std::forward<decltype(visitors)>(visitors)...);
            },
            std::forward<Variant1>(variant1),
            std::forward<RestVariants>(rest)...);
      },
      std::forward<Variant0>(variant0));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
