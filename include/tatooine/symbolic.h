#ifndef TATOOINE_SYMBOLIC_H
#define TATOOINE_SYMBOLIC_H

#include <ginac/ginac.h>

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================


#define sym(sym)                                                        \
  template <size_t... Is>                                                  \
  static auto& sym(size_t idx, std::index_sequence<Is...> /*is*/) {        \
    static std::array sym_arr{                                             \
        ((void)Is,                                                         \
         GiNaC::symbol{#sym + std::string{"_"} + std::to_string(Is)})...}; \
    return sym_arr[idx];                                                   \
  }                                                                        \
  static auto& sym(size_t idx) {                                           \
    return sym(idx, std::make_index_sequence<num_pos_symbols>{});          \
  }                                                                        \
  static auto& sym() {                                                     \
    static GiNaC::symbol sym{#sym};                                        \
    return sym;                                                            \
  }

struct symbol {
  static constexpr size_t num_pos_symbols = 100;
  sym(i) sym(j) sym(k) sym(x) sym(y) sym(z) sym(t)
};
#undef sym
#undef symarr

//----------------------------------------------------------------------------
/// substitudes expression with relations
template <typename... Relations>
auto ev(const GiNaC::ex& expr, Relations&&... relations) {
  GiNaC::lst substitutions{std::forward<Relations>(relations)...};
  return expr.subs(substitutions);
}

//----------------------------------------------------------------------------
/// substitudes expression with relations and casts substituted expression to
/// double
template <typename out_real_t = double, typename... Relations>
auto evtod(const GiNaC::ex& expr, Relations&&... relations) {
  return static_cast<out_real_t>(
      GiNaC::ex_to<GiNaC::numeric>(
          ev(expr, std::forward<Relations>(relations)...).evalf())
          .to_double());
}


//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

#endif
