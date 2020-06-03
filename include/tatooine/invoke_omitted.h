#ifndef TATOOINE_INVOKE_OMITTED_H
#define TATOOINE_INVOKE_OMITTED_H

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, typename... Params>
constexpr decltype(auto) invoke_omitted(F&& f, Params&&... params) {
  return std::invoke(f, std::forward<Params>(params)...);
}
//------------------------------------------------------------------------------
template <size_t i, size_t... is, typename F, typename Param,
          typename... Params>
constexpr decltype(auto) invoke_omitted(F&& f, Param&& param,
                                        Params&&... params) {
  if constexpr (i == 0) {
    return invoke_omitted<(is - 1)...>(
        [&](auto&&... lparams) -> decltype(auto) {
          return std::invoke(f, std::forward<decltype(lparams)>(lparams)...);
        },
        std::forward<Params>(params)...);
  } else {
    return invoke_omitted<i - 1, (is - 1)...>(
        [&](auto&&... lparams) -> decltype(auto) {
          return std::invoke(f, std::forward<Param>(param),
                             std::forward<decltype(lparams)>(lparams)...);
        },
        std::forward<Params>(params)...);
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
