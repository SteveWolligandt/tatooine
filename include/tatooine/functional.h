#ifndef TATOOINE_FUNCTIONAL_H
#define TATOOINE_FUNCTIONAL_H

#include <functional>
#include "cxxstd.h"

//==============================================================================
namespace tatooine {
//==============================================================================

#if has_cxx17_support()
/// maps unary function f to all single parameters of parameter pack ts
template <typename... Ts, typename F>
constexpr void map(F&& f, Ts&&... ts) {
  (f(std::forward<Ts>(ts)), ...);
}
#endif

//==============================================================================
#if has_cxx17_support()
/// binds first arguments of f (either all or only partially)
template <typename F, typename... Args>
constexpr auto bind(F&& f, Args&&... args) {
  return [&](auto&&... rest) -> decltype(auto) {
    return std::invoke(f, std::forward<Args>(args)...,
                       std::forward<decltype(rest)>(rest)...);
  };
}
#endif

//==============================================================================
#if has_cxx17_support()
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
#endif

//==============================================================================
#if has_cxx17_support()
template <typename Container>
struct unpack;

//==============================================================================
/// All arguments are bound -> just call f
template <typename F>
constexpr decltype(auto) invoke_unpacked(F&& f) {
  return std::invoke(f);
}

//------------------------------------------------------------------------------
/// \brief Recursive currying.
/// Curry first non-unpacked type to f.
/// \return Returns curried function.
template <typename F, typename T, typename... Ts>
constexpr decltype(auto) invoke_unpacked(F&& f, T&& t, Ts&&... ts) {
  return invoke_unpacked(tatooine::bind(std::forward<F>(f), std::forward<T>(t)),
                         std::forward<Ts>(ts)...);
}

//------------------------------------------------------------------------------
/// Curries unpacked parameters by calling invoke_unpacked(F&&, T&&, Ts&&...).
template <std::size_t... Is, typename F, typename T, typename... Ts>
constexpr decltype(auto) invoke_unpacked(std::index_sequence<Is...> /*is*/,
                                         F&& f, unpack<T> t, Ts&&... ts) {
  return invoke_unpacked(std::forward<F>(f), t.template get<Is>()...,
                         std::forward<Ts>(ts)...);
}

//------------------------------------------------------------------------------
template <typename F, typename T, typename... Ts>
constexpr decltype(auto) invoke_unpacked(F&& f, unpack<T> t, Ts&&... ts) {
  return invoke_unpacked(std::make_index_sequence<unpack<T>::n>{},
                         std::forward<F>(f), std::move(t),
                         std::forward<Ts>(ts)...);
}

//==============================================================================
// some unpack implementations

template <typename T, size_t N>
struct unpack<std::array<T, N>> {
  static constexpr size_t n = N;
  std::array<T, N>&       container;

  //----------------------------------------------------------------------------
  constexpr unpack(std::array<T, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto& get() {
    return container[I];
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return container[I];
  }
};
//==============================================================================
template <typename T, size_t N>
unpack(std::array<T, N>& c)->unpack<std::array<T, N>>;

//==============================================================================
template <typename T, size_t N>
struct unpack<const std::array<T, N>> {
  static constexpr size_t n = N;
  const std::array<T, N>& container;

  //----------------------------------------------------------------------------
  constexpr unpack(const std::array<T, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return container[I];
  }
};
//==============================================================================
template <typename T, size_t N>
unpack(const std::array<T, N>& c)->unpack<const std::array<T, N>>;

//==============================================================================
template <typename... Ts>
struct unpack<std::tuple<Ts...>> {
  static constexpr size_t n = sizeof...(Ts);
  std::tuple<Ts...>&      container;

  //----------------------------------------------------------------------------
  constexpr unpack(std::tuple<Ts...>&& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto& get() {
    return std::get<I>(container);
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return std::get<I>(container);
  }
};
//==============================================================================
template <typename... Ts>
unpack(std::tuple<Ts...>&& c)->unpack<std::tuple<Ts...>>;

//==============================================================================
template <typename... Ts>
struct unpack<const std::tuple<Ts...>> {
  static constexpr size_t  n = sizeof...(Ts);
  const std::tuple<Ts...>& container;

  //----------------------------------------------------------------------------
  constexpr unpack(std::tuple<Ts...>&& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return std::get<I>(container);
  }
};
//==============================================================================
template <typename... Ts>
unpack(const std::tuple<Ts...>& c)->unpack<const std::tuple<Ts...>>;

//==============================================================================
template <typename A, typename B>
struct unpack<std::pair<A, B>> {
  static constexpr size_t n = 2;
  std::pair<A, B>&        container;

  //----------------------------------------------------------------------------
  constexpr unpack(std::pair<A, B>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto& get() {
    return std::get<I>(container);
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return std::get<I>(container);
  }
};

//==============================================================================
template <typename A, typename B>
unpack(std::pair<A, B>& c)->unpack<std::pair<A, B>>;

//==============================================================================
template <typename A, typename B>
struct unpack<const std::pair<A, B>> {
  static constexpr size_t n = 2;
  const std::pair<A, B>&  container;

  //----------------------------------------------------------------------------
  constexpr unpack(std::pair<A, B>&& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return std::get<I>(container);
  }
};
//==============================================================================
template <typename A, typename B>
unpack(const std::pair<A, B>& c)->unpack<const std::pair<A, B>>;
#endif

//==============================================================================
template <typename T, typename... Ts>
decltype(auto) front_param(T&& head, Ts&&... /*tail*/) {
  return std::forward<T>(head);
}
//==============================================================================
template <typename T>
decltype(auto) back_param(T&& t) {
  return std::forward<T>(t);
}
//==============================================================================
template <typename T0, typename T1, typename... Ts>
decltype(auto) back_param(T0&& t0, T1&& t1, Ts&&... ts) {
  return back_param(std::forward<T1>(t1), std::forward<Ts>(ts)...);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
