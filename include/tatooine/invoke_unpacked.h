#ifndef TATOOINE_INVOKE_UNPACKED_H
#define TATOOINE_INVOKE_UNPACKED_H
//==============================================================================
#include <functional>
#include <tatooine/bind.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Container>
struct unpack;
//==============================================================================
/// All arguments are bound -> just call f
template <typename F>
constexpr decltype(auto) invoke_unpacked(F&& f) {
  return f();
}

//------------------------------------------------------------------------------
/// \brief Recursive currying.
/// Curry first non-unpacked type to f.
/// \return Returns curried function.
template <typename F, typename T, typename... Ts>
constexpr decltype(auto) invoke_unpacked(F&& f, T&& t, Ts&&... ts) {
  return invoke_unpacked(bind(std::forward<F>(f), std::forward<T>(t)),
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
  explicit constexpr unpack(std::array<T, N>& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& { return container[I]; }
  template <size_t I>
  constexpr auto get() const -> const auto& { return container[I]; }
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
  explicit constexpr unpack(const std::array<T, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
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
  explicit constexpr unpack(std::tuple<Ts...>&& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& {
    return std::get<I>(container);
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
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
  explicit constexpr unpack(std::tuple<Ts...>&& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
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
  explicit constexpr unpack(std::pair<A, B>& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& {
    return std::get<I>(container);
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
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
  explicit constexpr unpack(std::pair<A, B>&& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return std::get<I>(container);
  }
};
//==============================================================================
template <typename A, typename B>
unpack(const std::pair<A, B>& c)->unpack<const std::pair<A, B>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
