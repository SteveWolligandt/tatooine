#ifndef YAVIN_TUPLE_H
#define YAVIN_TUPLE_H
//==============================================================================
#include <utility>
#include <concepts>
//==============================================================================
namespace yavin {
//==============================================================================
template <typename... Ts>
struct tuple;
//------------------------------------------------------------------------------
template <typename Head, typename... Tail>
struct tuple<Head, Tail...> {
  Head           head;
  tuple<Tail...> tail;
  //============================================================================
  template <std::convertible_to<Head> Head_, typename... Tail_>
  tuple(Head_&& head_, Tail_&&... tail_)
      : head{std::forward<Head_>(head_)}, tail{std::forward<Tail_>(tail_)...} {}
  //----------------------------------------------------------------------------
  tuple()                                    = default;
  tuple(tuple const&)                        = default;
  tuple(tuple&&) noexcept                    = default;
  ~tuple()                                   = default;
  auto operator=(tuple const&) -> tuple&     = default;
  auto operator=(tuple&&) noexcept -> tuple& = default;
  //============================================================================
  template <typename T>
  auto as_pointer() {
    return reinterpret_cast<T*>(this);
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Head, typename... Tail>
tuple(Head&&, Tail&&...) -> tuple<std::decay_t<Head>, std::decay_t<Tail>...>;
//------------------------------------------------------------------------------
template <typename Head>
struct tuple<Head> {
  Head head;
  //============================================================================
  template <std::convertible_to<Head> Head_>
  tuple(Head_&& head_) : head{std::forward<Head_>(head_)} {}
  //----------------------------------------------------------------------------
  tuple()                                    = default;
  tuple(tuple const&)                        = default;
  tuple(tuple&&) noexcept                    = default;
  ~tuple()                                   = default;
  auto operator=(tuple const&) -> tuple&     = default;
  auto operator=(tuple&&) noexcept -> tuple& = default;
  //============================================================================
  template <typename T = void>
  auto as_pointer() {
    return reinterpret_cast<T*>(this);
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Head>
tuple(Head&&) -> tuple<std::decay_t<Head>>;
//==============================================================================
template <std::size_t Idx, typename Head, typename... Tail>
struct _tuple_get_t {
  static constexpr auto get(tuple<Head, Tail...> const& t) -> auto const& {
    return _tuple_get_t<Idx - 1, Tail...>::get(t.tail);
  }
  static constexpr auto get(tuple<Head, Tail...>& t) -> auto& {
    return _tuple_get_t<Idx - 1, Tail...>::get(t.tail);
  }
};
//------------------------------------------------------------------------------
template <typename Head, typename... Tail>
struct _tuple_get_t<0, Head, Tail...> {
  static constexpr auto get(tuple<Head, Tail...> const& t) -> auto const& {
    return t.head;
  }
  static constexpr auto get(tuple<Head, Tail...>& t) -> auto& {
    return t.head;
  }
};
//------------------------------------------------------------------------------
template <std::size_t Idx, typename... Ts>
constexpr auto get(tuple<Ts...> const& t) -> auto const& {
  return _tuple_get_t<Idx, Ts...>::get(t);
}
//------------------------------------------------------------------------------
template <std::size_t Idx, typename... Ts>
constexpr auto get(tuple<Ts...>& t) -> auto& {
  return _tuple_get_t<Idx, Ts...>::get(t);
}
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
