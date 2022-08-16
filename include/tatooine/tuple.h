#ifndef TATOOINE_TUPLE_H
#define TATOOINE_TUPLE_H
//==============================================================================
#include <concepts>
#include <utility>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename... Ts>
struct tuple;
//------------------------------------------------------------------------------
template <typename Head, typename... Tail>
struct tuple<Head, Tail...> {
  Head           head;
  tuple<Tail...> tail;
  static auto constexpr size() { return 1 + sizeof...(Tail); }
  //============================================================================
  template <typename... Tail_>
  tuple(Head&& head_, Tail_&&... tail_)
      : head{std::move(head_)}, tail{std::forward<Tail_>(tail_)...} {}
  //----------------------------------------------------------------------------
  template <std::convertible_to<Head> Head_, typename... Tail_>
  tuple(Head_&& head_, Tail_&&... tail_)
      : head{static_cast<Head>(std::forward<Head_>(head_))},
        tail{std::forward<Tail_>(tail_)...} {}
  //----------------------------------------------------------------------------
  tuple()                 = default;
  tuple(tuple const&)     = default;
  tuple(tuple&&) noexcept = default;
  ~tuple()                = default;
  auto operator=(tuple const&) -> tuple& = default;
  auto operator=(tuple&&) noexcept -> tuple& = default;
  //============================================================================
  template <typename T = Head>
  auto as_pointer() {
    return reinterpret_cast<T*>(this);
  }
  //----------------------------------------------------------------------------
  template <std::size_t I> requires (I < size())
  auto at() const -> auto const& {
    if constexpr (I == 0) {
      return head;
    } else {
      return tail.template at<I - 1>();
    }
  }
  //----------------------------------------------------------------------------
  template <std::size_t I> requires (I < size())
  auto at() -> auto& {
    if constexpr (I == 0) {
      return head;
    } else {
      return tail.template at<I - 1>();
    }
  }
  //----------------------------------------------------------------------------
  template <std::invocable<Head> F>
  requires(std::invocable<F, Tail>&&...)
  auto iterate(F&& f) {
    f(head);
    tail.iterate(std::forward<F>(f));
  }
  //----------------------------------------------------------------------------
  template <std::invocable<Head> F>
  requires(std::invocable<F, Tail>&&...)
  auto iterate(F&& f) const {
    f(static_cast<Head const&>(head));
    tail.iterate(std::forward<F>(f));
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Head, typename... Tail>
tuple(Head&&, Tail&&...) -> tuple<std::decay_t<Head>, std::decay_t<Tail>...>;
//------------------------------------------------------------------------------
template <typename Head>
struct tuple<Head> {
  static auto constexpr size() {return 1;}
  Head head;
  //============================================================================
  template <std::convertible_to<Head> Head_>
  tuple(Head_&& head_) : head{static_cast<Head>(std::forward<Head_>(head_))} {}
  //----------------------------------------------------------------------------
  tuple()                 = default;
  tuple(tuple const&)     = default;
  tuple(tuple&&) noexcept = default;
  ~tuple()                = default;
  auto operator=(tuple const&) -> tuple& = default;
  auto operator=(tuple&&) noexcept -> tuple& = default;
  //============================================================================
  template <typename T = Head>
  auto as_pointer() {
    return reinterpret_cast<T*>(this);
  }
  //----------------------------------------------------------------------------
  template <std::size_t I>
  requires(I == 0)
  auto at() const -> auto const& { return head; }
  //----------------------------------------------------------------------------
  template <std::size_t I>
  requires(I == 0)
  auto at() -> auto& { return head; }
  //----------------------------------------------------------------------------
  template <std::invocable<Head> F>
  auto iterate(F&& f) {
    f(head);
  }
  //----------------------------------------------------------------------------
  template <std::invocable<Head> F>
  auto iterate(F&& f) const {
    f(static_cast<Head const&>(head));
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Head>
tuple(Head&&) -> tuple<std::decay_t<Head>>;
//==============================================================================
template <std::size_t Idx, typename... Ts>
constexpr auto get(tuple<Ts...> const& t) -> auto const& {
  return t.template at<Idx>(t);
}
//------------------------------------------------------------------------------
template <std::size_t Idx, typename... Ts>
constexpr auto get(tuple<Ts...>& t) -> auto& {
  return t.template at<Idx>(t);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
