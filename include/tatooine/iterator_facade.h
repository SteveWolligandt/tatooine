#ifndef TATOOINE_ITERATOR_FACADE_H
#define TATOOINE_ITERATOR_FACADE_H
//==============================================================================
#include <tatooine/arrow_proxy.h>

#include <concepts>
#include <iterator>
#include <memory>
//==============================================================================
namespace tatooine {
//==============================================================================
struct iterator_sentinel{};
//==============================================================================
// infer value type
//==============================================================================
/// Just use the return type of derefence operator.
template <typename Iter>
struct infer_value_type {
  static Iter const &_it;
  using type = std::remove_cvref_t<decltype(*_it)>;
};
//------------------------------------------------------------------------------
/// If `value_type` is explicitly given use this typedef
template <typename T>
requires requires { typename T::value_type; }
struct infer_value_type<T> {
  using type = typename T::value_type;
};
//------------------------------------------------------------------------------
template <typename T>
using infer_value_type_t = typename infer_value_type<T>::type;
//==============================================================================
template <typename Iter>
concept implements_distance_to = requires(Iter const iter) {
  iter.distance_to(iter);
};
//------------------------------------------------------------------------------
template <typename T>
concept implements_decrement = requires(T iter) {
  iter.decrement();
};
//==============================================================================
template <typename T>
concept implements_dereference = requires(T iter) {
  iter.dereference();
};
//==============================================================================
template <typename T>
concept implements_increment = requires(T iter) {
  iter.increment();
};
//==============================================================================
// infer difference type
//==============================================================================
/// Base case
template <typename>
struct infer_difference_type_impl {
  using type = std::ptrdiff_t;
};
//------------------------------------------------------------------------------
/// Case when `Iter` provides a `distance_to`
template <implements_distance_to Iter>
struct infer_difference_type_impl<Iter> {
  using type = decltype(std::declval<Iter>().distance_to(std::declval<Iter>()));
};
//------------------------------------------------------------------------------
template <typename Iter>
using infer_difference_type = typename infer_difference_type_impl<Iter>::type;
//==============================================================================
// Check for .advance
template <typename T>
concept implements_advance = requires(T it) {
  it.advance(infer_difference_type<T>{});
};
//==============================================================================
template <typename Arg, typename Iter>
concept difference_type_arg =
    std::convertible_to<Arg, infer_difference_type<Iter>>;
//==============================================================================
// Check for .equal_to
template <typename T>
concept implements_equal = requires(T const it) {
  { it.equal(it) } -> std::convertible_to<bool>;
};
//==============================================================================
/// We can meet "random access" if it provides
/// both .advance() and .distance_to()
template <typename T>
concept meets_random_access =
    implements_advance<T> && implements_distance_to<T>;
//==============================================================================
/// We meet `bidirectional` if we are random_access, OR we have .decrement()
template <typename T>
concept meets_bidirectional = meets_random_access<T> || implements_decrement<T>;
//==============================================================================
/// Detect if the iterator declares itself to be a single-pass iterator.
template <typename T>
concept declares_single_pass = bool(T::single_pass_iterator);
//------------------------------------------------------------------------------
template <typename Iter>
concept implements_sentinel_type = requires {
  typename Iter::sentinel_type;
};
//------------------------------------------------------------------------------
template <typename T, typename Iter>
concept iter_sentinel_arg = 
  std::same_as<std::decay_t<T>, typename Iter::sentinel_type>;
//------------------------------------------------------------------------------
//template <typename Iter, typename Sentinel>
//concept implements_distance_to_sentinel = 
//    requires(Iter const it, Sentinel const sentinel) {
//  it.distance_to(sentinel);
//};
//==============================================================================
template <typename Iter>
concept implements_basic_iterator_facade =
    implements_increment<Iter> && implements_dereference<Iter>;
//==============================================================================
/// \brief C++20 implementation of an iterator facade.
///
/// Code originally taken from here:
/// https://vector-of-bool.github.io/2020/06/13/cpp20-iter-facade.html
///
/// iterator_facade uses <a
/// href="https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern">CRTP</a>
/// for inheritance.
///
/// It automatically infers `value_type` (see infer_value_type),
/// `difference_type` (see infer_difference_type), `iterator_category`,
/// `reference` and `pointer` in the std::iterator_traits. Alternatively
/// `value_type` can be explicitely specified in the iterator implementation.
/// `difference_type` can be implicitely specified by implementing a
/// `distance_to` method.
///
/// The implementation needs at least the methods `dereference`, `increment`,
/// `equal` and a default constructor.
///
/// One can also use <a
/// href="https://en.wikipedia.org/wiki/Sentinel_value">sentinels</a> by
/// specifying the type `sentinel_type` in the iterator implementation. With
/// this your iterator implementation needs to specify an `equal` method.
///
/// This is an example of how to implement an iterator class with help of
/// iterator_facade:
/// \include iterator_facade.cpp
//==============================================================================
template <typename Iter>
class iterator_facade;
//==============================================================================
template <typename Iter>
concept derived_from_iterator_facade =
    std::is_base_of_v<tatooine::iterator_facade<Iter>, Iter>;
//==============================================================================
template <typename Iter>
class iterator_facade {
 public:
  using iterator_type = Iter;
  using this_type = iterator_facade<iterator_type>;
  //==============================================================================
 private:
  [[nodiscard]] auto as_derived() -> auto & {
    return static_cast<iterator_type &>(*this);
  }
  [[nodiscard]] auto as_derived() const -> auto const & {
    return static_cast<iterator_type const &>(*this);
  }

 public:
  //==============================================================================
  decltype(auto) operator*() const requires implements_dereference<Iter> {
    return as_derived().dereference();
  }
  //==============================================================================
  auto operator->() const {
    decltype(auto) ref = **this;
    if constexpr (std::is_reference_v<decltype(ref)>) {
      // `ref` is a true reference, and we're safe to take its address
      return std::addressof(ref);
    } else {
      // `ref` is *not* a reference. Returning its address would be the
      // address of a local. Return that thing wrapped in an arrow_proxy.
      return arrow_proxy<std::decay_t<decltype(ref)>>{std::move(ref)};
    }
  }
  //==============================================================================
  auto operator++() -> auto &requires implements_increment<Iter> {
    as_derived().increment();
    return as_derived();
  }
  //==============================================================================
  auto operator++(int) requires implements_increment<Iter> {
    auto copy = as_derived();
    as_derived().increment();
    return copy;
  }
  //==============================================================================
  auto operator--() -> auto &
  requires implements_decrement<Iter> {
    as_derived().decrement();
    return as_derived();
  }
  //==============================================================================
  // Postfix:
  auto operator--(int) requires implements_decrement<iterator_type> {
    auto copy = *this;
    as_derived().decrement();
    return copy;
  }
  //==============================================================================
  friend auto operator+(iterator_type                                 left,
                        difference_type_arg<iterator_type> auto const off)
  requires implements_advance<iterator_type> {
    return left += off;
  }
  //==============================================================================
  friend auto operator+=(iterator_type                                &self,
                         difference_type_arg<iterator_type> auto const offset)
      -> auto &
  requires implements_advance<iterator_type> {
    self.advance(offset);
    return self;
  }
  //==============================================================================
  friend auto operator+(difference_type_arg<iterator_type> auto const offset,
                        iterator_type                                 right)
  requires implements_advance<iterator_type> {
    return right += offset;
  }
  //==============================================================================
  friend auto operator-(iterator_type                           left,
                        difference_type_arg<iterator_type> auto off)
  requires implements_advance<iterator_type> {
    return left + -off;
  }
  //==============================================================================
  friend auto operator-=(iterator_type                          &left,
                         difference_type_arg<iterator_type> auto off)
      -> auto&
  requires implements_advance<iterator_type> {
    return left = left - off;
  }
  //==============================================================================
  friend auto operator-=(iterator_type                        &left,
                         iter_sentinel_arg<iterator_type> auto sentinel)
      -> auto& 
  requires implements_advance<iterator_type> {
    left = left - sentinel;
    return left;
  }
  //==============================================================================
  auto operator[](difference_type_arg<iterator_type> auto pos)
      -> decltype(auto) 
  requires implements_advance<iterator_type> {
    return *(as_derived() + pos);
  }
  //==============================================================================
  friend auto operator-(iterator_type const &left, iterator_type const &right)
  requires implements_distance_to<iterator_type> {
    // How many times must we `++right` to reach `left`?
    return left.distance_to(right);
  }
  //==============================================================================
  friend auto operator-(iterator_type const &left,
                        iter_sentinel_arg<iterator_type> auto const sentinel)
  requires implements_distance_to<iterator_type> {
    // How many times must we `++right` to reach `left`?
    return left.distance_to(sentinel);
  }
  //==============================================================================
  friend auto operator-(iter_sentinel_arg<iterator_type> auto const sentinel,
                        iterator_type const &left)
  requires implements_distance_to<iterator_type> {
    // How many times must we `++right` to reach `left`?
    return -left.distance_to(sentinel);
  }
  //==============================================================================
  friend auto operator<=>(iterator_type const &left,
                          iterator_type const &right)
  requires implements_distance_to<iterator_type> {
    return (left - right) <=> 0;
  }
  //==============================================================================
  friend auto operator==(iterator_type const &left,
                         iterator_type const &right)
  requires implements_equal<Iter> {
    return left.equal(right);
  }
  //==============================================================================
  friend auto operator!=(iterator_type const &left,
                         iterator_type const &right) {
    return !left.equal(right);
  }
  //==============================================================================
  friend auto operator==(
      iterator_type const &iter,
      iter_sentinel_arg<iterator_type> auto const /*sentinel*/) -> bool {
    return iter.at_end();
  }
  //==============================================================================
  friend auto operator!=(
      iterator_type const &iter,
      iter_sentinel_arg<iterator_type> auto const /*sentinel*/) -> bool {
    return !iter.at_end();
  }
  //==============================================================================
  friend auto operator==(
      iter_sentinel_arg<iterator_type> auto const /*sentinel*/,
      iterator_type const &iter) -> bool {
    return iter.at_end();
  }
  //==============================================================================
  friend auto operator!=(
      iter_sentinel_arg<iterator_type> auto const /*sentinel*/,
      iterator_type const &iter) -> bool {
    return !iter.at_end();
  }
};
//==============================================================================
template <typename Range>
requires requires(Range &&range) { range.begin(); }
auto begin(Range &&range) { return range.begin(); }
//==============================================================================
template <typename Range>
requires requires(Range &&range) { range.end(); }
auto end(Range &&range) { return range.end(); }
//==============================================================================
template <derived_from_iterator_facade Iter>
auto next(Iter iter) {
  return ++iter;
}
//==============================================================================
template <derived_from_iterator_facade Iter>
auto next(Iter iter, difference_type_arg<Iter> auto off) {
  if constexpr (implements_advance<Iter>) {
    return iter += off;
  } else {
    for (decltype(off) i = 0; i < off; ++i) {
      ++iter;
    }
    return iter;
  }
}
//==============================================================================
template <derived_from_iterator_facade Iter>
requires implements_decrement<Iter>
auto prev(Iter iter) { return --iter; }
//==============================================================================
template <derived_from_iterator_facade Iter>
requires implements_decrement<Iter>
auto prev(Iter iter, difference_type_arg<Iter> auto off) {
  if constexpr (implements_advance<Iter>) {
    return iter -= off;
  } else {
    for (decltype(off) i = 0; i < off; ++i) {
      --iter;
    }
    return iter;
  }
}
//==============================================================================
template <derived_from_iterator_facade Iter>
requires implements_advance<Iter>
auto advance(Iter &iter) {
  return ++iter;
}
//==============================================================================
template <derived_from_iterator_facade Iter>
requires implements_advance<Iter>
auto advance(Iter &iter, difference_type_arg<Iter> auto off) {
  return iter.advance(off);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
template <tatooine::derived_from_iterator_facade Iter>
struct std::iterator_traits<Iter> {
  static Iter const &_it;
  using reference         = decltype(*_it);
  using pointer           = decltype(_it.operator->());
  using value_type        = tatooine::infer_value_type_t<Iter>;
  using difference_type   = tatooine::infer_difference_type<Iter>;
  using iterator_category = conditional_t<
      tatooine::meets_random_access<Iter>,
      // We meet the requirements of random-access:
      random_access_iterator_tag,
      // We don't:
      conditional_t<tatooine::meets_bidirectional<Iter>,
                    // We meet requirements for bidirectional usage:
                    bidirectional_iterator_tag,
                    // We don't:
                    conditional_t<tatooine::declares_single_pass<Iter>,
                                  // A single-pass iterator is an
                                  // input-iterator:
                                  input_iterator_tag,
                                  // Otherwise we are a forward iterator:
                                  forward_iterator_tag>>>;

  // Just set this to the iterator_category, for now
  using iterator_concept = iterator_category;
};

#endif
