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
// infer difference type
//==============================================================================
/// Base case
template <typename>
struct infer_difference_type {
  using type = std::ptrdiff_t;
};
//------------------------------------------------------------------------------
template <typename Iter>
concept implements_distance_to = requires(Iter const left, Iter const right) {
  left.distance_to(right);
};
//------------------------------------------------------------------------------
/// Case when `Iter` provides a `distance_to`
template <implements_distance_to Iter>
struct infer_difference_type<Iter> {
  static Iter const &iter;
  using type = decltype(iter.distance_to(iter));
};
//------------------------------------------------------------------------------
template <typename Iter>
using infer_difference_type_t = typename infer_difference_type<Iter>::type;
//==============================================================================
template <typename Arg, typename Iter>
concept difference_type_arg =
    std::convertible_to<Arg, infer_difference_type_t<Iter>>;
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
template <typename T>
concept implements_decrement = requires(T iter) {
  iter.decrement();
};
//==============================================================================
template <typename T>
concept implements_increment = requires(T iter) {
  iter.increment();
};
//==============================================================================
// Check for .advance
template <typename T>
concept implements_advance = requires(T                                it,
                                      infer_difference_type_t<T> const offset) {
  it.advance(offset);
};
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
//==============================================================================
template <typename T, typename Iter>
concept iter_sentinel_arg = std::same_as<std::decay_t<T>, typename Iter::sentinel_type>;
//------------------------------------------------------------------------------
template <typename Iter>
concept implements_distance_to_sentinel =
    requires(Iter const it, typename Iter::sentinel_type const sentinel) {
  it.distance_to(sentinel);
};
//==============================================================================
/// \brief C++20 implementation of an iterator facade.
///
/// Code is taken from here:
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
  //==============================================================================
 private:
  auto as_derived() -> auto & { return static_cast<iterator_type &>(*this); }
  [[nodiscard]] auto as_derived() const -> auto const & {
    return static_cast<iterator_type const &>(*this);
  }
 public:
  //==============================================================================
  decltype(auto) operator*() const { return as_derived().dereference(); }
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
  friend auto operator+=(iterator_type                          &self,
                         difference_type_arg<iterator_type> auto offset)
      -> iterator_type &
  requires implements_advance<iterator_type> {
    self.advance(offset);
    return self;
  }
  //==============================================================================
  auto operator++() -> iterator_type & {
    if constexpr (implements_increment<iterator_type>) {
      // Prefer .increment() if available
      as_derived().increment();
    } else {
      static_assert(implements_advance<iterator_type>,
                    "Iterator subclass must provide either "
                    ".advance() or .increment()");
      as_derived() += 1;
    }
    return as_derived();
  }
  //==============================================================================
  auto operator++(int) {
    auto copy = as_derived();
    ++*this;
    return copy;
  }
  //==============================================================================
  auto operator--() -> iterator_type & {
    if constexpr (implements_decrement<iterator_type>) {
      // Prefer .decrement() if available
      as_derived().decrement();
    } else {
      static_assert(implements_advance<iterator_type>,
                    "Iterator subclass must provide either "
                    ".advance() or .decrement()");
      as_derived() -= 1;
    }
    return as_derived();
  }
  //==============================================================================
  // Postfix:
  auto operator--(int) requires implements_decrement<iterator_type> {
    auto copy = *this;
    --*this;
    return copy;
  }
  //==============================================================================
  friend auto operator+(iterator_type                           left,
                        difference_type_arg<iterator_type> auto off)
      -> iterator_type
  requires implements_advance<iterator_type> {
    return left += off;
  }
  //==============================================================================
  friend auto operator+(difference_type_arg<iterator_type> auto off,
                        iterator_type                           right)
      -> iterator_type
  requires implements_advance<iterator_type> {
    return right += off;
  }
  //==============================================================================
  friend auto operator-(iterator_type                           left,
                        difference_type_arg<iterator_type> auto off)
      -> iterator_type
  requires implements_advance<iterator_type> {
    return left + -off;
  }
  //==============================================================================
  friend auto operator-=(iterator_type                          &left,
                         difference_type_arg<iterator_type> auto off)
      -> iterator_type &
  requires implements_advance<iterator_type> {
    return left = left - off;
  }
  //==============================================================================
  friend auto operator-=(iterator_type                          &left,
                         iter_sentinel_arg<iterator_type> auto sentinel)
      -> iterator_type &
  requires implements_advance<iterator_type> {
    return left = left - sentinel;
  }
  //==============================================================================
  auto operator[](difference_type_arg<iterator_type> auto pos)
      -> decltype(auto)
  requires implements_advance<iterator_type> {
    return *(as_derived() + pos);
  }
  //==============================================================================
  friend auto operator-(iterator_type const &left, iterator_type const &right)
      -> infer_difference_type_t<iterator_type>
  requires implements_distance_to<iterator_type>
  {
    // How many times must we `++right` to reach `left`?
    return right.distance_to(left);
  }
  //==============================================================================
  friend auto operator-(iterator_type const &left,
                        iter_sentinel_arg<iterator_type> auto const sentinel)
      -> infer_difference_type_t<iterator_type>
  requires implements_distance_to_sentinel<iterator_type> {
    // How many times must we `++right` to reach `left`?
    return -left.distance_to(sentinel);
  }
  //==============================================================================
  friend auto operator-(iter_sentinel_arg<iterator_type> auto const sentinel,
                        iterator_type const &left)
      -> infer_difference_type_t<iterator_type>
  requires implements_distance_to_sentinel<iterator_type> {
    // How many times must we `++right` to reach `left`?
    return left.distance_to(sentinel);
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
      -> bool {
    if constexpr (implements_equal<iterator_type>) {
      return left.equal(right);
    } else {
      static_assert(implements_distance_to<iterator_type>,
                    "Iterator must provide `.equal()` "
                    "or `.distance_to()`");
      return left.distance_to(right) == 0;
    }
  }
  //==============================================================================
  friend auto operator!=(iterator_type const &left,
                         iterator_type const &right)
      -> bool {
    return !operator==(left, right);
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
}  // namespace tatooine
//==============================================================================
template <tatooine::derived_from_iterator_facade Iter>
struct std::iterator_traits<Iter> {
  static Iter const &_it;
  using reference         = decltype(*_it);
  using pointer           = decltype(_it.operator->());
  using value_type        = tatooine::infer_value_type_t<Iter>;
  using difference_type   = tatooine::infer_difference_type_t<Iter>;
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
