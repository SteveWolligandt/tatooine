#ifndef TATOOINE_VEC_H
#define TATOOINE_VEC_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
#include <tatooine/random.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex T, size_t N>
struct vec : tensor<T, N> {
  using this_t   = vec<T, N>;
  using parent_t = tensor<T, N>;
  using parent_t::at;
  using parent_t::dimension;
  using parent_t::parent_t;
  using parent_t::operator();

  //----------------------------------------------------------------------------
  template <typename... Ts>
      requires((is_convertible<Ts, T> && ...)) &&
      (parent_t::dimension(0) == sizeof...(Ts)) constexpr vec(Ts const&... ts)
      : parent_t{ts...} {}

  using iterator = typename parent_t::array_parent_t::container_t::iterator;
  using const_iterator =
      typename parent_t::array_parent_t::container_t::const_iterator;
  //============================================================================
  static constexpr auto zeros() { return this_t{tag::fill<T>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::fill<T>{1}}; }
  //----------------------------------------------------------------------------
  static constexpr auto fill(T const& t) { return this_t{tag::fill<T>{t}}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randu(T min = 0, T max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(T mean = 0, T stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::normal<T>{eng, mean, stddev}};
  }
  //----------------------------------------------------------------------------
  constexpr vec(vec const&)           = default;
  constexpr vec(vec&& other) noexcept = default;
  //----------------------------------------------------------------------------
  constexpr auto operator=(vec const&) -> vec& = default;
  constexpr auto operator=(vec&& other) noexcept -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  explicit constexpr vec(base_tensor<OtherTensor, OtherReal, N> const& other)
      : parent_t{other} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  constexpr auto operator=(base_tensor<OtherTensor, OtherReal, N> const& other)
      -> vec& {
    this->assign_other_tensor(other);
    return *this;
  }
  //----------------------------------------------------------------------------
  ~vec() = default;

  auto begin() const { return this->data().begin(); }
  auto begin() { return this->data().begin(); }
  auto end() const { return this->data().end(); }
  auto end() { return this->data().end(); }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N >= 1) constexpr auto x() const -> auto const& {
    return this->at(0);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires(N >= 1) constexpr auto x() -> auto& {
    return this->at(0);
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N >= 2) constexpr auto y() const -> auto const& {
    return this->at(1);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires(N >= 2) constexpr auto y() -> auto& {
    return this->at(1);
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N >= 3) constexpr auto z() const -> auto const& {
    return this->at(2);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename = void>
  requires(N >= 3) constexpr auto z() -> auto& {
    return this->at(2);
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires(N >= 4)
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // - -
      constexpr auto w() const -> auto const& {
    return this->at(3);
  }
  template <typename = void>
  requires(N >= 4) constexpr auto w() -> auto& {
    return this->at(3);
  }

  template <typename Archive>
  auto serialize(Archive& ar, unsigned int const /*version*/) -> void {
    for (size_t i = 0; i < N; ++i) {
      ar& at(i);
    }
  }
};
//==============================================================================
// type traits
//==============================================================================
template <typename T, size_t N>
auto begin(vec<T, N> const& v) {
  return v.begin();
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto end(vec<T, N> const& v) {
  return v.ned();
}
//==============================================================================
template <typename... Ts>
vec(const Ts&...) -> vec<common_type<Ts...>, sizeof...(Ts)>;
//------------------------------------------------------------------------------
template <typename V, typename T, std::size_t N>
vec(base_tensor<V, T, N> const&) -> vec<T, N>;
//==============================================================================
namespace reflection {
template <typename T, size_t N>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (vec<T, N>), TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/vec_typedefs.h>
//==============================================================================
#endif
