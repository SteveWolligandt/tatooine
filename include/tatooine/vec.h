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
  static auto constexpr zeros() { return this_t{tag::fill<T>{0}}; }
  //----------------------------------------------------------------------------
  static auto constexpr ones() { return this_t{tag::fill<T>{1}}; }
  //----------------------------------------------------------------------------
  static auto constexpr fill(T const& t) { return this_t{tag::fill<T>{t}}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto constexpr randu(T min = 0, T max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto constexpr randn(T mean = 0, T stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random::normal<T>{eng, mean, stddev}};
  }
  //----------------------------------------------------------------------------
  constexpr vec(vec const&)           = default;
  constexpr vec(vec&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto constexpr operator=(vec const&) -> vec& = default;
  auto constexpr operator=(vec&& other) noexcept -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  explicit constexpr vec(base_tensor<OtherTensor, OtherReal, N> const& other)
      : parent_t{other} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  auto constexpr operator=(base_tensor<OtherTensor, OtherReal, N> const& other)
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
  auto constexpr x() const -> auto const& requires(N >= 1) { return at(0); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr x() -> auto& requires(N >= 1) { return at(0); }
  //----------------------------------------------------------------------------
  auto constexpr y() const -> auto const& requires(N >= 2) { return at(1); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr xy() const requires(N >= 2) { return vec<T, 2>{at(0), at(1)}; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr yx() const requires(N >= 2) { return vec<T, 2>{at(1), at(0)}; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr y() -> auto& requires(N >= 2) { return at(1); }
  //----------------------------------------------------------------------------
  auto constexpr xyz() const requires(N >= 3) {
    return vec<T, 3>{at(0), at(1), at(2)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr z() const -> auto const& requires(N >= 3) { return at(2); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr z() -> auto& requires(N >= 3) { return at(2); }
  //----------------------------------------------------------------------------
  auto constexpr xyzw() const requires(N >= 3) {
    return vec<T, 4>{at(0), at(1), at(2), at(3)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr w() const -> auto const& requires(N >= 4) { return at(3); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr w() -> auto& requires(N >= 4) { return at(3); }

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
