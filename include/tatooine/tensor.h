#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H

#include <lapacke.h>

#include <array>
#include <cassert>
#include <iostream>
#include <ostream>

#include "crtp.h"
#include "invoke_unpacked.h"
#include "functional.h"
#include "multidim_array.h"
#include "random.h"
#include "type_traits.h"
#include "utility.h"
#include "math.h"

#if TATOOINE_GINAC_AVAILABLE
#include "symbolic.h"
#endif

#ifdef I
#undef I
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
struct frobenius_t {};
static constexpr frobenius_t frobenius;
struct full_t {};
static constexpr full_t full;
struct economy_t {};
static constexpr economy_t economy;
//------------------------------------------------------------------------------
struct eye_t {};
static constexpr eye_t eye;
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t FixedDim, size_t... Dims>
struct tensor_slice;

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
struct base_tensor : crtp<Tensor> {
  using value_type = Real;
  using real_t     = Real;
  using tensor_t   = Tensor;
  using this_t     = base_tensor<Tensor, Real, Dims...>;
  using parent_t   = crtp<Tensor>;
  using parent_t::as_derived;
  using resolution_t = static_multidim_resolution<x_fastest, Dims...>;

  //============================================================================
  static constexpr auto num_dimensions() { return sizeof...(Dims); }
  //------------------------------------------------------------------------------
  static constexpr auto num_components() {
    return resolution_t::num_elements();
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimensions() {
    return std::array<size_t, num_dimensions()>{Dims...};
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimension(const size_t i) {
    return template_helper::getval<size_t>(i, Dims...);
  }
  //------------------------------------------------------------------------------
  static constexpr auto indices() { return resolution_t::indices(); }
  //------------------------------------------------------------------------------
  template <typename F>
  static constexpr auto for_indices(F&& f) {
    for (auto is : indices()) {
      invoke_unpacked(std::forward<F>(f), unpack(is));
    }
  }
  //============================================================================
  constexpr base_tensor() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  explicit constexpr base_tensor(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    assign_other_tensor(other);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  constexpr auto operator=(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> base_tensor& {
    assign_other_tensor(other);
    return *this;
  }
  //============================================================================
  template <typename F>
  auto unary_operation(F&& f) -> auto& {
    for_indices([this, &f](const auto... is) { at(is...) = f(at(is...)); });
    return as_derived();
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherTensor, typename OtherReal>
  auto binary_operation(
      F&& f, const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> decltype(auto) {
    for_indices([this, &f, &other](const auto... is) {
      at(is...) = f(at(is...), other(is...));
    });
    return as_derived();
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  constexpr void assign_other_tensor(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    for_indices([this, &other](const auto... is) { at(is...) = other(is...); });
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) at(const Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return as_derived().at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) at(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return as_derived().at(is...);
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) operator()(const Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) operator()(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return at(is...);
  }

  //----------------------------------------------------------------------------
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) {
    static_assert(FixedDim < num_dimensions(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<
        Tensor, Real, FixedDim,
        dimension(sliced_indices<num_dimensions(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim>
  constexpr auto slice(size_t fixed_index) {
    static_assert(FixedDim < num_dimensions(),
                  "fixed dimensions must be in range of number of dimensions");
    return slice<FixedDim>(fixed_index,
                           std::make_index_sequence<num_dimensions() - 1>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) const {
    static_assert(FixedDim < num_dimensions(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<
        const Tensor, Real, FixedDim,
        dimension(sliced_indices<num_dimensions(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim>
  [[nodiscard]] constexpr auto slice(size_t fixed_index) const {
    static_assert(FixedDim < num_dimensions(),
                  "fixed dimensions must be in range of number of dimensions");
    return slice<FixedDim>(fixed_index,
                           std::make_index_sequence<num_dimensions() - 1>{});
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  static constexpr auto array_index(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions(),
                  "number of indices does not match number of dimensions");
    return static_multidim_resolution<x_fastest, Dims...>::plain_idx(is...);
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  auto operator+=(const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> auto& {
    for_indices([&](const auto... is) { at(is...) += other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename OtherReal,
            enable_if_arithmetic_or_symbolic<OtherReal> = true>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  auto operator+=(const OtherReal& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) += other; });
    return *this;
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  auto operator-=(const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> auto& {
    for_indices([&](const auto... is) { at(is...) -= other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename OtherReal,
            enable_if_arithmetic_or_symbolic<OtherReal> = true>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  auto operator-=(const OtherReal& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) -= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename OtherReal,
            enable_if_arithmetic_or_symbolic<OtherReal> = true>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  auto operator*=(const OtherReal& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) *= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename OtherReal,
            enable_if_arithmetic_or_symbolic<OtherReal> = true>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  auto operator/=(const OtherReal& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) /= other; });
    return *this;
  }
};
//==============================================================================
template <typename Tensor, size_t... Dims>
struct abs_tensor : base_tensor<abs_tensor<Tensor, Dims...>,
                                typename Tensor::real_t, Dims...> {
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit abs_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, Dims...>&
          internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto operator()(Is... is) const {
    static_assert(sizeof...(Is) == sizeof...(Dims));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto at(Is... is) const {
    static_assert(sizeof...(Is) == sizeof...(Dims));
    return std::abs(m_internal_tensor(is...));
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto abs(const base_tensor<Tensor, Real, Dims...>& t) {
  return abs_tensor<Tensor, Dims...>{t.as_derived()};
}
//==============================================================================
template <typename Tensor, size_t VecN, size_t M, size_t N>
struct const_diag_tensor : base_tensor<const_diag_tensor<Tensor, VecN, M, N>,
                                       typename Tensor::real_t, M, N> {
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit const_diag_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, VecN>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(size_t i, size_t j) const { return at(i, j); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(size_t i, size_t j) const -> typename Tensor::real_t {
    assert(i < M);
    assert(j < N);
    if (i == j) {
      return m_internal_tensor(i);
    } else {
      return 0;
    }
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//==============================================================================
template <typename Tensor, size_t VecN, size_t M, size_t N>
struct diag_tensor
    : base_tensor<diag_tensor<Tensor, VecN, M, N>, typename Tensor::real_t, M, N> {
  //============================================================================
 private:
  Tensor& m_internal_tensor;
  typename Tensor::real_t zero=0;

  //============================================================================
 public:
  constexpr explicit diag_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, VecN>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr auto operator()(size_t i, size_t j) const { return at(i, j); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(size_t i, size_t j) -> auto& {
    assert(i < M);
    assert(j < N);
    zero = 0;
    if (i == j) {
      return m_internal_tensor(i);
    } else {
      return zero;
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto at(size_t i, size_t j) const -> typename Tensor::real_t {
    assert(i < M);
    assert(j < N);
    if (i == j) {
      return m_internal_tensor(i);
    } else {
      return 0;
    }
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t VecN>
constexpr auto diag(const base_tensor<Tensor, Real, VecN>& t) {
  return const_diag_tensor<Tensor, VecN, VecN, VecN>{t};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t VecN>
constexpr auto diag(base_tensor<Tensor, Real, VecN>& t) {
  return diag_tensor<Tensor, VecN, VecN, VecN>{t};
}
//------------------------------------------------------------------------------
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(const base_tensor<Tensor, Real, VecN>& t) {
  return const_diag_tensor<Tensor, VecN, M, N>{t};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t M, size_t N, typename Tensor, typename Real, size_t VecN>
constexpr auto diag_rect(base_tensor<Tensor, Real, VecN>& t) {
  return diag_tensor<Tensor, VecN, M, N>{t};
}
//==============================================================================
template <typename Tensor, size_t M, size_t N>
struct const_transposed_tensor
    : base_tensor<const_transposed_tensor<Tensor, M, N>,
                  typename Tensor::real_t, M, N> {
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit const_transposed_tensor(
      const base_tensor<Tensor, typename Tensor::real_t, N, M>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(const size_t r, const size_t c) const -> const
      auto& {
    return m_internal_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(const size_t r, const size_t c) const -> const auto& {
    return m_internal_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, size_t M, size_t N>
struct transposed_tensor : base_tensor<transposed_tensor<Tensor, M, N>,
                                       typename Tensor::real_t, M, N> {
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit transposed_tensor(
      base_tensor<Tensor, typename Tensor::real_t, N, M>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(const size_t r, const size_t c) const -> const
      auto& {
    return m_internal_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(const size_t r, const size_t c) -> auto& {
    return m_internal_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  constexpr auto at(const size_t r, const size_t c) const -> const auto& {
    return m_internal_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(const size_t r, const size_t c) -> auto& {
    return m_internal_tensor(c, r);
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t M, size_t N>
auto transpose(const base_tensor<Tensor, Real, M, N>& t) {
  return const_transposed_tensor{t};
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transpose(base_tensor<Tensor, Real, M, N>& t) {
  return transposed_tensor<Tensor, N, M>{t};
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transpose(
    base_tensor<transposed_tensor<Tensor, M, N>, Real, M, N>& transposed_tensor)
    -> auto& {
  return transposed_tensor.as_derived().internal_tensor();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transpose(const base_tensor<transposed_tensor<Tensor, M, N>,
                                           Real, M, N>& transposed_tensor)
    -> const auto& {
  return transposed_tensor.as_derived().internal_tensor();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transpose(
    const base_tensor<const_transposed_tensor<Tensor, M, N>, Real, M, N>&
        transposed_tensor) -> const auto& {
  return transposed_tensor.as_derived().internal_tensor();
}
//==============================================================================
template <typename Real, size_t... Dims>
struct tensor : base_tensor<tensor<Real, Dims...>, Real, Dims...>,  // NOLINT
                static_multidim_array<Real, x_fastest, stack, Dims...> {
  //============================================================================
  using this_t          = tensor<Real, Dims...>;
  using tensor_parent_t = base_tensor<this_t, Real, Dims...>;
  using array_parent_t = static_multidim_array<Real, x_fastest, stack, Dims...>;
  using tensor_parent_t::tensor_parent_t;
  using tensor_parent_t::operator=;
  using array_parent_t::at;
  using tensor_parent_t::dimension;
  using tensor_parent_t::num_components;
  using tensor_parent_t::num_dimensions;
  using array_parent_t::operator();

  //============================================================================
 public:
  constexpr tensor()              = default;
  constexpr tensor(const tensor&) = default;
  constexpr auto operator=(const tensor&) -> tensor& = default;

  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  explicit constexpr tensor(tensor&& other) noexcept
      : array_parent_t{std::move(other)} {}

  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  constexpr auto operator=(tensor&& other) noexcept -> tensor& {
    array_parent_t::operator=(std::move(other));
    return *this;
  }
  ~tensor() = default;

  //============================================================================
 public:
  template <typename... Ts, size_t _N = tensor_parent_t::num_dimensions(),
            size_t _Dim0                    = tensor_parent_t::dimension(0),
            std::enable_if_t<_N == 1, bool> = true,
            std::enable_if_t<_Dim0 == sizeof...(Ts), bool> = true>
  explicit constexpr tensor(const Ts&... ts) : array_parent_t{ts...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _Real = Real, enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(zeros_t zeros) : array_parent_t{zeros} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _Real = Real, enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(ones_t ones) : array_parent_t{ones} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename FillReal, typename _Real = Real,
            enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(fill<FillReal> f) : array_parent_t{f} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine, typename _Real = Real,
            enable_if_arithmetic<RandomReal> = true>
  explicit constexpr tensor(random_uniform<RandomReal, Engine>&& rand)
      : array_parent_t{std::move(rand)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine, typename _Real = Real,
            enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(random_normal<RandomReal, Engine>&& rand)
      : array_parent_t{std::move(rand)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  explicit constexpr tensor(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    this->assign_other_tensor(other);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
  constexpr auto operator=(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other) -> tensor& {
    this->assign_other_tensor(other);
    return *this;
  }
  //----------------------------------------------------------------------------
  static constexpr auto zeros() { return this_t{fill<Real>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{fill<Real>{1}}; }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randu(Real min = 0, Real max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(Real mean = 0, Real stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_normal<Real>{eng, mean, stddev}};
  }
  //----------------------------------------------------------------------------
  template <typename OtherReal>
  auto operator==(const tensor<OtherReal, Dims...>& other) const {
    return this->data() == other.data();
  }
  //----------------------------------------------------------------------------
  template <typename OtherReal>
  auto operator<(const tensor<OtherReal, Dims...>& other) const {
    return this->data() < other.data();
  }
  //============================================================================
  template <typename F>
  auto unary_operation(F&& f) -> auto& {
    array_parent_t::unary_operation(std::forward<F>(f));
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherTensor, typename OtherReal>
  auto binary_operation(
      F&& f, const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> decltype(auto) {
    tensor_parent_t::binary_operation(std::forward<F>(f), other);
    return *this;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <size_t C, typename... Rows>
tensor(Rows const(&&... rows)[C])  // NOLINT
    ->tensor<promote_t<Rows...>, sizeof...(Rows), C>;
//==============================================================================
template <typename Real, size_t N>
struct vec : tensor<Real, N> {  // NOLINT
  using parent_t = tensor<Real, N>;
  using parent_t::at;
  using parent_t::dimension;
  using parent_t::num_dimensions;
  using parent_t::parent_t;
  using parent_t::operator();

  template <typename... Ts, size_t _Dim0 = parent_t::dimension(0),
            std::enable_if_t<_Dim0 == sizeof...(Ts), bool> = true>
  explicit constexpr vec(const Ts&... ts) : parent_t{ts...} {}

  using iterator = typename parent_t::array_parent_t::container_t::iterator;
  using const_iterator =
      typename parent_t::array_parent_t::container_t::const_iterator;

  //----------------------------------------------------------------------------
  constexpr vec(const vec&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  explicit constexpr vec(vec&& other) noexcept : parent_t{std::move(other)} {}
  //----------------------------------------------------------------------------
  constexpr auto operator=(const vec&) -> vec& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  constexpr auto operator=(vec&& other) noexcept -> vec& {
    parent_t::operator=(std::move(other));
    return *this;
  }
  template <typename OtherTensor, typename OtherReal>
  constexpr vec(const base_tensor<OtherTensor, OtherReal, N>& other) {
    for (size_t i = 0; i < N; ++i) { at(i) = other(i); }
  }
  //----------------------------------------------------------------------------
  ~vec() = default;
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
vec(const Ts&...) -> vec<promote_t<Ts...>, sizeof...(Ts)>;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
using vec2 = vec<double, 2>;
using vec3 = vec<double, 3>;
using vec4 = vec<double, 4>;

//==============================================================================
template <typename Real, size_t M, size_t N>
struct mat : tensor<Real, M, N> {  // NOLINT
  using this_t   = mat<Real, M, N>;
  using parent_t = tensor<Real, M, N>;
  using parent_t::parent_t;
  //----------------------------------------------------------------------------
  constexpr mat(const mat&) = default;
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal
#if TATOOINE_GINAC_AVAILABLE
            , enable_if_arithmetic_or_complex<TensorReal> = true
#endif
            >
  constexpr mat(const base_tensor<Tensor, TensorReal, M, N>& other)
      : parent_t{other} {}
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  explicit constexpr mat(mat&& other) noexcept : parent_t{std::move(other)} {}
#else
  constexpr mat(mat&& other) noexcept = default;
#endif
  //----------------------------------------------------------------------------
  constexpr auto operator=(const mat&) -> mat& = default;
  //----------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
  template <typename Real_                         = Real,
            enable_if_arithmetic_or_complex<Real_> = true>
  constexpr auto operator=(mat&& other) noexcept -> mat& {
    parent_t::operator=(std::move(other));
    return *this;
  }
#else
  constexpr auto operator=(mat&& other) noexcept -> mat& = default;
#endif
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal
#if TATOOINE_GINAC_AVAILABLE
            , enable_if_arithmetic_or_complex<TensorReal> = true
#endif
            >
  constexpr auto operator=(
      const base_tensor<Tensor, TensorReal, M, N>& other) noexcept -> mat& {
    parent_t::operator=(other);
    return *this;
  }

  //----------------------------------------------------------------------------
  constexpr mat(eye_t /*flag*/) : parent_t{zeros} {
    for (size_t i = 0; i < std::min(M, N); ++i) { this->at(i, i) = 1; }
  }
  //----------------------------------------------------------------------------
  ~mat() = default;
  //----------------------------------------------------------------------------
  static constexpr auto eye() {
    return this_t{tatooine::eye};
  }

#if TATOOINE_GINAC_AVAILABLE
  template <typename... Rows, enable_if_arithmetic_or_symbolic<Rows...> = true>
#else
  template <typename... Rows, enable_if_arithmetic<Rows...> = true>
#endif
  constexpr mat(Rows(&&... rows)[parent_t::dimension(1)]) {  // NOLINT
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](const auto& row) mutable {
      for (size_t c = 0; c < parent_t::dimension(1); ++c) {
        this->at(r, c) = static_cast<Real>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
  }
  //------------------------------------------------------------------------------
  constexpr auto row(size_t i) { return this->template slice<0>(i); }
  constexpr auto row(size_t i) const { return this->template slice<0>(i); }
  //------------------------------------------------------------------------------
  constexpr auto col(size_t i) { return this->template slice<1>(i); }
  constexpr auto col(size_t i) const { return this->template slice<1>(i); }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <size_t C, typename... Rows>
mat(Rows const(&&... rows)[C])  // NOLINT
    ->mat<promote_t<Rows...>, sizeof...(Rows), C>;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
using mat2 = mat<double, 2, 2>;
using mat3 = mat<double, 3, 3>;
using mat4 = mat<double, 4, 4>;

//==============================================================================
// operations
//==============================================================================
/// invert symmetric matrix
/// A = [a,b]
///     [b,c]
template <typename Tensor, typename Real>
constexpr auto inv_sym(const base_tensor<Tensor, Real, 2, 2>& A) {
  const auto& a = A(0, 0);
  const auto& b = A(1, 0);
  const auto& c = A(1, 1);
  const auto  d = 1 / (a * c - b * b);
  const auto  e = -b * d;
  return mat{{c * d, e    },
             {e,     a * d}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b]
///     [c,d]
template <typename Tensor, typename Real>
constexpr auto inv(const base_tensor<Tensor, Real, 2, 2>& A) {
  const auto& b = A(0, 1);
  const auto& c = A(1, 0);
  if (std::abs(b - c) < 1e-10) { return inv_sym(A); }
  const auto& a = A(0, 0);
  const auto& d = A(1, 1);
  const auto  e = 1/(a * d - b * c);
  return mat{{ d * e, -b * e},
             {-c * e,  a * e}};
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c]
///     [b,d,e]
///     [c,e,f]
template <typename Tensor, typename Real>
constexpr auto inv_sym(const base_tensor<Tensor, Real, 3, 3>& A) {
  const auto& a = A(0, 0);
  const auto& b = A(1, 0);
  const auto& c = A(2, 0);
  const auto& d = A(1, 1);
  const auto& e = A(2, 1);
  const auto& f = A(2, 2);
  const auto  div =
      1 / ((a * d - b * b) * f - a * e * e + 2 * b * c * e - c * c * d);
  return mat{
      { (d * f - e * e) * div, -(b * f - c * e) * div,  (b * e - c * d) * div},
      {-(b * f - c * e) * div,  (a * f - c * c) * div, -(a * e - b * c) * div},
      { (b * e - c * d) * div, -(a * e - b * c) * div,  (a * d - b * b) * div}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b,c]
///     [d,e,f]
///     [g,h,i]
template <typename Tensor, typename Real>
constexpr auto inv(const base_tensor<Tensor, Real, 3, 3>& A) {
  const auto& b = A(0, 1);
  const auto& c = A(0, 2);
  const auto& d = A(1, 0);
  const auto& g = A(2, 0);
  const auto& f = A(1, 2);
  const auto& h = A(2, 1);
  if (std::abs(b - d) < 1e-10 && std::abs(c - g) < 1e-10 &&
      std::abs(f - h) < 1e-10) {
    return inv_sym(A);
  }
  const auto& a = A(0, 0);
  const auto& e = A(1, 1);
  const auto& i = A(2, 2);
  const auto  div =
      1 / ((a * e - b * d) * i + (c * d - a * f) * h + (b * f - c * e) * g);
  return mat{
      { (e * i - f * h) * div, -(b * i - c * h) * div,  (b * f - c * e) * div},
      {-(d * i - f * g) * div,  (a * i - c * g) * div, -(a * f - c * d) * div},
      { (d * h - e * g) * div, -(a * h - b * g) * div,  (a * e - b * d) * div}};
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c,d]
///     [b,e,f,g]
///     [c,f,h,i]
///     [d,g,i,j]
template <typename Tensor, typename Real>
constexpr auto inv_sym(const base_tensor<Tensor, Real, 4, 4>& A) {
  const auto& a = A(0, 0);
  const auto& b = A(1, 0);
  const auto& c = A(2, 0);
  const auto& d = A(3, 0);
  const auto& e = A(1, 1);
  const auto& f = A(2, 1);
  const auto& g = A(3, 1);
  const auto& h = A(2, 2);
  const auto& i = A(3, 2);
  const auto& j = A(3, 3);
  const auto  div =
      1 / (((a * e - b * b) * h - a * f * f + 2 * b * c * f - c * c * e) * j +
           (b * b - a * e) * i * i +
           ((2 * a * f - 2 * b * c) * g - 2 * b * d * f + 2 * c * d * e) * i +
           (-a * g * g + 2 * b * d * g - d * d * e) * h + c * c * g * g -
           2 * c * d * f * g + d * d * f * f);
  return mat{
      {((e * h - f * f) * j - e * i * i + 2 * f * g * i - g * g * h) * div,
       -((b * h - c * f) * j - b * i * i + (c * g + d * f) * i - d * g * h) *
           div,
       ((b * f - c * e) * j + (d * e - b * g) * i + c * g * g - d * f * g) *
           div,
       -((b * f - c * e) * i + (d * e - b * g) * h + c * f * g - d * f * f) *
           div},
      {-((b * h - c * f) * j - b * i * i + (c * g + d * f) * i - d * g * h) *
           div,
       ((a * h - c * c) * j - a * i * i + 2 * c * d * i - d * d * h) * div,
       -((a * f - b * c) * j + (b * d - a * g) * i + c * d * g - d * d * f) *
           div,
       ((a * f - b * c) * i + (b * d - a * g) * h + c * c * g - c * d * f) *
           div},
      {((b * f - c * e) * j + (d * e - b * g) * i + c * g * g - d * f * g) *
           div,
       -((a * f - b * c) * j + (b * d - a * g) * i + c * d * g - d * d * f) *
           div,
       ((a * e - b * b) * j - a * g * g + 2 * b * d * g - d * d * e) * div,
       -((a * e - b * b) * i + (b * c - a * f) * g + b * d * f - c * d * e) *
           div},
      {-((b * f - c * e) * i + (d * e - b * g) * h + c * f * g - d * f * f) *
           div,
       ((a * f - b * c) * i + (b * d - a * g) * h + c * c * g - c * d * f) *
           div,
       -((a * e - b * b) * i + (b * c - a * f) * g + b * d * f - c * d * e) *
           div,
       ((a * e - b * b) * h - a * f * f + 2 * b * c * f - c * c * e) * div}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b,c,d]
///     [e,f,g,h]
///     [i,j,k,l]
///     [m,n,o,p]
template <typename Tensor, typename Real>
constexpr auto inv(const base_tensor<Tensor, Real, 4, 4>& A) {
  const auto& b = A(0, 1);
  const auto& c = A(0, 2);
  const auto& d = A(0, 3);
  const auto& e = A(1, 0);
  const auto& g = A(1, 2);
  const auto& h = A(1, 3);
  const auto& i = A(2, 0);
  const auto& j = A(2, 1);
  const auto& l = A(2, 3);
  const auto& m = A(3, 0);
  const auto& n = A(3, 1);
  const auto& o = A(3, 2);

  if (std::abs(b - e) < 1e-10 &&
      std::abs(c - i) < 1e-10 &&
      std::abs(d - m) < 1e-10 &&
      std::abs(g - j) < 1e-10 &&
      std::abs(h - n) < 1e-10 &&
      std::abs(l - o) < 1e-10) {
    return inv_sym(A);
  }

  const auto& a = A(0, 0);
  const auto& f = A(1, 1);
  const auto& k = A(2, 2);
  const auto& p = A(3, 3);
  const auto  div =
      1 /
      ((((a * f - b * e) * k + (c * e - a * g) * j + (b * g - c * f) * i) * p +
        ((b * e - a * f) * l + (a * h - d * e) * j + (d * f - b * h) * i) * o +
        ((a * g - c * e) * l + (d * e - a * h) * k + (c * h - d * g) * i) * n +
        ((c * f - b * g) * l + (b * h - d * f) * k + (d * g - c * h) * j) * m));
  return mat{
      {((f * k - g * j) * p + (h * j - f * l) * o + (g * l - h * k) * n) * div,
       -((b * k - c * j) * p + (d * j - b * l) * o + (c * l - d * k) * n) * div,
       ((b * g - c * f) * p + (d * f - b * h) * o + (c * h - d * g) * n) * div,
       -((b * g - c * f) * l + (d * f - b * h) * k + (c * h - d * g) * j) *
           div},
      {-((e * k - g * i) * p + (h * i - e * l) * o + (g * l - h * k) * m) * div,
       ((a * k - c * i) * p + (d * i - a * l) * o + (c * l - d * k) * m) * div,
       -((a * g - c * e) * p + (d * e - a * h) * o + (c * h - d * g) * m) * div,
       ((a * g - c * e) * l + (d * e - a * h) * k + (c * h - d * g) * i) * div},
      {((e * j - f * i) * p + (h * i - e * l) * n + (f * l - h * j) * m) * div,
       -((a * j - b * i) * p + (d * i - a * l) * n + (b * l - d * j) * m) * div,
       ((a * f - b * e) * p + (d * e - a * h) * n + (b * h - d * f) * m) * div,
       -((a * f - b * e) * l + (d * e - a * h) * j + (b * h - d * f) * i) *
           div},
      {-((e * j - f * i) * o + (g * i - e * k) * n + (f * k - g * j) * m) * div,
       ((a * j - b * i) * o + (c * i - a * k) * n + (b * k - c * j) * m) * div,
       -((a * f - b * e) * o + (c * e - a * g) * n + (b * g - c * f) * m) * div,
       ((a * f - b * e) * k + (c * e - a * g) * j + (b * g - c * f) * i) *
           div}};
}
//==============================================================================
/// Returns the cosine of the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
constexpr auto cos_angle(const base_tensor<Tensor0, Real0, N>& v0,
                         const base_tensor<Tensor1, Real1, N>& v1) {
  return dot(normalize(v0), normalize(v1));
}

//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
auto angle(const base_tensor<Tensor0, Real0, N>& v0,
           const base_tensor<Tensor1, Real1, N>& v1) {
  return std::acos(cos_angle(v0, v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
auto min_angle(const base_tensor<Tensor0, Real0, N>& v0,
               const base_tensor<Tensor1, Real1, N>& v1) {
  return std::min(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
auto max_angle(const base_tensor<Tensor0, Real0, N>& v0,
               const base_tensor<Tensor1, Real1, N>& v1) {
  return std::max(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Real0,
          typename Real1, typename Real2, size_t N>
constexpr auto cos_angle(const base_tensor<Tensor0, Real0, N>& v0,
                         const base_tensor<Tensor1, Real1, N>& v1,
                         const base_tensor<Tensor2, Real2, N>& v2) {
  return cos_angle(v0 - v1, v2 - v1);
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Real0,
          typename Real1, typename Real2, size_t N>
auto angle(const base_tensor<Tensor0, Real0, N>& v0,
           const base_tensor<Tensor1, Real1, N>& v1,
           const base_tensor<Tensor2, Real2, N>& v2) {
  return std::acos(cos_angle(v0, v1, v2));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto min(const base_tensor<Tensor, Real, Dims...>& t) {
  Real m = std::numeric_limits<Real>::max();
  t.for_indices([&](const auto... is) { m = std::min(m, t(is...)); });
  return m;
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto max(const base_tensor<Tensor, Real, Dims...>& t) {
  Real m = -std::numeric_limits<Real>::max();
  t.for_indices([&](const auto... is) { m = std::max(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto norm(const base_tensor<Tensor, Real, N>& t, unsigned p = 2)
    -> Real {
  Real n = 0;
  for (size_t i = 0; i < N; ++i) { n += std::pow(t(i), p); }
  return std::pow(n, Real(1) / Real(p));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto norm_inf(const base_tensor<Tensor, Real, N>& t) -> Real {
  Real norm = -std::numeric_limits<Real>::max();
  for (size_t i = 0; i < N; ++i) { norm = std::max(norm, std::abs(t(i))); }
  return norm;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto norm1(const base_tensor<Tensor, Real, N>& t) {
  return sum(abs(t));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto sqr_length(const base_tensor<Tensor, Real, N>& t_in) {
  return dot(t_in, t_in);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto length(const base_tensor<Tensor, Real, N>& t_in) -> Real {
  return std::sqrt(sqr_length(t_in));
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto sqr_norm(const base_tensor<Tensor, Real, M, N>& mat,
                        frobenius_t) {
  Real n = 0;
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < M; ++i) { n += std::abs(mat(i, j)); }
  }
  return n;
}
//------------------------------------------------------------------------------
/// Frobenius norm of a rank-2 tensor
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto norm(const base_tensor<Tensor, Real, M, N>& mat, frobenius_t) {
  return std::sqrt(sqr_norm(mat, frobenius));
}
//------------------------------------------------------------------------------
/// 1-norm of a MxN Tensor
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto norm1(const base_tensor<Tensor, Real, M, N>& mat) {
  Real       max    = -std::numeric_limits<Real>::max();
  const auto absmat = abs(mat);
  for (size_t i = 0; i < N; ++i) {
    max = std::max(max, sum(absmat.template slice<1>(i)));
  }
  return max;
}
//------------------------------------------------------------------------------
/// infinity-norm of a MxN tensor
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto norm_inf(const base_tensor<Tensor, Real, M, N>& mat) {
  Real max = -std::numeric_limits<Real>::max();
  for (size_t i = 0; i < M; ++i) {
    max = std::max(max, sum(abs(mat.template slice<0>(i))));
  }
  return max;
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto sqr_norm(const base_tensor<Tensor, Real, M, N>& mat) {
  return sqr_norm(mat, frobenius);
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto norm(const base_tensor<Tensor, Real, M, N>& mat) {
  return norm(mat, frobenius);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto normalize(const base_tensor<Tensor, Real, N>& t_in)
    -> vec<Real, N> {
  const auto l = length(t_in);
  if (std::abs(l) < 1e-13) { return vec<Real, N>::zeros(); }
  return t_in / l;
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t N>
constexpr auto distance(const base_tensor<Tensor0, Real0, N>& lhs,
                        const base_tensor<Tensor1, Real1, N>& rhs) {
  return length(rhs - lhs);
}

//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename Tensor, typename Real, size_t VecDim>
constexpr auto sum(const base_tensor<Tensor, Real, VecDim>& v) {
  Real s = 0;
  for (size_t i = 0; i < VecDim; ++i) { s += v(i); }
  return s;
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t N>
constexpr auto dot(const base_tensor<Tensor0, Real0, N>& lhs,
                   const base_tensor<Tensor1, Real1, N>& rhs) {
  promote_t<Real0, Real1> d = 0;
  for (size_t i = 0; i < N; ++i) { d += lhs(i) * rhs(i); }
  return d;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real>
constexpr auto det(const base_tensor<Tensor, Real, 2, 2>& m) -> Real {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real>
constexpr auto detAtA(const base_tensor<Tensor, Real, 2, 2>& m) -> Real {
  return m(0, 0) * m(0, 0) * m(1, 1) * m(1, 1) +
         m(0, 1) * m(0, 1) * m(1, 0) * m(1, 0) -
         2 * m(0, 0) * m(1, 0) * m(0, 1) * m(1, 1);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real>
constexpr auto det(const base_tensor<Tensor, Real, 3, 3>& m) -> Real {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1>
constexpr auto cross(const base_tensor<Tensor0, Real0, 3>& lhs,
                     const base_tensor<Tensor1, Real1, 3>& rhs) {
  return vec<promote_t<Real0, Real1>, 3>{lhs(1) * rhs(2) - lhs(2) * rhs(1),
                                         lhs(2) * rhs(0) - lhs(0) * rhs(2),
                                         lhs(0) * rhs(1) - lhs(1) * rhs(0)};
}

//------------------------------------------------------------------------------
template <typename F, typename Tensor, typename Real, size_t N>
constexpr auto unary_operation(F&&                                 f,
                               const base_tensor<Tensor, Real, N>& t_in) {
  using RealOut         = typename std::result_of<decltype(f)(Real)>::type;
  vec<RealOut, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, typename Real, size_t M, size_t N>
constexpr auto unary_operation(F&&                                    f,
                               const base_tensor<Tensor, Real, M, N>& t_in) {
  using RealOut            = typename std::result_of<decltype(f)(Real)>::type;
  mat<RealOut, M, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, typename Real, size_t... Dims>
constexpr auto unary_operation(F&&                                       f,
                               const base_tensor<Tensor, Real, Dims...>& t_in) {
  using RealOut = typename std::result_of<decltype(f)(Real)>::type;
  tensor<RealOut, Dims...> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename Tensor0, typename Real0, typename Tensor1,
          typename Real1, size_t N>
constexpr auto binary_operation(F&&                                   f,
                                const base_tensor<Tensor0, Real0, N>& lhs,
                                const base_tensor<Tensor1, Real1, N>& rhs) {
  using RealOut = typename std::result_of<decltype(f)(Real0, Real1)>::type;
  vec<RealOut, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, typename Real0, typename Tensor1,
          typename Real1, size_t M, size_t N>
constexpr auto binary_operation(F&&                                      f,
                                const base_tensor<Tensor0, Real0, M, N>& lhs,
                                const base_tensor<Tensor1, Real1, M, N>& rhs) {
  using RealOut = typename std::result_of<decltype(f)(Real0, Real1)>::type;
  mat<RealOut, M, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, typename Real0, typename Tensor1,
          typename Real1, size_t... Dims>
constexpr auto binary_operation(
    F&& f, const base_tensor<Tensor0, Real0, Dims...>& lhs,
    const base_tensor<Tensor1, Real1, Dims...>& rhs) {
  using RealOut = typename std::result_of<decltype(f)(Real0, Real1)>::type;
  tensor<RealOut, Dims...> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto operator-(const base_tensor<Tensor, Real, Dims...>& t) {
  return unary_operation([](const auto& c) { return -c; }, t);
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Real1, size_t... Dims,
          enable_if_arithmetic<Real1> = true>
constexpr auto operator+(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                         Real1                                       scalar) {
  return unary_operation([scalar](const auto& c) { return c + scalar; }, lhs);
}

//------------------------------------------------------------------------------
/// matrix-matrix multiplication
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t M, size_t N, size_t O>
constexpr auto operator*(const base_tensor<Tensor0, Real0, M, N>& lhs,
                         const base_tensor<Tensor1, Real1, N, O>& rhs) {
  mat<promote_t<Real0, Real1>, M, O> product;
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < O; ++c) {
      product(r, c) = dot(lhs.template slice<0>(r), rhs.template slice<1>(c));
    }
  }
  return product;
}

//------------------------------------------------------------------------------
/// component-wise multiplication
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t... Dims, std::enable_if_t<(sizeof...(Dims) != 2), bool> = true>
constexpr auto operator%(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                         const base_tensor<Tensor1, Real1, Dims...>& rhs) {
  return binary_operation(std::multiplies<promote_t<Real0, Real1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t... Dims>
constexpr auto operator/(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                         const base_tensor<Tensor1, Real1, Dims...>& rhs) {
  return binary_operation(std::divides<promote_t<Real0, Real1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t... Dims>
constexpr auto operator+(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                         const base_tensor<Tensor1, Real1, Dims...>& rhs) {
  return binary_operation(std::plus<promote_t<Real0, Real1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims,
          enable_if_arithmetic_complex_or_symbolic<ScalarReal> = true>
#else
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims, enable_if_arithmetic_or_complex<ScalarReal> = true>
#endif
constexpr auto operator*(const base_tensor<Tensor, TensorReal, Dims...>& t,
                         const ScalarReal scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#if TATOOINE_GINAC_AVAILABLE
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims,
          enable_if_arithmetic_complex_or_symbolic<ScalarReal> = true>
#else
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims, enable_if_arithmetic_or_complex<ScalarReal> = true>
#endif
constexpr auto operator*(const ScalarReal                                scalar,
                         const base_tensor<Tensor, TensorReal, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}

//------------------------------------------------------------------------------
#if TATOOINE_GINAC_AVAILABLE
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims,
          enable_if_arithmetic_complex_or_symbolic<ScalarReal> = true>
#else
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims, enable_if_arithmetic_or_complex<ScalarReal> = true>
#endif
constexpr auto operator/(const base_tensor<Tensor, TensorReal, Dims...>& t,
                         const ScalarReal scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#if TATOOINE_GINAC_AVAILABLE
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims,
          enable_if_arithmetic_complex_or_symbolic<ScalarReal> = true>
#else
template <typename Tensor, typename TensorReal, typename ScalarReal,
          size_t... Dims, enable_if_arithmetic_or_complex<ScalarReal> = true>
#endif
constexpr auto operator/(const ScalarReal                                scalar,
                         const base_tensor<Tensor, TensorReal, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return scalar / component; }, t);
}

//------------------------------------------------------------------------------
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t... Dims>
constexpr auto operator-(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                         const base_tensor<Tensor1, Real1, Dims...>& rhs) {
  return binary_operation(std::minus<promote_t<Real0, Real1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t M, size_t N>
constexpr auto operator*(const base_tensor<Tensor0, Real0, M, N>& lhs,
                         const base_tensor<Tensor1, Real1, N>&    rhs) {
  vec<promote_t<Real0, Real1>, M> product;
  for (size_t i = 0; i < M; ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}
//------------------------------------------------------------------------------
/// vector-matrix-multiplication
template <typename Tensor0, typename Real0, typename Tensor1, typename Real1,
          size_t M, size_t N>
constexpr auto operator*(const base_tensor<Tensor0, Real0, M>&    lhs,
                         const base_tensor<Tensor1, Real1, M, N>& rhs) {
  vec<promote_t<Real0, Real1>, N> product;
  for (size_t i = 0; i < N; ++i) {
    product(i) = dot(lhs, rhs.template slice<1>(i));
  }
  return product;
}
//------------------------------------------------------------------------------
namespace lapack_job {
//------------------------------------------------------------------------------
struct A_t{};
struct S_t{};
struct O_t{};
struct N_t{};
static constexpr A_t A;
static constexpr S_t S;
static constexpr O_t O;
static constexpr N_t N;
//------------------------------------------------------------------------------
}  // namespace lapack_job
//------------------------------------------------------------------------------
namespace lapack {
//------------------------------------------------------------------------------
template <typename T, size_t M, size_t N,
          enable_if_floating_point_or_complex<T> = true>
auto getrf(tensor<T, M, N>&& A) {
  vec<int, tatooine::min(M,N)> p;
  if constexpr (std::is_same_v<double, T>) {
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (std::is_same_v<float, T>) {
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (std::is_same_v<std::complex<double>, T>) {
    LAPACKE_zgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (std::is_same_v<std::complex<float>, T>) {
    LAPACKE_cgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else {
    throw std::runtime_error{"[tatooine::lapack::getrf] - type not accepted"};
  }
  return A;
}
template <typename T, size_t M, size_t N,
          enable_if_floating_point_or_complex<T> = true>
auto getrf(tensor<T, M, N>& A) {
  vec<int, tatooine::min(M,N)> p;
  if constexpr (std::is_same_v<double, T>) {
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (std::is_same_v<float, T>) {
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (std::is_same_v<std::complex<double>, T>) {
    LAPACKE_zgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (std::is_same_v<std::complex<float>, T>) {
    LAPACKE_cgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  }
  return A;
}
//------------------------------------------------------------------------------
template <size_t N>
auto gesv(tensor<float, N, N> A, tensor<float, N> b) {
  std::array<int, N> ipiv;
  int                nrhs = 1;
  LAPACKE_sgesv(LAPACK_COL_MAJOR, N, nrhs, A.data_ptr(), N, ipiv.data(),
                b.data_ptr(), N);
  return b;
}
//------------------------------------------------------------------------------
template <size_t N>
auto gesv(tensor<double, N, N> A, tensor<double, N> b) {
  vec<int, N> ipiv;
  int                nrhs = 1;
  LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, A.data_ptr(), N, ipiv.data_ptr(),
                b.data_ptr(), N);
  return b;
}
//------------------------------------------------------------------------------
template <size_t M, size_t N>
auto gesv(tensor<float, M, M> A, const tensor<float, M, N>& B) {
  auto               X = B;
  vec<int, N> ipiv;
  LAPACKE_sgesv(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, ipiv.data_ptr(),
                X.data_ptr(), M);
  return X;
}
template <size_t M, size_t N>
auto gesv(tensor<double, M, M> A, const tensor<double, M, N>& B) {
  auto               X = B;
  std::array<int, N> ipiv;
  LAPACKE_dgesv(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, ipiv.data(),
                X.data_ptr(), M);
  return X;
}
//------------------------------------------------------------------------------
template <typename T, size_t M, size_t N,
          enable_if_floating_point_or_complex<T> = true>
auto lange(const tensor<T, M, N>& A, const char norm) {
  if constexpr (std::is_same_v<double, T>) {
    return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else if constexpr (std::is_same_v<float, T>) {
    return LAPACKE_slange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else if constexpr (std::is_same_v<std::complex<double>, T>) {
    return LAPACKE_zlange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else if constexpr (std::is_same_v<std::complex<float>, T>) {
    return LAPACKE_clange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else {
    throw std::runtime_error{"[tatooine::lapack::lange] - type not accepted"};
  }
}
//------------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number of a general matrix A.
/// http://www.netlib.org/lapack/explore-html/d7/db5/lapacke__dgecon_8c_a7c007823b949b0b118acf7e0235a6fc5.html
/// https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga188b8d30443d14b1a3f7f8331d87ae60.html
template <typename T, size_t N,
          enable_if_floating_point_or_complex<T> = true>
auto gecon(tensor<T, N, N>&& A) {
  T              rcond = 0;
  constexpr char p     = '1';
  const auto     n     = lange(A, p);
  getrf(A);
  const auto info = [&] {
    if constexpr (std::is_same_v<double, T>) {
      return LAPACKE_dgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else if constexpr (std::is_same_v<float, T>) {
      return LAPACKE_sgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
      return LAPACKE_cgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else if constexpr (std::is_same_v<std::complex<double>, T>) {
      return LAPACKE_zgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else {
      throw std::runtime_error{"[tatooine::lapack::gecon] - type not accepted"};
    }
  }();
  if (info < 0) {
    throw std::runtime_error{"[tatooine::lapack::gecon] - " + std::to_string(-info) +
                             "-th argument is invalid"};
  }
  return rcond;
}
/// Estimates the reciprocal of the condition number of a general matrix A.
/// http://www.netlib.org/lapack/explore-html/d7/db5/lapacke__dgecon_8c_a7c007823b949b0b118acf7e0235a6fc5.html
/// https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga188b8d30443d14b1a3f7f8331d87ae60.html
template <typename T, size_t N,
          enable_if_floating_point_or_complex<T> = true>
auto gecon(tensor<T, N, N>& A) {
  T              rcond = 0;
  constexpr char p     = 'I';
  getrf(A);
  const auto     info  = [&] {
    if constexpr (std::is_same_v<double, T>) {
      return LAPACKE_dgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else if constexpr (std::is_same_v<float, T>) {
      return LAPACKE_sgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
      return LAPACKE_cgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else if constexpr (std::is_same_v<std::complex<double>, T>) {
      return LAPACKE_zgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else {
      throw std::runtime_error{"[tatooine::lapack::gecon] - type not accepted"};
    }
  }();
  if (info < 0) {
    throw std::runtime_error{"[tatooine::lapack::gecon] - " + std::to_string(-info) +
                             "-th argument is invalid"};
  }
  return rcond;
}
//------------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number of a general matrix A.
/// http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
/// http://www.netlib.org/lapack/explore-html/d0/dee/lapacke__dgesvd_8c_af31b3cb47f7cc3b9f6541303a2968c9f.html
template <typename T, size_t M, size_t N,
          typename JOBU, typename JOBVT,
          enable_if_floating_point_or_complex<T> = true>
auto gesvd(tensor<T, M, N>&& A, JOBU, JOBVT) {
  static_assert(!std::is_same_v<JOBU,  lapack_job::O_t> ||
                !std::is_same_v<JOBVT, lapack_job::O_t>,
                "either jobu or jobvt must not be O");
  vec<T, tatooine::min(M, N)> s;
  constexpr char jobu = [&]{
    if constexpr (std::is_same_v<lapack_job::A_t, JOBU>) {
      return 'A';
    } else if constexpr (std::is_same_v<lapack_job::S_t, JOBU>) {
      return 'S';
    } else if constexpr (std::is_same_v<lapack_job::O_t, JOBU>) {
      return 'O';
    } else if constexpr (std::is_same_v<lapack_job::N_t, JOBU>) {
      return 'N';
    } else {
      return '\0';
    }
  }();
  constexpr char jobvt = [&] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBVT>) {
      return 'A';
    } else if constexpr (std::is_same_v<lapack_job::S_t, JOBVT>) {
      return 'S';
    } else if constexpr (std::is_same_v<lapack_job::O_t, JOBVT>) {
      return 'O';
    } else if constexpr (std::is_same_v<lapack_job::N_t, JOBVT>) {
      return 'N';
    } else {
      return '\0';
    }
  }();

  auto U = [] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBU>) {
      return mat<T, M, M>{};
    } else if constexpr (std::is_same_v<lapack_job::S_t, JOBU>) {
      return mat<T, M, tatooine::min(M, N)>{};
    } else {
      return nullptr;
    }}();

  auto VT = [] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBVT>) {
      return mat<T, N, N>{};
    } else if constexpr (std::is_same_v<lapack_job::S_t, JOBVT>) {
      return mat<T, tatooine::min(M,N), N>{};
    } else {
      return nullptr;
    }}();
  constexpr auto ldu = [&U] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBU> ||
                  std::is_same_v<lapack_job::S_t, JOBU>) {
      return U.dimension(0);
    } else {
      return 1;
    }}();
  constexpr auto ldvt = [&VT] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBVT> ||
                  std::is_same_v<lapack_job::S_t, JOBVT>) {
      return VT.dimension(0);
    } else {
      return 1;
    }}();
  T* U_ptr = [&U] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBU> ||
                  std::is_same_v<lapack_job::S_t, JOBU>) {
      return U.data_ptr();
    } else {
      return nullptr;
    }}();
  T* VT_ptr = [&VT] {
    if constexpr (std::is_same_v<lapack_job::A_t, JOBVT> ||
                  std::is_same_v<lapack_job::S_t, JOBVT>) {
      return VT.data_ptr();
    } else {
      return nullptr;
    }}();
  std::array<T, tatooine::min(M, N) - 1> superb;

  const auto info = [&] {
    if constexpr (std::is_same_v<double, T>) {
      return LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr,
                            ldvt, superb.data());
    } else if constexpr (std::is_same_v<float, T>) {
      return LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr,
                            ldvt, superb.data());
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
      return LAPACKE_cgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr,
                            ldvt, superb.data());
    } else if constexpr (std::is_same_v<std::complex<double>, T>) {
      return LAPACKE_zgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr,
                            ldvt, superb.data());
    }
  }();
  if (info < 0) {
    throw std::runtime_error{"[tatooine::lapack::gesvd] - " + std::to_string(-info) +
                             "-th argument is invalid"};
  } else if (info > 0) {
    throw std::runtime_error{"[tatooine::lapack::gesvd] - DBDSQR did not converge. " +
                             std::to_string(info) +
                             " superdiagonals of an intermediate bidiagonal "
                             "form B did not converge to zero."};
  }
  if constexpr (std::is_same_v<lapack_job::N_t, JOBU>) {
    if constexpr (std::is_same_v<lapack_job::N_t, JOBVT>) {
      return s;
    } else {
      return std::tuple{s, VT};
    }
  } else {
    if constexpr (std::is_same_v<lapack_job::N_t, JOBVT>) {
      return std::tuple{U, s};
    } else {
      return std::tuple{U, s, VT};
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace lapack
//------------------------------------------------------------------------------
/// compute condition number
template <typename T, size_t N, typename P, enable_if_integral<P> = true>
auto condition_number(const tensor<T, N, N>& A, P p = 2) {
  if (p == 1) {
    return 1 / lapack::gecon(tensor{A});
  } else if (p == 2) {
    const auto s = singular_values(A);
    return s(0) / s(N-1);
  } else {
    throw std::runtime_error {
      "p = " + std::to_string(p) + " is no valid base. p must be either 1 or 2."
    };
  }
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N, typename PReal>
auto condition_number(const base_tensor<Tensor, T, N, N>& A, PReal p) {
  return condition_number(tensor{A}, p);
}
//==============================================================================
template <size_t N>
auto eigenvalues(tensor<float, N, N> A) -> vec<std::complex<float>, N> {
  [[maybe_unused]] lapack_int info;
  std::array<float, N>        wr;
  std::array<float, N>        wi;
  info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, nullptr, N);

  vec<std::complex<float>, N> vals;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  return vals;
}
template <size_t N>
auto eigenvalues(tensor<double, N, N> A) -> vec<std::complex<double>, N> {
  [[maybe_unused]] lapack_int info;
  std::array<double, N>       wr;
  std::array<double, N>       wi;
  info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, nullptr, N);
  vec<std::complex<double>, N> vals;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  return vals;
}
//------------------------------------------------------------------------------
template <size_t N>
auto eigenvectors(tensor<float, N, N> A)
    -> std::pair<mat<std::complex<float>, N, N>, vec<std::complex<float>, N>> {
  [[maybe_unused]] lapack_int info;
  std::array<float, N>        wr;
  std::array<float, N>        wi;
  std::array<float, N * N>    vr;
  info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, vr.data(), N);

  vec<std::complex<float>, N>    vals;
  mat<std::complex<float>, N, N> vecs;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      if (wi[j] == 0) {
        vecs(i, j) = {vr[i + j * N], 0};
      } else {
        vecs(i, j)     = {vr[i + j * N], vr[i + (j + 1) * N]};
        vecs(i, j + 1) = {vr[i + j * N], -vr[i + (j + 1) * N]};
        if (i == N - 1) { ++j; }
      }
    }
  }

  return {std::move(vecs), std::move(vals)};
}
template <size_t N>
auto eigenvectors(tensor<double, N, N> A)
    -> std::pair<mat<std::complex<double>, N, N>,
                 vec<std::complex<double>, N>> {
  [[maybe_unused]] lapack_int info;
  std::array<double, N>       wr;
  std::array<double, N>       wi;
  std::array<double, N * N>   vr;
  info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, vr.data(), N);

  vec<std::complex<double>, N>    vals;
  mat<std::complex<double>, N, N> vecs;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      if (wi[j] == 0) {
        vecs(i, j) = {vr[i + j * N], 0};
      } else {
        vecs(i, j)     = {vr[i + j * N], vr[i + (j + 1) * N]};
        vecs(i, j + 1) = {vr[i + j * N], -vr[i + (j + 1) * N]};
        if (i == N - 1) { ++j; }
      }
    }
  }

  return {std::move(vecs), std::move(vals)};
}
//------------------------------------------------------------------------------
template <size_t N>
auto eigenvalues_sym(tensor<float, N, N> A) {
  vec<float, N>               vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'N', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());
  return vals;
}
template <size_t N>
auto eigenvalues_sym(tensor<double, N, N> A) {
  vec<double, N>              vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());

  return vals;
}

//------------------------------------------------------------------------------
template <size_t N>
auto eigenvectors_sym(mat<float, N, N> A)
    -> std::pair<mat<float, N, N>, vec<float, N>> {
  vec<float, N>               vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());
  return {std::move(A), std::move(vals)};
}
template <size_t N>
auto eigenvectors_sym(mat<double, N, N> A)
    -> std::pair<mat<double, N, N>, vec<double, N>> {
  vec<double, N>              vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());
  return {std::move(A), std::move(vals)};
}
//==============================================================================
template <typename T, size_t M, size_t N>
auto svd(const tensor<T, M, N>& A, full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd(const tensor<T, M, N>& A, economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd(const tensor<T, M, N>& A) {
  return svd(A, full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(const tensor<T, M, N>& A, full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(const tensor<T, M, N>& A, economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(const tensor<T, M, N>& A) {
  return svd_left(A, full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(const tensor<T, M, N>& A, full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(const tensor<T, M, N>& A, economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(const tensor<T, M, N>& A) {
  return svd_right(A, full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(const base_tensor<Tensor, T, M, N>& A, full_t /*tag*/) {
  tensor copy{A};
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(const base_tensor<Tensor, T, M, N>& A, economy_t /*tag*/) {
  tensor copy{A};
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(const base_tensor<Tensor, T, M, N>& A) {
  return svd(A, full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(const base_tensor<Tensor, T, M, N>& A, full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(const base_tensor<Tensor, T, M, N>& A, economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(const base_tensor<Tensor, T, M, N>& A) {
  return svd_left(A, full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(const base_tensor<Tensor, T, M, N>& A, full_t /*tag*/) {
  tensor copy{A};
  return gesvd(tensor{A}, lapack_job::N, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(const base_tensor<Tensor, T, M, N>& A, economy_t /*tag*/) {
  tensor copy{A};
  return gesvd(tensor{A}, lapack_job::N, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(const base_tensor<Tensor, T, M, N>& A) {
  return svd_right(A, full);
}
template <typename Tensor, typename T>
constexpr auto singular_values22(const base_tensor<Tensor, T, 2, 2>& A) {
  const auto a = A(0, 0);
  const auto b = A(0, 1);
  const auto c = A(1, 0);
  const auto d = A(1, 1);

  const auto aa = a * a;
  const auto bb = b * b;
  const auto cc = c * c;
  const auto dd = d * d;
  const auto s1 = aa + bb + cc + dd;
  const auto s2 = std::sqrt((aa + bb - cc - dd) * (aa + bb - cc - dd) +
                            4 * (a * c + b * d) * (a * c + b * d));
  const auto sigma1  = std::sqrt((s1 + s2) / 2);
  const auto sigma2  = std::sqrt((s1 - s2) / 2);
  return vec{tatooine::max(sigma1, sigma2),
             tatooine::min(sigma1, sigma2)};
}
//------------------------------------------------------------------------------
template <typename T, size_t M, size_t N>
constexpr auto singular_values(tensor<T, M, N>&& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return gesvd(A, lapack_job::N, lapack_job::N);
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
constexpr auto singular_values(const tensor<T, M, N>& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::N);
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto singular_values(const base_tensor<Tensor, T, M, N>& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return singular_values(tensor{A});
  }
}
//==============================================================================
/// for comparison
template <typename Tensor0, typename Real0,
          typename Tensor1, typename Real1,
          size_t... Dims,
          std::enable_if_t<std::is_floating_point<Real0>::value ||
                           std::is_floating_point<Real1>::value,
                           bool> = true>
constexpr auto approx_equal(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                            const base_tensor<Tensor1, Real1, Dims...>& rhs,
                            promote_t<Real0, Real1> eps = 1e-6) {
  bool equal = true;
  lhs.for_indices([&](const auto... is) {
    if (std::abs(lhs(is...) - rhs(is...)) > eps) { equal = false; }
  });
  return equal;
}

//==============================================================================
// views
//==============================================================================
template <typename Tensor, typename Real, size_t FixedDim, size_t... Dims>
struct tensor_slice : base_tensor<tensor_slice<Tensor, Real, FixedDim, Dims...>,
                                  Real, Dims...> {
  using tensor_t          = Tensor;
  using this_t            = tensor_slice<Tensor, Real, FixedDim, Dims...>;
  using parent_t          = base_tensor<this_t, Real, Dims...>;
  using parent_t::operator=;
  using parent_t::num_components;
  using parent_t::num_dimensions;

  //============================================================================
 private:
  Tensor* m_tensor;
  size_t  m_fixed_index;

  //============================================================================
 public:
  constexpr tensor_slice(Tensor* tensor, size_t fixed_index)
      : m_tensor{tensor}, m_fixed_index{fixed_index} {}

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr auto at(const Is... is) const -> decltype(auto) {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == num_dimensions()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(
          at_, unpack(extract<0, FixedDim - 1>(is...)), m_fixed_index,
          unpack(extract<FixedDim, num_dimensions() - 1>(is...)));
    };
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true,
            typename _tensor_t                                       = Tensor,
            std::enable_if_t<!std::is_const<_tensor_t>::value, bool> = true>
  constexpr auto at(const Is... is) -> decltype(auto) {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == num_dimensions()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(
          at_, unpack(extract<0, FixedDim - 1>(is...)), m_fixed_index,
          unpack(extract<FixedDim, num_dimensions() - 1>(is...)));
    };
  }
};

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct const_imag_complex_tensor
    : base_tensor<const_imag_complex_tensor<Tensor, Real, Dims...>, Real,
                  Dims...> {
  using this_t   = const_imag_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::num_dimensions;

  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit constexpr const_imag_complex_tensor(
      const base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr decltype(auto) operator()(const Indices... indices) const {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr decltype(auto) at(const Indices... indices) const {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).imag();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct imag_complex_tensor
    : base_tensor<imag_complex_tensor<Tensor, Real, Dims...>, Real, Dims...> {
  using this_t   = imag_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::num_dimensions;
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit constexpr imag_complex_tensor(
      base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).imag();
  }
  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).imag();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
auto imag(const base_tensor<Tensor, std::complex<Real>, Dims...>& tensor) {
  return const_imag_complex_tensor<Tensor, Real, Dims...>{tensor.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
auto imag(base_tensor<Tensor, std::complex<Real>, Dims...>& tensor) {
  return imag_complex_tensor<Tensor, Real, Dims...>{tensor.as_derived()};
}

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct const_real_complex_tensor
    : base_tensor<const_real_complex_tensor<Tensor, Real, Dims...>, Real,
                  Dims...> {
  using this_t   = const_real_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::num_dimensions;
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit const_real_complex_tensor(
      const base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).real();
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct real_complex_tensor
    : base_tensor<real_complex_tensor<Tensor, Real, Dims...>, Real, Dims...> {
  using this_t   = real_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::num_dimensions;
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit real_complex_tensor(
      base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).real();
  }
  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == num_dimensions());
    return m_internal_tensor(indices...).real();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
auto real(const base_tensor<Tensor, std::complex<Real>, Dims...>& t) {
  return const_real_complex_tensor<Tensor, Real, Dims...>{t.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
auto real(base_tensor<Tensor, std::complex<Real>, Dims...>& t) {
  return real_complex_tensor<Tensor, Real, Dims...>{t.as_derived()};
}

#if TATOOINE_GINAC_AVAILABLE
//==============================================================================
// symbolic
//==============================================================================
template <typename RealOut = double, typename Tensor, size_t... Dims,
          typename... Relations>
auto evtod(const base_tensor<Tensor, GiNaC::ex, Dims...>& t_in,
           Relations&&... relations) {
  tensor<RealOut, Dims...> t_out;

  t_out.for_indices([&](const auto... is) {
    t_out(is...) = symbolic::evtod<RealOut>(
        t_in(is...), std::forward<Relations>(relations)...);
  });

  return t_out;
}

//------------------------------------------------------------------------------
template <typename RealOut = double, typename Tensor, size_t... Dims>
auto diff(const base_tensor<Tensor, GiNaC::ex, Dims...>& t_in,
          const GiNaC::symbol& symbol, unsigned nth = 1) {
  return unary_operation(
      [&](const auto& component) { return component.diff(symbol, nth); }, t_in);
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
auto to_ginac_matrix(const base_tensor<Tensor, GiNaC::ex, M, N>& m_in) {
  GiNaC::matrix m_out(M, N);
  m_in.for_indices([&](const auto... is) { m_out(is...) = m_in(is...); });
  return m_out;
}

//------------------------------------------------------------------------------
template <size_t M, size_t N>
auto to_mat(const GiNaC::matrix& m_in) {
  assert(m_in.rows() == M);
  assert(m_in.cols() == N);
  mat<GiNaC::ex, M, N> m_out;
  m_out.for_indices([&](const auto... is) { m_out(is...) = m_in(is...); });
  return m_out;
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void eval(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).eval(); });
}
//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void evalf(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).evalf(); });
}
//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void evalm(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).evalm(); });
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void expand(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).expand(); });
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void normal(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...).normal(); });
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
auto inverse(const base_tensor<Tensor, GiNaC::ex, M, N>& m_in) {
  return to_mat<M, N>(to_ginac_matrix(m_in).inverse());
}
#endif

//==============================================================================
// I/O
//==============================================================================
/// printing vector
template <typename Tensor, typename Real, size_t N>
auto operator<<(std::ostream& out, const base_tensor<Tensor, Real, N>& v)
    -> auto& {
  out << "[ ";
  out << std::scientific;
  for (size_t i = 0; i < N; ++i) {
    if constexpr (!is_complex_v<Real>) {}
    out << v(i) << ' ';
  }
  out << "]";
  out << std::defaultfloat;
  return out;
}

template <typename Tensor, typename Real, size_t M, size_t N>
auto operator<<(std::ostream& out, const base_tensor<Tensor, Real, M, N>& m)
    -> auto& {
  out << std::scientific;
  for (size_t j = 0; j < M; ++j) {
    out << "[ ";
    for (size_t i = 0; i < N; ++i) {
      if constexpr (!is_complex_v<Real>) {
        if (m(j, i) >= 0) { out << ' '; }
      }
      out << m(j, i) << ' ';
    }
    out << "]\n";
  }
  out << std::defaultfloat;
  return out;
}

//==============================================================================
// type traits
//==============================================================================
template <typename Tensor, typename Real, size_t n>
struct num_components<base_tensor<Tensor, Real, n>>
    : std::integral_constant<size_t, n> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t n>
struct num_components<tensor<Real, n>> : std::integral_constant<size_t, n> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t n>
struct num_components<vec<Real, n>> : std::integral_constant<size_t, n> {};

//==============================================================================
template <typename Real, size_t N>
struct internal_data_type<vec<Real, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t M, size_t N>
struct internal_data_type<mat<Real, M, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t... Dims>
struct internal_data_type<tensor<Real, Dims...>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
struct internal_data_type<base_tensor<Tensor, Real, Dims...>> {
  using type = Real;
};

//==============================================================================
template <typename Tensor, typename Real, size_t N>
struct unpack<base_tensor<Tensor, Real, N>> {
  static constexpr size_t       n = N;
  base_tensor<Tensor, Real, N>& container;
  //----------------------------------------------------------------------------
  explicit constexpr unpack(base_tensor<Tensor, Real, N>& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& {
    return container(I);
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container(I);
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Tensor, typename Real, size_t N>
unpack(base_tensor<Tensor, Real, N>& c) -> unpack<base_tensor<Tensor, Real, N>>;

//==============================================================================
template <typename Tensor, typename Real, size_t N>
struct unpack<const base_tensor<Tensor, Real, N>> {
  static constexpr size_t             n = N;
  const base_tensor<Tensor, Real, N>& container;
  //----------------------------------------------------------------------------
  explicit constexpr unpack(const base_tensor<Tensor, Real, N>& c)
      : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container(I);
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Tensor, typename Real, size_t N>
unpack(const base_tensor<Tensor, Real, N>& c)
    -> unpack<const base_tensor<Tensor, Real, N>>;

//==============================================================================
template <typename Real, size_t N>
struct unpack<tensor<Real, N>> {
  static constexpr size_t n = N;
  tensor<Real, N>&        container;

  //----------------------------------------------------------------------------
  explicit constexpr unpack(tensor<Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& {
    return container[I];
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container[I];
  }
};
//==============================================================================
template <typename Real, size_t N>
unpack(tensor<Real, N>& c) -> unpack<tensor<Real, N>>;

//==============================================================================
template <typename Real, size_t N>
struct unpack<const tensor<Real, N>> {
  static constexpr size_t n = N;
  const tensor<Real, N>&  container;

  //----------------------------------------------------------------------------
  explicit constexpr unpack(const tensor<Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container[I];
  }
};
//==============================================================================
template <typename Real, size_t N>
unpack(const tensor<Real, N>& c) -> unpack<const tensor<Real, N>>;

template <typename CastReal, typename Real, size_t N>
auto cast_tensor_type_impl(const vec<Real, N>&) {
  return vec<CastReal, N>::zeros();
}
template <typename CastReal, typename Real, size_t M, size_t N>
auto cast_tensor_type_impl(const mat<Real, M, N>&) {
  return mat<CastReal, M, N>::zeros();
}
template <typename CastReal, typename Real, size_t... Dims>
auto cast_tensor_type_impl(const tensor<Real, Dims...>&) {
  return tensor<CastReal, Dims...>::zeros();
}

template <typename CastedReal, typename Tensor>
struct cast_tensor_real {
  using type =
      decltype(cast_tensor_type_impl<CastedReal>(std::declval<Tensor>()));
};

template <typename CastedReal, typename Tensor>
using cast_tensor_real_t = typename cast_tensor_real<CastedReal, Tensor>::type;

//==============================================================================
template <typename NewReal, typename Tensor, typename Real, size_t... Dims>
auto cast(const base_tensor<Tensor, Real, Dims...>& to_cast) {
  auto casted = tensor<NewReal, Dims...>::zeros;
  for (size_t i = 0; i < casted.num_components(); ++i) {
    casted[i] = static_cast<NewReal>(to_cast[i]);
  }
  return casted;
}
//------------------------------------------------------------------------------
template <typename NewReal, typename Real, size_t M, size_t N>
auto cast(const mat<Real, M, N>& to_cast) {
  auto casted = mat<NewReal, M, N>::zeros();
  for (size_t i = 0; i < M * N; ++i) {
    casted[i] = static_cast<NewReal>(to_cast[i]);
  }
  return casted;
}
//------------------------------------------------------------------------------
template <typename NewReal, typename Real, size_t N>
auto cast(const vec<Real, N>& to_cast) {
  auto casted = vec<NewReal, N>::zeros();
  for (size_t i = 0; i < N; ++i) {
    casted[i] = static_cast<NewReal>(to_cast[i]);
  }
  return casted;
}
//==============================================================================
template <typename T>
struct is_tensor : std::false_type {};
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
struct is_tensor<base_tensor<Tensor, Real, Dims...>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t... Dims>
struct is_tensor<tensor<Real, Dims...>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t N>
struct is_tensor<vec<Real, N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N>
struct is_tensor<mat<Real, M, N>> : std::true_type {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T>
static constexpr auto is_tensor_v = is_tensor<T>::value;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_tensor = std::enable_if_t<(is_tensor_v<Ts> && ...), bool>;
//==============================================================================
template <typename T>
struct is_vector : std::false_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t N>
struct is_vector<vec<Real, N>> : std::true_type {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T>
static constexpr auto is_vector_v = is_vector<T>::value;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_vector = std::enable_if_t<(is_vector_v<Ts> && ...), bool>;
//==============================================================================
template <typename T>
struct is_matrix : std::false_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N>
struct is_matrix<mat<Real, M, N>> : std::true_type {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T>
static constexpr auto is_matrix_v = is_matrix<T>::value;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_matrix = std::enable_if_t<(is_matrix_v<Ts> && ...), bool>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
