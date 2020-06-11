#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H
//==============================================================================
#include <array>

#include "tags.h"
#include "multidim_array.h"
#include <tatooine/base_tensor.h>
#include "utility.h"
#include "math.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t... Dims>
struct tensor : base_tensor<tensor<Real, Dims...>, Real, Dims...>,  // NOLINT
                static_multidim_array<Real, x_fastest, tag::stack, Dims...> {
  //============================================================================
  using this_t          = tensor<Real, Dims...>;
  using tensor_parent_t = base_tensor<this_t, Real, Dims...>;
  using array_parent_t = static_multidim_array<Real, x_fastest, tag::stack, Dims...>;
  using tensor_parent_t::tensor_parent_t;
  using tensor_parent_t::operator=;
  using array_parent_t::at;
  using tensor_parent_t::dimension;
  using tensor_parent_t::num_components;
  using tensor_parent_t::rank;
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
  template <typename... Ts, size_t _N = rank(),
            size_t _Dim0                    = tensor_parent_t::dimension(0),
            std::enable_if_t<_N == 1, bool> = true,
            std::enable_if_t<_Dim0 == sizeof...(Ts), bool> = true>
  explicit constexpr tensor(const Ts&... ts) : array_parent_t{ts...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _Real = Real, enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(tag::zeros_t zeros) : array_parent_t{zeros} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _Real = Real, enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(tag::ones_t ones) : array_parent_t{ones} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename FillReal, typename _Real = Real,
            enable_if_arithmetic<_Real> = true>
  explicit constexpr tensor(tag::fill<FillReal> f) : array_parent_t{f} {}
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
  static constexpr auto zeros() { return this_t{tag::fill<Real>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_t{tag::fill<Real>{1}}; }
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
}  // namespace tatooine
//==============================================================================
#include <tatooine/vec.h>
#include <tatooine/mat.h>
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#include <tatooine/tensor_cast.h>
#include <tatooine/tensor_lapack_utility.h>
#include <tatooine/tensor_operations.h>
#include <tatooine/tensor_type_traits.h>
#include <tatooine/tensor_unpack.h>
#include <tatooine/transposed_tensor.h>
//==============================================================================
#endif
