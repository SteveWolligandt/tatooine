#ifndef TATOOINE_BASE_TENSOR_H
#define TATOOINE_BASE_TENSOR_H
//==============================================================================
#include <cassert>
#include "crtp.h"
//==============================================================================
namespace tatooine {
//==============================================================================
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
  static constexpr auto rank() { return sizeof...(Dims); }
  //------------------------------------------------------------------------------
  static constexpr auto num_components() {
    return resolution_t::num_elements();
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimensions() {
    return std::array<size_t, rank()>{Dims...};
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
    for_indices([this, &other](const auto... is) {
      at(is...) = other(is...);
    });
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) at(const Is... is) const {
    static_assert(sizeof...(Is) == rank(),
                  "number of indices does not match rank");
    return as_derived().at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) at(const Is... is) {
    static_assert(sizeof...(Is) == rank(),
                  "number of indices does not match rank");
    return as_derived().at(is...);
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) operator()(const Is... is) const {
    static_assert(sizeof...(Is) == rank(),
                  "number of indices does not match rank");
    return at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...> = true>
  constexpr decltype(auto) operator()(const Is... is) {
    static_assert(sizeof...(Is) == rank(),
                  "number of indices does not match rank");
    return at(is...);
  }

  //----------------------------------------------------------------------------
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<
        Tensor, Real, FixedDim,
        dimension(sliced_indices<rank(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim>
  constexpr auto slice(size_t fixed_index) {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return slice<FixedDim>(fixed_index,
                           std::make_index_sequence<rank() - 1>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) const {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<
        const Tensor, Real, FixedDim,
        dimension(sliced_indices<rank(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim>
  [[nodiscard]] constexpr auto slice(size_t fixed_index) const {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return slice<FixedDim>(fixed_index,
                           std::make_index_sequence<rank() - 1>{});
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...> = true>
  static constexpr auto array_index(const Is... is) {
    static_assert(sizeof...(Is) == rank(),
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
}  // namespace tatooine
//==============================================================================
#include <tatooine/abs_tensor.h>
#include <tatooine/tensor_io.h>
#include <tatooine/tensor_utility.h>
#include <tatooine/tensor_slice.h>
#include <tatooine/complex_tensor_views.h>
#include <tatooine/diag_tensor.h>

#include <tatooine/tensor.h>
#include <tatooine/tensor_symbolic.h>
//==============================================================================
#endif
