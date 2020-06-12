#ifndef TATOOINE_BASE_TENSOR_H
#define TATOOINE_BASE_TENSOR_H
//==============================================================================
#include <cassert>
#include "crtp.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, real_or_complex_number T, size_t FixedDim, size_t... Dims>
struct tensor_slice;
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T, size_t... Dims>
struct base_tensor : crtp<Tensor> {
  using value_type = T;
  using tensor_t   = Tensor;
  using this_t     = base_tensor<Tensor, T, Dims...>;
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
  template <typename OtherTensor, real_or_complex_number OtherReal>
  explicit constexpr base_tensor(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    assign_other_tensor(other);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, real_or_complex_number OtherReal>
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
  template <typename F, typename OtherTensor, real_or_complex_number OtherReal>
  auto binary_operation(
      F&& f, const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> decltype(auto) {
    for_indices([this, &f, &other](const auto... is) {
      at(is...) = f(at(is...), other(is...));
    });
    return as_derived();
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  constexpr void assign_other_tensor(
      const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    for_indices([this, &other](const auto... is) { at(is...) = other(is...); });
  }
  //----------------------------------------------------------------------------
  constexpr decltype(auto) at(integral auto const... is) const {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return as_derived().at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr decltype(auto) at(integral auto const... is) {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return as_derived().at(is...);
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) operator()(integral auto const... is) const {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr decltype(auto) operator()(integral auto const... is) {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return at(is...);
  }

  //----------------------------------------------------------------------------
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<
        Tensor, T, FixedDim,
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
        const Tensor, T, FixedDim,
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
  static constexpr auto array_index(integral auto const... is) {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return static_multidim_resolution<x_fastest, Dims...>::plain_idx(is...);
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  auto operator+=(const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> auto& {
    for_indices([&](const auto... is) { at(is...) += other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator+=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) += other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  auto operator-=(const base_tensor<OtherTensor, OtherReal, Dims...>& other)
      -> auto& {
    for_indices([&](const auto... is) { at(is...) -= other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator-=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) -= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator*=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](const auto... is) { at(is...) *= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator/=(real_or_complex_number auto const& other) -> auto& {
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
