#ifndef TATOOINE_BASE_TENSOR_H
#define TATOOINE_BASE_TENSOR_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/crtp.h>
#include <tatooine/index_ordering.h>
#include <tatooine/multidim_size.h>
#include <tatooine/template_helper.h>

#include <cassert>
#include <type_traits>
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
  using multidim_size_t = static_multidim_size<x_fastest, Dims...>;

  //============================================================================
  static constexpr auto rank() { return sizeof...(Dims); }
  //------------------------------------------------------------------------------
  static constexpr auto num_components() {
    return multidim_size_t::num_components();
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimensions() {
    return std::array<size_t, rank()>{Dims...};
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimension(size_t const i) {
    return template_helper::getval<size_t>(i, Dims...);
  }
  //------------------------------------------------------------------------------
  static constexpr auto is_tensor() -> bool { return true; }
  static constexpr auto is_vec() -> bool { return rank() == 1; }
  static constexpr auto is_mat() -> bool { return rank() == 2; }
  static constexpr auto is_quadratic_mat() -> bool {
    return rank() == 2 && dimension(0) == dimension(1);
  }
  //------------------------------------------------------------------------------
  static constexpr auto indices() { return multidim_size_t::indices(); }
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
      base_tensor<OtherTensor, OtherReal, Dims...> const& other) {
    assign_other_tensor(other);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, real_or_complex_number OtherReal>
  constexpr auto operator=(
      base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> base_tensor& {
    assign_other_tensor(other);
    return *this;
  }
  //============================================================================
  template <typename F>
  auto unary_operation(F&& f) -> auto& {
    for_indices([this, &f](auto const... is) { at(is...) = f(at(is...)); });
    return as_derived();
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherTensor, real_or_complex_number OtherReal>
  auto binary_operation(
      F&& f, base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> decltype(auto) {
    for_indices([this, &f, &other](auto const... is) {
      this->at(is...) = f(at(is...), other(is...));
    });
    return as_derived();
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  constexpr void assign_other_tensor(
      base_tensor<OtherTensor, OtherReal, Dims...> const& other) {
    for_indices([this, &other](auto const... is) {
      this->at(is...) = other(is...);
    });
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
                  "number of indices does not match rank");
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
        Tensor const, T, FixedDim,
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
    return static_multidim_size<x_fastest, Dims...>::plain_idx(is...);
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  auto operator==(base_tensor<OtherTensor, OtherReal, Dims...> const& other) {
    bool equal = true;
    for_indices([&](auto const... is) {
      if (at(is...) != other(is...)) { equal = false; }
    });
    return equal;
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  auto operator+=(base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> auto& {
    for_indices([&](auto const... is) { at(is...) += other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator+=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) += other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, real_or_complex_number OtherReal>
  auto operator-=(base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> auto& {
    for_indices([&](auto const... is) { at(is...) -= other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator-=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) -= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator*=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) *= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator/=(real_or_complex_number auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) /= other; });
    return *this;
  }
};
//==============================================================================
template <typename T, typename Void = void>
struct is_tensor : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_tensor_v = is_tensor<T>::value;
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
template <typename T>
struct is_tensor<T, std::void_t<decltype(std::decay_t<T>::is_tensor())>>
    : std::integral_constant<bool, std::decay_t<T>::is_tensor()> {};
//------------------------------------------------------------------------------
template <typename T, typename Void = void>
struct is_vec : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_vec_v = is_vec<T>::value;
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
template <typename T>
struct is_vec<T, std::void_t<decltype(std::decay_t<T>::is_vec())>>
    : std::integral_constant<bool, std::decay_t<T>::is_vec()> {};
//==============================================================================
template <typename T, typename Void = void>
struct is_mat : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_mat_v = is_mat<T>::value;
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
template <typename T>
struct is_mat<T, std::void_t<decltype(std::decay_t<T>::is_mat())>>
    : std::integral_constant<bool, std::decay_t<T>::is_mat()> {};
//==============================================================================
template <typename T, typename Void = void>
struct is_quadratic_mat : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_quadratic_mat_v = is_quadratic_mat<T>::value;
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
template <typename T>
struct is_quadratic_mat<
    T, std::void_t<decltype(std::decay_t<T>::is_quadratic_mat())>>
    : std::integral_constant<bool, std::decay_t<T>::is_quadratic_mat()> {};
//==============================================================================
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_or_complex_number T, size_t... Dims>
struct inner_value_type<base_tensor<Tensor, T, Dims...>> {
  using type = T;
};
template <real_or_complex_number T, size_t... Dims>
struct tensor;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
#include <tatooine/vec.h>
#include <tatooine/mat.h>
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#include <tatooine/tensor_cast.h>
#include <tatooine/tensor_lapack_utility.h>
#include <tatooine/tensor_type_traits.h>
#include <tatooine/tensor_unpack.h>
#include <tatooine/tensor_operations.h>
//==============================================================================
#include <tatooine/abs_tensor.h>
#include <tatooine/tensor_io.h>
#include <tatooine/tensor_utility.h>
#include <tatooine/rank.h>
#include <tatooine/diag_tensor.h>
#include <tatooine/transposed_tensor.h>
#include <tatooine/tensor_slice.h>
#include <tatooine/complex_tensor_views.h>

//#include <tatooine/tensor_symbolic.h>
//==============================================================================
#endif
