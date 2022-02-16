#ifndef TATOOINE_BASE_TENSOR_H
#define TATOOINE_BASE_TENSOR_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/crtp.h>
#include <tatooine/einstein_notation.h>
#include <tatooine/index_order.h>
#include <tatooine/multidim_size.h>
#include <tatooine/template_helper.h>
#include <tatooine/type_traits.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cassert>
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T, std::size_t FixedDim,
          std::size_t... Dims>
struct tensor_slice;
//------------------------------------------------------------------------------
template <typename Tensor, typename T, std::size_t... Dims>
struct base_tensor : crtp<Tensor> {
  using value_type = T;
  using tensor_type   = Tensor;
  using this_type     = base_tensor<Tensor, T, Dims...>;
  using parent_type   = crtp<Tensor>;
  using parent_type::as_derived;
  using multidim_size_t = static_multidim_size<x_fastest, Dims...>;
  static_assert(is_arithmetic<T> || is_complex<T>,
                "A tensor can only hold real or complex values.");
  static_assert(sizeof...(Dims) > 0,
                "base tensor needs to have more than one dimension defined.");
  //============================================================================
  static constexpr auto rank() { return sizeof...(Dims); }
  //------------------------------------------------------------------------------
  static constexpr auto num_components() {
    return multidim_size_t::num_components();
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimensions() {
    return std::array{Dims...};
  }
  //------------------------------------------------------------------------------
  static constexpr auto dimension(std::size_t const i) {
    return dimensions()[i];
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
  template <typename OtherTensor, typename OtherReal>
  explicit constexpr base_tensor(
      base_tensor<OtherTensor, OtherReal, Dims...> const& other) {
    assign_other_tensor(other);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherReal>
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
  template <typename F, typename OtherTensor, typename OtherReal>
  auto binary_operation(
      F&& f, base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> decltype(auto) {
    for_indices([this, &f, &other](auto const... is) {
      this->at(is...) = f(at(is...), other(is...));
    });
    return as_derived();
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  constexpr void assign_other_tensor(
      base_tensor<OtherTensor, OtherReal, Dims...> const& other) {
    for_indices(
        [this, &other](auto const... is) { this->at(is...) = other(is...); });
  }
  //----------------------------------------------------------------------------
  template <einstein_notation::index... Is>
  auto constexpr at(Is const... /*is*/) {
    static_assert(sizeof...(Is) == rank(),
                  "number of indices does not match rank");
    return einstein_notation::indexed_tensor<Tensor&, Is...>{as_derived()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  auto constexpr at(Is const... /*is*/) const {
    static_assert(sizeof...(Is) == rank(),
                  "number of indices does not match rank");
    return einstein_notation::indexed_tensor<Tensor const&, Is...>{
        as_derived()};
  }
  //----------------------------------------------------------------------------
  template <einstein_notation::index... Is>
  auto constexpr operator()(Is const... is) const {
    static_assert(sizeof...(Is) == rank(),
                  "Number of indices does not match number of dimensions.");
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  auto constexpr operator()(Is const... is) {
    static_assert(sizeof...(Is) == rank(),
                  "Number of indices does not match number of dimensions.");
    return at(is...);
  }
  //----------------------------------------------------------------------------
  auto constexpr at(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return as_derived().at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr at(integral auto const... is) -> decltype(auto) {
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
                  "Number of indices does not match number of dimensions.");
    return at(is...);
  }

  //----------------------------------------------------------------------------
  template <std::size_t FixedDim, std::size_t... Is>
  constexpr auto slice(std::size_t fixed_index, std::index_sequence<Is...>)
      -> decltype(auto) {
    if constexpr (rank() > 1) {
      static_assert(
          FixedDim < rank(),
          "Fixed dimensions must be in range of number of dimensions.");
      return tensor_slice<Tensor, T, FixedDim,
                          dimension(sliced_indices<rank(), FixedDim>()[Is])...>{
          &as_derived(), fixed_index};
    } else {
      return at(fixed_index);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t FixedDim>
  constexpr auto slice(std::size_t fixed_index) -> decltype(auto) {
    if constexpr (rank() > 1) {
      static_assert(
          FixedDim < rank(),
          "Fixed dimensions must be in range of number of dimensions.");
      return slice<FixedDim>(fixed_index,
                             std::make_index_sequence<rank() - 1>{});
    } else {
      return at(fixed_index);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t FixedDim, std::size_t... Is>
  constexpr auto slice(std::size_t fixed_index,
                       std::index_sequence<Is...>) const {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<Tensor const, T, FixedDim,
                        dimension(sliced_indices<rank(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t FixedDim>
  [[nodiscard]] constexpr auto slice(std::size_t fixed_index) const {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return slice<FixedDim>(fixed_index, std::make_index_sequence<rank() - 1>{});
  }

  //----------------------------------------------------------------------------
  static constexpr auto array_index(integral auto const... is) {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return static_multidim_size<x_fastest, Dims...>::plain_idx(is...);
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  auto operator+=(base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> auto& {
    for_indices([&](auto const... is) { at(is...) += other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator+=(convertible_to<T> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) += other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  auto operator-=(base_tensor<OtherTensor, OtherReal, Dims...> const& other)
      -> auto& {
    for_indices([&](auto const... is) { at(is...) -= other(is...); });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator-=(convertible_to<T> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) -= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator*=(convertible_to<T> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) *= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator/=(convertible_to<T> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) /= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto isnan() const {
    bool b = false;
    for_indices([&](auto const... is) {
      if (std::isnan(at(is...))) {
        b = true;
        return false;
      }
      return true;
    });
    return b;
  }
  //----------------------------------------------------------------------------
  auto isinf() const {
    bool b = false;
    for_indices([&](auto const... is) {
      if (std::isinf(at(is...))) {
        b = true;
        return false;
      }
      return true;
    });
    return b;
  }

 private:
  friend class boost::serialization::access;
};
//==============================================================================
template <typename T, typename Void = void>
struct is_tensor_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_tensor = is_tensor_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_tensor_impl<T, std::void_t<decltype(std::decay_t<T>::is_tensor())>>
    : std::integral_constant<bool, std::decay_t<T>::is_tensor()> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t... Dimensions>
concept fixed_size_tensor = is_tensor<T> && T::dimensions() ==
                              std::array{Dimensions...};
//------------------------------------------------------------------------------
template <typename T, typename Void = void>
struct is_vec_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_vec = is_vec_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_vec_impl<T, std::void_t<decltype(std::decay_t<T>::is_vec())>>
    : std::integral_constant<bool, std::decay_t<T>::is_vec()> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t N>
concept fixed_size_vec = is_vec<T> && T::dimension(0) == N;
//==============================================================================
template <typename T, typename Void = void>
struct is_mat_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_mat = is_mat_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_mat_impl<T, std::void_t<decltype(std::decay_t<T>::is_mat())>>
    : std::integral_constant<bool, std::decay_t<T>::is_mat()> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t M, std::size_t N>
concept fixed_size_mat = is_mat<T> &&
                         T::dimension(0) == M &&
                         T::dimension(1) == N;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t N>
concept fixed_num_cols_mat = is_mat<T> &&
                             T::dimension(1) == N;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t M>
concept fixed_num_rows_mat = is_mat<T> &&
                             T::dimension(0) == M;
//==============================================================================
template <typename T, typename Void = void>
struct is_quadratic_mat_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_quadratic_mat = is_quadratic_mat_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_quadratic_mat_impl<
    T, std::void_t<decltype(std::decay_t<T>::is_quadratic_mat())>>
    : std::integral_constant<bool, std::decay_t<T>::is_quadratic_mat()> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t N>
concept fixed_size_quadratic_mat = is_quadratic_mat<T> && T::dimension(0) == N;
//==============================================================================
template <arithmetic_or_complex T, std::size_t... Dims>
struct tensor;
template <arithmetic_or_complex T, std::size_t M, std::size_t N>
struct mat;
template <arithmetic_or_complex T, std::size_t N>
struct vec;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/tensor_utility.h>
//==============================================================================
#endif
