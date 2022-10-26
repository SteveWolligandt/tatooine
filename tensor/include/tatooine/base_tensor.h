#ifndef TATOOINE_BASE_TENSOR_H
#define TATOOINE_BASE_TENSOR_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/crtp.h>
#include <tatooine/dynamic_multidim_size.h>
#include <tatooine/einstein_notation.h>
#include <tatooine/index_order.h>
#include <tatooine/template_helper.h>
#include <tatooine/tensor_operations/same_dimensions.h>
#include <tatooine/type_traits.h>

#include <cassert>
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, arithmetic_or_complex ValueType,
          std::size_t FixedDim, std::size_t... Dims>
struct tensor_slice;
//------------------------------------------------------------------------------
template <typename Tensor, arithmetic_or_complex ValueType, std::size_t... Dims>
struct base_tensor : crtp<Tensor> {
  using value_type  = ValueType;
  using tensor_type = Tensor;
  using this_type   = base_tensor<Tensor, ValueType, Dims...>;
  using parent_type = crtp<Tensor>;
  using parent_type::as_derived;
  using multidim_size_t = static_multidim_size<x_fastest, Dims...>;
  template <einstein_notation::index... Is>
  using const_indexed_type =
      einstein_notation::indexed_static_tensor<Tensor const&, Is...>;
  template <einstein_notation::index... Is>
  using indexed_type = einstein_notation::indexed_static_tensor<Tensor&, Is...>;

  static_assert(is_arithmetic<ValueType> || is_complex<ValueType>,
                "A tensor can only hold real or complex values.");
  static_assert(sizeof...(Dims) > 0,
                "base tensor needs to have more than one dimension defined.");
  //============================================================================
  static auto constexpr rank() { return sizeof...(Dims); }
  //------------------------------------------------------------------------------
  static auto constexpr num_components() {
    return multidim_size_t::num_components();
  }
  //------------------------------------------------------------------------------
  static auto constexpr dimensions() { return std::array{Dims...}; }
  //------------------------------------------------------------------------------
  static auto constexpr dimension(std::size_t const i) {
    return dimensions()[i];
  }
  //------------------------------------------------------------------------------
  static auto constexpr is_static() -> bool { return true; }
  static auto constexpr is_tensor() -> bool { return true; }
  //------------------------------------------------------------------------------
  static auto constexpr indices() { return multidim_size_t::indices(); }
  //------------------------------------------------------------------------------
  static auto constexpr for_indices(invocable<decltype(Dims)...> auto&& f) {
    for_loop(std::forward<decltype(f)>(f), Dims...);
  }
  //============================================================================
  constexpr base_tensor() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <static_tensor Other>
  requires (same_dimensions<this_type, Other>()) &&
           (convertible_to<tensor_value_type<Other>,
                      value_type>)
  explicit constexpr base_tensor(Other&& other) {
    assign(std::forward<Other>(other));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <static_tensor Other>
  requires(same_dimensions<this_type, Other>()) &&
      (convertible_to<tensor_value_type<Other>, value_type>)auto constexpr
      operator=(Other&& other) -> base_tensor& {
    assign(std::forward<Other>(other));
    return *this;
  }
  //============================================================================
  template <static_tensor Other>
  requires (same_dimensions<this_type, Other>()) &&
           (convertible_to<tensor_value_type<Other>,
                      value_type>)
  auto constexpr assign(Other&& other) -> void {
    for_indices([this, &other](auto const... is) {
      this->at(is...) = static_cast<value_type>(other(is...));
    });
  }
  //----------------------------------------------------------------------------
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) == rank()) auto constexpr at(Is const... /*is*/) {
    return indexed_type<Is...>{as_derived()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) ==
           rank()) auto constexpr at(Is const... /*is*/) const {
    return const_indexed_type<Is...>{as_derived()};
  }
  //----------------------------------------------------------------------------
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) == rank()) auto constexpr operator()(
      Is const... is) const {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) == rank()) auto constexpr operator()(Is const... is) {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  auto constexpr at(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) == rank()) {
    return as_derived().at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr at(integral auto const... is)
      -> decltype(auto) requires(sizeof...(is) == rank()) {
    return as_derived().at(is...);
  }  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto constexpr at(integral_range auto const& is,
                    std::index_sequence<Seq...> /*seq*/) -> decltype(auto) {
    return at(is[Seq]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t... Seq>
  auto constexpr at(integral_range auto const& is,
                    std::index_sequence<Seq...> /*seq*/) const
      -> decltype(auto) {
    return at(is[Seq]...);
  }
  //----------------------------------------------------------------------------
 public:
  auto constexpr at(integral_range auto const& is) -> decltype(auto) {
    assert(is.size() == rank());
    return at(is, std::make_index_sequence<rank()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr at(integral_range auto const& is) const -> decltype(auto) {
    assert(is.size() == rank());
    return at(is, std::make_index_sequence<rank()>{});
  }
  //----------------------------------------------------------------------------
  auto constexpr operator()(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) == rank()) {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr operator()(integral auto const... is)
      -> decltype(auto) requires(sizeof...(is) == rank()) {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <std::size_t FixedDim, std::size_t... Is>
  auto constexpr slice(std::size_t fixed_index, std::index_sequence<Is...>)
      -> decltype(auto) {
    if constexpr (rank() > 1) {
      static_assert(
          FixedDim < rank(),
          "Fixed dimensions must be in range of number of dimensions.");
      return tensor_slice<Tensor, ValueType, FixedDim,
                          dimension(sliced_indices<rank(), FixedDim>()[Is])...>{
          &as_derived(), fixed_index};
    } else {
      return at(fixed_index);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t FixedDim>
  auto constexpr slice(std::size_t fixed_index) -> decltype(auto) {
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
  auto constexpr slice(std::size_t fixed_index,
                       std::index_sequence<Is...>) const {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return tensor_slice<Tensor const, ValueType, FixedDim,
                        dimension(sliced_indices<rank(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t FixedDim>
  [[nodiscard]] auto constexpr slice(std::size_t fixed_index) const {
    static_assert(FixedDim < rank(),
                  "fixed dimensions must be in range of number of dimensions");
    return slice<FixedDim>(fixed_index, std::make_index_sequence<rank() - 1>{});
  }

  //----------------------------------------------------------------------------
  static auto constexpr array_index(integral auto const... is) {
    static_assert(sizeof...(is) == rank(),
                  "number of indices does not match number of dimensions");
    return static_multidim_size<x_fastest, Dims...>::plain_idx(is...);
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherValueType>
  auto operator+=(
      base_tensor<OtherTensor, OtherValueType, Dims...> const& other) -> auto& {
    for_indices([&](auto const... is) {
      at(is...) += static_cast<ValueType>(other(is...));
    });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator+=(convertible_to<ValueType> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) += other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherValueType>
  auto operator-=(
      base_tensor<OtherTensor, OtherValueType, Dims...> const& other) -> auto& {
    for_indices([&](auto const... is) {
      at(is...) -= static_cast<ValueType>(other(is...));
    });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator-=(convertible_to<ValueType> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) -= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator*=(convertible_to<ValueType> auto const& other) -> auto& {
    for_indices([&](auto const... is) { at(is...) *= other; });
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator/=(convertible_to<ValueType> auto const& other) -> auto& {
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
};
//==============================================================================
template <arithmetic_or_complex ValueType, std::size_t... Dims>
struct tensor;
template <arithmetic_or_complex ValueType, std::size_t M, std::size_t N>
struct mat;
template <arithmetic_or_complex ValueType, std::size_t N>
struct vec;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/tensor_utility.h>
//==============================================================================
#endif
