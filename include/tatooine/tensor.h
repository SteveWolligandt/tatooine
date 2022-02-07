#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
#include <tatooine/is_transposed_tensor.h>
#include <tatooine/math.h>
#include <tatooine/multidim_array.h>
#include <tatooine/tags.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex T, size_t... Dims>
struct tensor : base_tensor<tensor<T, Dims...>, T, Dims...>,
                static_multidim_array<T, x_fastest, tag::stack, Dims...> {
  //============================================================================
  using this_t          = tensor<T, Dims...>;
  using tensor_parent_t = base_tensor<this_t, T, Dims...>;
  using value_type      = typename tensor_parent_t::value_type;
  using array_parent_t =
      static_multidim_array<T, x_fastest, tag::stack, Dims...>;

  using tensor_parent_t::dimension;
  using tensor_parent_t::num_components;
  using tensor_parent_t::rank;
  using tensor_parent_t::tensor_parent_t;
  //============================================================================
 public:
  constexpr tensor()                        = default;
  constexpr tensor(tensor const&)           = default;
  constexpr tensor(tensor&& other) noexcept = default;
  constexpr auto operator=(tensor const&) -> tensor& = default;
  constexpr auto operator=(tensor&& other) noexcept -> tensor& = default;
  ~tensor()                                                    = default;
  //============================================================================
  template <typename... Is>
  auto constexpr at(Is const... is) -> decltype(auto) {
    if constexpr (einstein_notation::is_index<Is...>) {
      return tensor_parent_t::at(is...);
    } else {
      return array_parent_t::at(is...);
    }
  }
  template <typename... Is>
  auto constexpr at(Is const... is) const -> decltype(auto) {
    if constexpr (einstein_notation::is_index<Is...>) {
      return tensor_parent_t::at(is...);
    } else {
      return array_parent_t::at(is...);
    }
  }
  //----------------------------------------------------------------------------
  template <typename... Is>
  auto constexpr operator()(Is const... is) -> decltype(auto) {
    return at(is...);
  }
  template <typename... Is>
  auto constexpr operator()(Is const... is) const -> decltype(auto) {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <convertible_to<T>... Ts>
  requires(tensor_parent_t::rank() == 1) &&
      (tensor_parent_t::dimension(0) ==
       sizeof...(Ts)) explicit constexpr tensor(Ts const&... ts)
      : array_parent_t{ts...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  explicit constexpr tensor(tag::zeros_t zeros) requires is_arithmetic<T>
      : array_parent_t{zeros} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  explicit constexpr tensor(tag::ones_t ones) requires is_arithmetic<T>
      : array_parent_t{ones} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename FillReal>
  requires is_arithmetic<T>
  explicit constexpr tensor(tag::fill<FillReal> f) : array_parent_t{f} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  requires is_arithmetic<T>
  explicit constexpr tensor(random::uniform<RandomReal, Engine>&& rand)
      : array_parent_t{std::move(rand)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic RandomReal, typename Engine>
  explicit constexpr tensor(random::uniform<RandomReal, Engine>& rand)
      : array_parent_t{rand} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic RandomReal, typename Engine>
  requires is_arithmetic<T>
  explicit constexpr tensor(random::normal<RandomReal, Engine>&& rand)
      : array_parent_t{std::move(rand)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic RandomReal, typename Engine>
  requires is_arithmetic<T>
  explicit constexpr tensor(random::normal<RandomReal, Engine>& rand)
      : array_parent_t{rand} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherT>
  explicit constexpr tensor(
      base_tensor<OtherTensor, OtherT, Dims...> const& other) {
    this->assign_other_tensor(other);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename OtherT>
  constexpr auto operator=(
      base_tensor<OtherTensor, OtherT, Dims...> const& other) -> tensor& {
    // Check if the same matrix gets assigned as its transposed version. If yes
    // just swap components.
    if constexpr (is_transposed_tensor<OtherTensor>) {
      if (this == &other.as_derived().internal_tensor()) {
        for (size_t col = 0; col < dimension(1) - 1; ++col) {
          for (size_t row = col + 1; row < dimension(0); ++row) {
            std::swap(at(row, col), at(col, row));
          }
        }
        return *this;
      }
    }
    this->assign_other_tensor(other);
    return *this;
  }
  //----------------------------------------------------------------------------
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
  template <typename OtherT>
  auto operator<(tensor<OtherT, Dims...> const& other) const {
    return this->data() < other.data();
  }
  //============================================================================
  template <typename F>
  auto unary_operation(F&& f) -> auto& {
    array_parent_t::unary_operation(std::forward<F>(f));
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherTensor, typename OtherT>
  auto binary_operation(F&&                                              f,
                        base_tensor<OtherTensor, OtherT, Dims...> const& other)
      -> decltype(auto) {
    tensor_parent_t::binary_operation(std::forward<F>(f), other);
    return *this;
  }
};
template <std::size_t... Dimensions>
using Tensor = tensor<real_t, Dimensions...>;

using tensor222    = Tensor<2, 2, 2>;
using tensor222    = Tensor<2, 2, 2>;
using tensor333    = Tensor<3, 3, 3>;
using tensor444    = Tensor<4, 4, 4>;
using tensor2222   = Tensor<2, 2, 2, 2>;
using tensor2222   = Tensor<2, 2, 2, 2>;
using tensor3333   = Tensor<3, 3, 3, 3>;
using tensor4444   = Tensor<4, 4, 4, 4>;
using tensor22222  = Tensor<2, 2, 2, 2, 2>;
using tensor22222  = Tensor<2, 2, 2, 2, 2>;
using tensor33333  = Tensor<3, 3, 3, 3, 3>;
using tensor44444  = Tensor<4, 4, 4, 4, 4>;
using tensor222222 = Tensor<2, 2, 2, 2, 2, 2>;
using tensor222222 = Tensor<2, 2, 2, 2, 2, 2>;
using tensor333333 = Tensor<3, 3, 3, 3, 3, 3>;
using tensor444444 = Tensor<4, 4, 4, 4, 4, 4>;
//==============================================================================
namespace reflection {
template <typename T, size_t... Dims>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (tensor<T, Dims...>), TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/dynamic_tensor.h>
//==============================================================================
#include <tatooine/mat.h>
#include <tatooine/vec.h>
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#include <tatooine/tensor_cast.h>
#include <tatooine/tensor_operations.h>
#include <tatooine/tensor_type_traits.h>
#include <tatooine/tensor_unpack.h>
//==============================================================================
#include <tatooine/complex_tensor_views.h>
#include <tatooine/diag_tensor.h>
#include <tatooine/rank.h>
#include <tatooine/tensor_io.h>
#include <tatooine/tensor_slice.h>
//==============================================================================
#endif
