#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
#include <tatooine/math.h>
#include <tatooine/multidim_array.h>
#include <tatooine/tags.h>
#include <tatooine/tensor_operations/same_dimensions.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex ValueType, std::size_t... Dims>
struct tensor
    : static_multidim_array<ValueType, x_fastest, tag::stack, Dims...>,
      base_tensor<tensor<ValueType, Dims...>, ValueType, Dims...> {
  //============================================================================
  using this_type          = tensor<ValueType, Dims...>;
  using tensor_parent_type = base_tensor<this_type, ValueType, Dims...>;
  using value_type         = typename tensor_parent_type::value_type;
  using array_parent_type =
      static_multidim_array<ValueType, x_fastest, tag::stack, Dims...>;

  using tensor_parent_type::dimension;
  using tensor_parent_type::num_components;
  using tensor_parent_type::rank;
  using tensor_parent_type::tensor_parent_type;
  //============================================================================
 public:
  constexpr tensor()                        = default;
  constexpr tensor(tensor const&)           = default;
  constexpr tensor(tensor&& other) noexcept = default;
  constexpr auto operator=(tensor const&) -> tensor& = default;
  constexpr auto operator=(tensor&& other) noexcept -> tensor& = default;
  ~tensor()                                                    = default;
  //============================================================================
  auto constexpr at(integral auto const... is) -> decltype(auto) {
    return array_parent_type::at(is...);
  }
  auto constexpr at(integral auto const... is) const -> decltype(auto) {
    return array_parent_type::at(is...);
  }
  auto constexpr at(einstein_notation::index auto const... is)
      -> decltype(auto) {
    return tensor_parent_type::at(is...);
  }
  auto constexpr at(einstein_notation::index auto const... is) const
      -> decltype(auto) {
    return tensor_parent_type::at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is>
  requires((einstein_notation::index<Is> && ...) ||
           (integral<Is> && ...)) auto constexpr
  operator()(Is const... is) -> decltype(auto) {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <typename... Is>
  requires((einstein_notation::index<Is> && ...) ||
           (integral<Is> && ...)) auto constexpr
  operator()(Is const... is) const -> decltype(auto) {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  template <convertible_to<ValueType>... Ts>
  requires(tensor_parent_type::num_components() ==
           sizeof...(Ts)) explicit constexpr tensor(Ts const&... ts)
      : array_parent_type{ts...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  explicit constexpr tensor(tag::zeros_t zeros) requires
      is_arithmetic<ValueType> : array_parent_type{zeros} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  explicit constexpr tensor(tag::ones_t ones) requires is_arithmetic<ValueType>
      : array_parent_type{ones} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename FillReal>
  requires is_arithmetic<ValueType>
  explicit constexpr tensor(tag::fill<FillReal> f) : array_parent_type{f} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine>
  requires is_arithmetic<ValueType>
  explicit constexpr tensor(random::uniform<RandomReal, Engine>&& rand)
      : array_parent_type{std::move(rand)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic RandomReal, typename Engine>
  explicit constexpr tensor(random::uniform<RandomReal, Engine>& rand)
      : array_parent_type{rand} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic RandomReal, typename Engine>
  requires is_arithmetic<ValueType>
  explicit constexpr tensor(random::normal<RandomReal, Engine>&& rand)
      : array_parent_type{std::move(rand)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic RandomReal, typename Engine>
  requires is_arithmetic<ValueType>
  explicit constexpr tensor(random::normal<RandomReal, Engine>& rand)
      : array_parent_type{rand} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <static_tensor OtherTensor>
  requires(same_dimensions<this_type, OtherTensor>()) explicit constexpr tensor(
      OtherTensor&& other)
      : tensor_parent_type{std::forward<OtherTensor>(other)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <static_tensor OtherTensor>
  requires(same_dimensions<this_type, OtherTensor>()) constexpr auto operator=(
      OtherTensor&& other) -> tensor& {
    // Check if the same matrix gets assigned as its transposed version. If yes
    // just swap components.
    if constexpr (transposed_tensor<OtherTensor>) {
      if (this == &other.as_derived().internal_tensor()) {
        for (std::size_t col = 0; col < dimension(1) - 1; ++col) {
          for (std::size_t row = col + 1; row < dimension(0); ++row) {
            std::swap(at(row, col), at(col, row));
          }
        }
        return *this;
      }
    }
    this->assign(std::forward<OtherTensor>(other));
    return *this;
  }
  //----------------------------------------------------------------------------
  static constexpr auto zeros() { return this_type{tag::fill<ValueType>{0}}; }
  //----------------------------------------------------------------------------
  static constexpr auto ones() { return this_type{tag::fill<ValueType>{1}}; }
  //----------------------------------------------------------------------------
  static constexpr auto fill(ValueType const& t) {
    return this_type{tag::fill<ValueType>{t}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randu(ValueType min = 0, ValueType max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{random::uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(ValueType mean = 0, ValueType stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{random::normal<ValueType>{eng, mean, stddev}};
  }
  //----------------------------------------------------------------------------
  template <typename OtherT>
  auto operator<(tensor<OtherT, Dims...> const& other) const {
    return this->data() < other.data();
  }
  //============================================================================
  template <typename F>
  auto unary_operation(F&& f) -> auto& {
    array_parent_type::unary_operation(std::forward<F>(f));
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename F, typename OtherTensor, typename OtherT>
  auto binary_operation(F&&                                              f,
                        base_tensor<OtherTensor, OtherT, Dims...> const& other)
      -> decltype(auto) {
    tensor_parent_type::binary_operation(std::forward<F>(f), other);
    return *this;
  }
};
template <std::size_t... Dimensions>
using Tensor = tensor<real_number, Dimensions...>;

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
template <typename ValueType, std::size_t... Dims>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (tensor<ValueType, Dims...>),
    TATOOINE_REFLECTION_INSERT_METHOD(data, data()))
}  // namespace reflection
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/dynamic_tensor.h>
#include <tatooine/tensor_typedefs.h>
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
