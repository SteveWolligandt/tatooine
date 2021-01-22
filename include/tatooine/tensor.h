#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
#include <tatooine/math.h>
#include <tatooine/multidim_array.h>
#include <tatooine/tags.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, size_t... Dims>
struct tensor : base_tensor<tensor<T, Dims...>, T, Dims...>,  // NOLINT
                static_multidim_array<T, x_fastest, tag::stack, Dims...> {
  static_assert(is_arithmetic<T> || is_complex<T>);
  //============================================================================
  using this_t          = tensor<T, Dims...>;
  using tensor_parent_t = base_tensor<this_t, T, Dims...>;
  using typename tensor_parent_t::value_type;
  using array_parent_t =
      static_multidim_array<T, x_fastest, tag::stack, Dims...>;

  using tensor_parent_t::dimension;
  using tensor_parent_t::num_components;
  using tensor_parent_t::rank;
  using tensor_parent_t::tensor_parent_t;

  using array_parent_t::at;
  using array_parent_t::operator();

  //============================================================================
 public:
  constexpr tensor()                        = default;
  constexpr tensor(tensor const&)           = default;
  constexpr tensor(tensor&& other) noexcept = default;
  constexpr auto operator=(tensor const&) -> tensor& = default;
  constexpr auto operator=(tensor&& other) noexcept -> tensor& = default;
  ~tensor()                                                    = default;
  //============================================================================
 public:
  template <typename... Ts, size_t _N = tensor_parent_t::rank(),
            size_t _Dim0 = tensor_parent_t::dimension(0)>
      requires(_N == 1) &&
      (_Dim0 == sizeof...(Ts)) explicit constexpr tensor(Ts const&... ts)
      : array_parent_t{ts...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<T>
#else
  template <typename = void, enable_if_arithmetic<T> = true>
#endif
      explicit constexpr tensor(tag::zeros_t zeros) : array_parent_t{zeros} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename = void>
  requires is_arithmetic<T>
#else
  template <typename = void, enable_if_arithmetic<T> = true>
#endif
      explicit constexpr tensor(tag::ones_t ones) : array_parent_t{ones} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename FillReal>
  requires is_arithmetic<T>
#else
  template <typename FillReal, enable_if_arithmetic<T> = true>
#endif
      explicit constexpr tensor(tag::fill<FillReal> f) : array_parent_t{f} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandomReal, typename Engine>
  requires is_arithmetic<T>
#else
  template <typename RandomReal, typename Engine,
            enable_if_arithmetic<T> = true>
#endif
      explicit constexpr tensor(random_uniform<RandomReal, Engine>&& rand)
      : array_parent_t{std::move(rand)} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic RandomReal, typename Engine>
#else
  template <typename RandomReal, typename Engine,
            enable_if_arithmetic<RandomReal> = true>
#endif
  explicit constexpr tensor(random_uniform<RandomReal, Engine>& rand)
      : array_parent_t{rand} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic RandomReal, typename Engine>
#else
  template <typename RandomReal, typename Engine,
            enable_if_arithmetic<RandomReal> = true>
#endif
  requires is_arithmetic<T> explicit constexpr tensor(
      random_normal<RandomReal, Engine>&& rand)
      : array_parent_t{std::move(rand)} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic RandomReal, typename Engine>
#else
  template <typename RandomReal, typename Engine,
            enable_if_arithmetic<RandomReal> = true>
#endif
  requires is_arithmetic<T> explicit constexpr tensor(
      random_normal<RandomReal, Engine>& rand)
      : array_parent_t{rand} {
  }
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
    if constexpr (is_transposed_tensor_v<OtherTensor>) {
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
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randu(T min = 0, T max = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_uniform{min, max, std::forward<RandEng>(eng)}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static constexpr auto randn(T mean = 0, T stddev = 1,
                              RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{random_normal<T>{eng, mean, stddev}};
  }
  //----------------------------------------------------------------------------
  template <typename OtherT>
  auto operator==(tensor<OtherT, Dims...> const& other) const {
    return this->data() == other.data();
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
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
