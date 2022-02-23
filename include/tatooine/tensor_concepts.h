#ifndef TATOOINE_TENSOR_CONCEPTS_H
#define TATOOINE_TENSOR_CONCEPTS_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
concept general_tensor = std::decay_t<T>::is_tensor() && requires(T t) {
  { t.rank() } -> integral;
  { t.dimensions() } -> integral_range;
  { t.dimension(std::declval<std::size_t>()) } -> std::integral;
  typename std::decay_t<T>::value_type;
};
//==============================================================================
template <typename T>
concept dynamic_tensor = general_tensor<T> && std::decay_t<T>::is_dynamic();
//==============================================================================
template <typename T>
concept static_tensor = general_tensor<T> && std::decay_t<T>::is_static();
//==============================================================================
template <typename T>
concept static_vec = static_tensor<T> && std::decay_t<T>::rank()
== 1;
//==============================================================================
template <typename T>
concept static_mat = static_tensor<T> && std::decay_t<T>::rank()
== 2;
//==============================================================================
template <typename T, std::size_t... Dimensions>
concept fixed_size_tensor = static_tensor<T> && std::decay_t<T>::dimensions()
== std::array{Dimensions...};
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
concept fixed_size_vec = static_vec<T> && std::decay_t<T>::dimension(0)
== N;
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
concept fixed_size_real_vec =
    fixed_size_vec<T, N> && arithmetic<typename T::value_type>;
//==============================================================================
template <typename T, std::size_t M, std::size_t N>
concept fixed_size_mat = static_mat<T> && std::decay_t<T>::dimension(0)
== M&& std::decay_t<T>::dimension(1) == N;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t N>
concept fixed_num_cols_mat = static_mat<T> && std::decay_t<T>::dimension(1)
== N;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t M>
concept fixed_num_rows_mat = static_mat<T> && std::decay_t<T>::dimension(0)
== M;
//==============================================================================
template <static_tensor Tensor,
          std::size_t   I = std::decay_t<Tensor>::rank() - 1>
requires requires {std::decay_t<Tensor>::rank() > 2;}
struct is_square_impl {
  static auto constexpr value =
    std::decay_t<Tensor>::dimension(0) == std::decay_t<Tensor>::dimension(I) &&
    is_square_impl<Tensor, I - 1>::value;
};
//------------------------------------------------------------------------------
template <static_tensor Tensor>
requires requires {std::decay_t<Tensor>::rank() > 2;}
struct is_square_impl<Tensor, 1> {
  static auto constexpr value =
      std::decay_t<Tensor>::dimension(0) == std::decay_t<Tensor>::dimension(1);
};
//------------------------------------------------------------------------------
template <static_tensor Tensor>
requires requires {std::decay_t<Tensor>::rank() > 2;}
struct is_square_impl<Tensor, 0> {
  static auto constexpr value = true;
};
//------------------------------------------------------------------------------
template <static_tensor Tensor>
static auto constexpr is_square = is_square_impl<Tensor>::value;
//==============================================================================
template <typename T>
concept static_quadratic_tensor = static_tensor<T> && is_square<T>;
//==============================================================================
template <typename T>
concept static_quadratic_mat = static_mat<T> && is_square<T>;
//==============================================================================
template <typename T, std::size_t N>
concept fixed_size_quadratic_tensor = static_quadratic_tensor<T> &&
    std::decay_t<T>::dimension(0)
== N;
//==============================================================================
template <typename T, std::size_t N>
concept fixed_size_quadratic_mat = static_quadratic_mat<T> &&
    std::decay_t<T>::dimension(0)
== N;
//==============================================================================
template <typename T>
concept transposed_tensor = general_tensor<T> &&
    std::decay_t<T>::is_transposed();
//==============================================================================
// template <typename T>
// concept transposed_static_tensor =
//  transposed_tensor<T> &&
//  static_tensor<T>;
//==============================================================================
// template <typename T>
// concept transposed_dynamic_tensor =
//  transposed_tensor<T> &&
//  dynamic_tensor<T>;
//==============================================================================
template <typename T>
concept diag_tensor = general_tensor<T> && std::decay_t<T>::is_diag();
//==============================================================================
// template <typename T>
// concept diag_static_tensor =
//  diag_tensor<T> &&
//  static_tensor<T>;
//==============================================================================
// template <typename T>
// concept diag_dynamic_tensor =
//  diag_tensor<T> &&
//  dynamic_tensor<T>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
