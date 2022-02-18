#ifndef TATOOINE_TENSOR_CONCEPTS_H
#define TATOOINE_TENSOR_CONCEPTS_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
concept general_tensor = requires(T t) {
  { t.rank() } -> integral;
  { t.dimensions() } -> integral_range;
  { t.dimension(std::declval<std::size_t>()) } -> integral;
}
&&std::decay_t<T>::is_tensor() && requires {
  typename std::decay_t<T>::value_type;
};
//==============================================================================
template <typename T>
concept dynamic_tensor =
  general_tensor<T> &&
  std::decay_t<T>::is_dynamic() ;
//==============================================================================
template <typename T>
concept static_tensor =
  general_tensor<T> &&
  std::decay_t<T>::is_static();
//==============================================================================
template <typename T>
concept static_vec =
  static_tensor<T> &&
  std::decay_t<T>::rank() == 1;
//==============================================================================
template <typename T>
concept static_mat =
  static_tensor<T> &&
  std::decay_t<T>::rank() == 2;
//==============================================================================
template <typename T, std::size_t... Dimensions>
concept fixed_size_tensor =
  static_tensor<T> &&
  std::decay_t<T>::dimensions() == std::array{Dimensions...};
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
concept fixed_size_vec =
  static_vec<T> &&
  std::decay_t<T>::dimension(0) == N;
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
concept fixed_size_real_vec =
  fixed_size_vec<T, N> &&
  arithmetic<typename T::value_type>;
//==============================================================================
template <typename T, std::size_t M, std::size_t N>
concept fixed_size_mat =
  static_mat<T> &&
  std::decay_t<T>::dimension(0) == M &&
  std::decay_t<T>::dimension(1) == N;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t N>
concept fixed_num_cols_mat =
  static_mat<T> &&
  std::decay_t<T>::dimension(1) == N;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t M>
concept fixed_num_rows_mat =
  static_mat<T> &&
  std::decay_t<T>::dimension(0) == M;
//==============================================================================
template <typename T>
concept static_quadratic_tensor =
  static_tensor<T> &&
  std::decay_t<T>::is_square();
//==============================================================================
template <typename T>
concept static_quadratic_mat =
  static_mat<T> &&
  std::decay_t<T>::is_square();
//==============================================================================
template <typename T, std::size_t N>
concept fixed_size_quadratic_tensor =
  static_quadratic_tensor<T> &&
  std::decay_t<T>::dimension(0) == N;
//==============================================================================
template <typename T, std::size_t N>
concept fixed_size_quadratic_mat =
  static_quadratic_mat<T> &&
  std::decay_t<T>::dimension(0) == N;
//==============================================================================
template <typename T>
concept transposed_tensor =
  general_tensor<T> &&
  std::decay_t<T>::is_transposed();
//==============================================================================
//template <typename T>
//concept transposed_static_tensor =
//  transposed_tensor<T> &&
//  static_tensor<T>;
//==============================================================================
//template <typename T>
//concept transposed_dynamic_tensor =
//  transposed_tensor<T> &&
//  dynamic_tensor<T>;
//==============================================================================
template <typename T>
concept diag_tensor =
  general_tensor<T> &&
  std::decay_t<T>::is_diag();
//==============================================================================
//template <typename T>
//concept diag_static_tensor =
//  diag_tensor<T> &&
//  static_tensor<T>;
//==============================================================================
//template <typename T>
//concept diag_dynamic_tensor =
//  diag_tensor<T> &&
//  dynamic_tensor<T>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
