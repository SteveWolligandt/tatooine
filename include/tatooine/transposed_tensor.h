#ifndef TATOOINE_TRANSPOSED_TENSOR_H
#define TATOOINE_TRANSPOSED_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, size_t M, size_t N>
struct const_transposed_tensor
    : base_tensor<const_transposed_tensor<Tensor, M, N>,
                  typename Tensor::value_type, M, N> {
  static_assert(Tensor::dimension(0) == N);
  static_assert(Tensor::dimension(1) == M);
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit const_transposed_tensor(
      const base_tensor<Tensor, typename Tensor::value_type, N, M>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(const size_t r, const size_t c) const -> const
      auto& {
    return m_internal_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(const size_t r, const size_t c) const -> const auto& {
    return m_internal_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, size_t M, size_t N>
struct transposed_tensor : base_tensor<transposed_tensor<Tensor, M, N>,
                                       typename Tensor::value_type, M, N> {
  static_assert(Tensor::dimension(0) == N);
  static_assert(Tensor::dimension(1) == M);
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  constexpr explicit transposed_tensor(
      base_tensor<Tensor, typename Tensor::value_type, N, M>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}
  //----------------------------------------------------------------------------
  constexpr auto operator()(const size_t r, const size_t c) const -> const
      auto& {
    return m_internal_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(const size_t r, const size_t c) -> auto& {
    return m_internal_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  constexpr auto at(const size_t r, const size_t c) const -> const auto& {
    return m_internal_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(const size_t r, const size_t c) -> auto& {
    return m_internal_tensor(c, r);
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t M, size_t N>
auto transposed(const base_tensor<Tensor, Real, M, N>& t) {
  return const_transposed_tensor{t};
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transposed(base_tensor<Tensor, Real, M, N>& t) {
  return transposed_tensor<Tensor, N, M>{t};
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transposed(
    base_tensor<transposed_tensor<Tensor, M, N>, Real, M, N>& transposed_tensor)
    -> auto& {
  return transposed_tensor.as_derived().internal_tensor();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transposed(const base_tensor<transposed_tensor<Tensor, M, N>,
                                           Real, M, N>& transposed_tensor)
    -> const auto& {
  return transposed_tensor.as_derived().internal_tensor();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t M, size_t N>
constexpr auto transposed(
    const base_tensor<const_transposed_tensor<Tensor, M, N>, Real, M, N>&
        transposed_tensor) -> const auto& {
  return transposed_tensor.as_derived().internal_tensor();
}
//==============================================================================
template <typename T>
struct is_transposed_tensor : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_transposed_tensor_v = is_transposed_tensor<T>::value;
//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
struct is_transposed_tensor<const_transposed_tensor<Tensor, M, N>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
struct is_transposed_tensor<transposed_tensor<Tensor, M, N>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t M, size_t N>
struct is_transposed_tensor<
    base_tensor<transposed_tensor<Tensor, M, N>, T, M, N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t M, size_t N>
struct is_transposed_tensor<
    base_tensor<const_transposed_tensor<Tensor, M, N>, T, M, N>>
    : std::true_type {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif