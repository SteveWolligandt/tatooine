#ifndef TATOOINE_TRANSPOSED_TENSOR_H
#define TATOOINE_TRANSPOSED_TENSOR_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/is_transposed_tensor.h>
#include <tatooine/concepts.h>
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
      const base_tensor<Tensor, typename Tensor::value_type, N, M>&
          internal_tensor)
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
template <typename Tensor, size_t M, size_t N>
struct is_transposed_tensor_impl<const_transposed_tensor<Tensor, M, N>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
struct is_transposed_tensor_impl<transposed_tensor<Tensor, M, N>> : std::true_type {
};
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t M, size_t N>
struct is_transposed_tensor_impl<
    base_tensor<transposed_tensor<Tensor, M, N>, T, M, N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t M, size_t N>
struct is_transposed_tensor_impl<
    base_tensor<const_transposed_tensor<Tensor, M, N>, T, M, N>>
    : std::true_type {};
//==============================================================================
// dynamic tensor
//==============================================================================
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
struct const_transposed_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor const& m_tensor;
  //============================================================================
  auto internal_tensor() -> auto& { return m_tensor; }
  auto internal_tensor() const -> auto const& { return m_tensor; }
  //----------------------------------------------------------------------------
  auto size() const {
    auto s = m_tensor.size();
    std::reverse(begin(s), end(s));
    return s;
  }
  //============================================================================
  auto at(integral auto const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[const_transposed_dynamic_tensor::at] need exactly two indices"};
  }
  auto at(integral auto const r, integral auto const c) const
      -> value_type const& {
    return m_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> value_type const& {
    if (sizeof...(is) == 2) {
      return at(is...);
    }
    throw std::runtime_error{
        "[const_transposed_dynamic_tensor::operator()] need exactly two "
        "indices"};
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const i) const { return m_tensor.size(1 - i); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_dynamic_tensor_impl<const_transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_transposed_tensor_impl<const_transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
struct transposed_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor& m_tensor;
  //============================================================================
  auto internal_tensor() -> auto& { return m_tensor; }
  auto internal_tensor() const -> auto const& { return m_tensor; }
  //----------------------------------------------------------------------------
  auto size() const {
    auto s = m_tensor.size();
    std::reverse(begin(s), end(s));
    return s;
  }
  //============================================================================
  auto at(integral auto const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[transposed_dynamic_tensor::at] need exactly two indices"};
  }
  auto at(integral auto const... /*is*/) -> value_type& {
    throw std::runtime_error{
        "[transposed_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto const r, integral auto const c) const
      -> value_type const& {
    return m_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral_range auto const& indices) const -> auto const& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral_range auto const& indices) -> auto& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral_range auto indices) -> auto& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral_range auto indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto const r, integral auto const c) -> value_type& {
    return m_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> value_type const& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto const... is) -> value_type& {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const i) const { return m_tensor.size(1 - i); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_dynamic_tensor_impl<transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_transposed_tensor_impl<transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
auto transposed(DynamicTensor const& A) {
  assert(A.num_dimensions() == 2);
  return const_transposed_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
auto transposed(DynamicTensor& A) {
  assert(A.num_dimensions() == 2);
  return transposed_dynamic_tensor<DynamicTensor>{A};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
