#ifndef TATOOINE_COMPLEX_TENSOR_VIEWS_H
#define TATOOINE_COMPLEX_TENSOR_VIEWS_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct const_imag_complex_tensor
    : base_tensor<const_imag_complex_tensor<Tensor, Real, Dims...>, Real,
                  Dims...> {
  using this_t   = const_imag_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::rank;

  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit constexpr const_imag_complex_tensor(
      const base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr decltype(auto) operator()(const Indices... indices) const {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr decltype(auto) at(const Indices... indices) const {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).imag();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct imag_complex_tensor
    : base_tensor<imag_complex_tensor<Tensor, Real, Dims...>, Real, Dims...> {
  using this_t   = imag_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::rank;
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit constexpr imag_complex_tensor(
      base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).imag();
  }
  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).imag();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
auto imag(const base_tensor<Tensor, std::complex<Real>, Dims...>& tensor) {
  return const_imag_complex_tensor<Tensor, Real, Dims...>{tensor.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
auto imag(base_tensor<Tensor, std::complex<Real>, Dims...>& tensor) {
  return imag_complex_tensor<Tensor, Real, Dims...>{tensor.as_derived()};
}

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct const_real_complex_tensor
    : base_tensor<const_real_complex_tensor<Tensor, Real, Dims...>, Real,
                  Dims...> {
  using this_t   = const_real_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::rank;
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit const_real_complex_tensor(
      const base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).real();
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct real_complex_tensor
    : base_tensor<real_complex_tensor<Tensor, Real, Dims...>, Real, Dims...> {
  using this_t   = real_complex_tensor<Tensor, Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::rank;
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit real_complex_tensor(
      base_tensor<Tensor, std::complex<Real>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto operator()(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).real();
  }
  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) const -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Indices, enable_if_integral<Indices...> = true>
  constexpr auto at(const Indices... indices) -> decltype(auto) {
    static_assert(sizeof...(Indices) == rank());
    return m_internal_tensor(indices...).real();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
auto real(const base_tensor<Tensor, std::complex<Real>, Dims...>& t) {
  return const_real_complex_tensor<Tensor, Real, Dims...>{t.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
auto real(base_tensor<Tensor, std::complex<Real>, Dims...>& t) {
  return real_complex_tensor<Tensor, Real, Dims...>{t.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
