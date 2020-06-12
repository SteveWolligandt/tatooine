#ifndef TATOOINE_COMPLEX_TENSOR_VIEWS_H
#define TATOOINE_COMPLEX_TENSOR_VIEWS_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, real_number T, size_t... Dims>
struct const_imag_complex_tensor
    : base_tensor<const_imag_complex_tensor<Tensor, T, Dims...>, T,
                  Dims...> {
  static_assert(std::is_same_v<Tensor::value_type, std::complex<T>>);
  using this_t   = const_imag_complex_tensor<Tensor, T, Dims...>;
  using parent_t = base_tensor<this_t, T, Dims...>;
  using parent_t::rank;

  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit constexpr const_imag_complex_tensor(
      const base_tensor<Tensor, std::complex<T>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr decltype(auto) operator()(integral auto const... is) const {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr decltype(auto) at(integral auto const... is) const {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).imag();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, real_number T, size_t... Dims>
struct imag_complex_tensor
    : base_tensor<imag_complex_tensor<Tensor, T, Dims...>, T, Dims...> {
  static_assert(std::is_same_v<typename Tensor::value_type, std::complex<T>>);
  using this_t   = imag_complex_tensor<Tensor, T, Dims...>;
  using parent_t = base_tensor<this_t, T, Dims...>;
  using parent_t::rank;
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit constexpr imag_complex_tensor(
      base_tensor<Tensor, std::complex<T>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(integral auto const... is) -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).imag();
  }
  //----------------------------------------------------------------------------
  constexpr auto at(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).imag();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral auto const... is) -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).imag();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t... Dims>
auto imag(const base_tensor<Tensor, std::complex<T>, Dims...>& tensor) {
  return const_imag_complex_tensor<Tensor, T, Dims...>{tensor.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_number T, size_t... Dims>
auto imag(base_tensor<Tensor, std::complex<T>, Dims...>& tensor) {
  return imag_complex_tensor<Tensor, T, Dims...>{tensor.as_derived()};
}

//==============================================================================
template <typename Tensor, real_number T, size_t... Dims>
struct const_real_complex_tensor
    : base_tensor<const_real_complex_tensor<Tensor, T, Dims...>, T,
                  Dims...> {
  static_assert(std::is_same_v<typename Tensor::value_type, std::complex<T>>);
  using this_t   = const_real_complex_tensor<Tensor, T, Dims...>;
  using parent_t = base_tensor<this_t, T, Dims...>;
  using parent_t::rank;
  //============================================================================
 private:
  const Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit const_real_complex_tensor(
      const base_tensor<Tensor, std::complex<T>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).real();
  }
  //----------------------------------------------------------------------------
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//==============================================================================
template <typename Tensor, real_number T, size_t... Dims>
struct real_complex_tensor
    : base_tensor<real_complex_tensor<Tensor, T, Dims...>, T, Dims...> {
  static_assert(std::is_same_v<typename Tensor::value_type, std::complex<T>>);
  using this_t   = real_complex_tensor<Tensor, T, Dims...>;
  using parent_t = base_tensor<this_t, T, Dims...>;
  using parent_t::rank;
  //============================================================================
 private:
  Tensor& m_internal_tensor;

  //============================================================================
 public:
  explicit real_complex_tensor(
      base_tensor<Tensor, std::complex<T>, Dims...>& internal_tensor)
      : m_internal_tensor{internal_tensor.as_derived()} {}

  //----------------------------------------------------------------------------
  constexpr auto operator()(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(integral auto const... is) -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).real();
  }
  //----------------------------------------------------------------------------
  constexpr auto at(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).real();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral auto const... is) -> decltype(auto) {
    static_assert(sizeof...(is) == rank());
    return m_internal_tensor(is...).real();
  }

  //----------------------------------------------------------------------------
  auto internal_tensor() -> auto& { return m_internal_tensor; }
  auto internal_tensor() const -> const auto& { return m_internal_tensor; }
};

//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t... Dims>
auto real(const base_tensor<Tensor, std::complex<T>, Dims...>& t) {
  return const_real_complex_tensor<Tensor, T, Dims...>{t.as_derived()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_number T, size_t... Dims>
auto real(base_tensor<Tensor, std::complex<T>, Dims...>& t) {
  return real_complex_tensor<Tensor, T, Dims...>{t.as_derived()};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
