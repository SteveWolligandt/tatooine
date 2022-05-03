#ifndef TATOOINE_TENSOR_UNPACK_H
#define TATOOINE_TENSOR_UNPACK_H
//==============================================================================
#include <tatooine/invoke_unpacked.h>
#include <tatooine/tensor.h>

#ifdef I
#define ____I I
#undef I
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename Real, size_t N>
struct unpack<base_tensor<Tensor, Real, N>> {
  static constexpr size_t       n = N;
  base_tensor<Tensor, Real, N>& container;
  //----------------------------------------------------------------------------
  explicit constexpr unpack(base_tensor<Tensor, Real, N>& c) : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& {
    return container(I);
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container(I);
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Tensor, typename Real, size_t N>
unpack(base_tensor<Tensor, Real, N>& c) -> unpack<base_tensor<Tensor, Real, N>>;

//==============================================================================
template <typename Tensor, typename Real, size_t N>
struct unpack<const base_tensor<Tensor, Real, N>> {
  static constexpr size_t             n = N;
  const base_tensor<Tensor, Real, N>& container;
  //----------------------------------------------------------------------------
  explicit constexpr unpack(const base_tensor<Tensor, Real, N>& c)
      : container{c} {}
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container(I);
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Tensor, typename Real, size_t N>
unpack(const base_tensor<Tensor, Real, N>& c)
    -> unpack<const base_tensor<Tensor, Real, N>>;

//==============================================================================
template <typename Real, size_t N>
struct unpack<tensor<Real, N>> {
  static constexpr size_t n = N;
  tensor<Real, N>&        container;

  //----------------------------------------------------------------------------
  explicit constexpr unpack(tensor<Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() -> auto& {
    return container[I];
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container[I];
  }
};
//==============================================================================
template <typename Real, size_t N>
unpack(tensor<Real, N>& c) -> unpack<tensor<Real, N>>;

//==============================================================================
template <typename Real, size_t N>
struct unpack<const tensor<Real, N>> {
  static constexpr size_t n = N;
  const tensor<Real, N>&  container;

  //----------------------------------------------------------------------------
  explicit constexpr unpack(const tensor<Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto get() const -> const auto& {
    return container[I];
  }
};
//==============================================================================
template <typename Real, size_t N>
unpack(const tensor<Real, N>& c) -> unpack<const tensor<Real, N>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#ifdef ____I
#define I ____I
#undef ____I
#endif
#endif
