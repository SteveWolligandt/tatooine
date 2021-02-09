#ifndef TATOOINE_TENSOR_SLICE_H
#define TATOOINE_TENSOR_SLICE_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T, size_t FixedDim, size_t... Dims>
struct tensor_slice
    : base_tensor<tensor_slice<Tensor, T, FixedDim, Dims...>, T, Dims...> {
  using tensor_t          = Tensor;
  using this_t            = tensor_slice<Tensor, T, FixedDim, Dims...>;
  using parent_t          = base_tensor<this_t, T, Dims...>;
  using parent_t::operator=;
  using parent_t::num_components;
  using parent_t::rank;

  //============================================================================
 private:
  Tensor* m_tensor;
  size_t  m_fixed_index;

  //============================================================================
 public:
  constexpr tensor_slice(Tensor* tensor, size_t fixed_index)
      : m_tensor{tensor}, m_fixed_index{fixed_index} {}

  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  constexpr auto at(Is const... is) const -> decltype(auto) {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == rank()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(at_, unpack(extract<0, FixedDim - 1>(is...)),
                             m_fixed_index,
                             unpack(extract<FixedDim, rank() - 1>(is...)));
    };
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename _Tensor              = Tensor,
            integral... Is,
            enable_if<is_non_const<_Tensor>> = true>
#else
  template <typename _Tensor              = Tensor,
            typename... Is,
            enable_if<is_integral<Is...>>     = true,
            enable_if<is_non_const<_Tensor>> = true>
#endif
  constexpr auto at(Is const... is) -> decltype(auto) {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == rank()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(at_, unpack(extract<0, FixedDim - 1>(is...)),
                             m_fixed_index,
                             unpack(extract<FixedDim, rank() - 1>(is...)));
    };
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
