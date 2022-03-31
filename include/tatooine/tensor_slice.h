#ifndef TATOOINE_TENSOR_SLICE_H
#define TATOOINE_TENSOR_SLICE_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, arithmetic_or_complex ValueType, size_t FixedDim,
          size_t... Dims>
struct tensor_slice
    : base_tensor<tensor_slice<Tensor, ValueType, FixedDim, Dims...>, ValueType,
                  Dims...> {
  using tensor_type       = Tensor;
  using this_type         = tensor_slice<Tensor, ValueType, FixedDim, Dims...>;
  using parent_t          = base_tensor<this_type, ValueType, Dims...>;
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
  constexpr auto at(integral auto const... is) const -> decltype(auto) {
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
  constexpr auto at(integral auto const... is)
      -> decltype(auto) requires is_non_const<Tensor> {
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
