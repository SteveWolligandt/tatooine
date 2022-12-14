#ifndef TATOOINE_TENSOR_SLICE_H
#define TATOOINE_TENSOR_SLICE_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/variadic_helpers.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, arithmetic_or_complex ValueType, std::size_t FixedDim,
          std::size_t... Dims>
struct tensor_slice
    : base_tensor<tensor_slice<Tensor, ValueType, FixedDim, Dims...>, ValueType,
                  Dims...> {
  using tensor_type       = Tensor;
  using this_type         = tensor_slice<Tensor, ValueType, FixedDim, Dims...>;
  using parent_type       = base_tensor<this_type, ValueType, Dims...>;
  using typename parent_type::value_type;

  using parent_type::operator=;
  //template <static_tensor Other>
  //requires (same_dimensions<this_type, Other>()) &&
  //         (convertible_to<value_type<std::decay_t<Other>>, value_type>)
  //auto operator=(Other&& other) -> tensor_slice& {
  //  parent_type::operator=(std::forward<Other>(other));
  //  return *this;
  //}

  using parent_type::num_components;
  using parent_type::rank;

  //============================================================================
 private:
  Tensor* m_tensor;
  std::size_t  m_fixed_index;

  //============================================================================
 public:
  constexpr tensor_slice(Tensor* tensor, std::size_t fixed_index)
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
      return invoke_unpacked(at_,
                             unpack(variadic::extract<0, FixedDim>(
                                 static_cast<std::size_t>(is)...)),
                             m_fixed_index,
                             unpack(variadic::extract<FixedDim, rank()>(
                                 static_cast<std::size_t>(is)...)));
    };
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(integral auto const... is)
      -> decltype(auto)
      requires is_non_const<Tensor> {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == rank()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(at_,
                             unpack(variadic::extract<0, FixedDim>(
                                 static_cast<std::size_t>(is)...)),
                             m_fixed_index,
                             unpack(variadic::extract<FixedDim, rank()>(
                                 static_cast<std::size_t>(is)...)));
    };
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
