#ifndef TATOOINE_EINSTEIN_NOTATION_ADDED_CONTACTED_TENSOR_H
#define TATOOINE_EINSTEIN_NOTATION_ADDED_CONTACTED_TENSOR_H
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <typename... ContractedTensors>
struct added_contracted_tensor {
 private:
  std::tuple<ContractedTensors...> m_tensors;

 public:
  explicit added_contracted_tensor(ContractedTensors... tensors)
      : m_tensors{tensors...} {}
  //----------------------------------------------------------------------------
  template <std::size_t I>
  auto at() const {
    return std::get<I>(m_tensors);
  }
  template <std::size_t I>
  auto at() {
    return std::get<I>(m_tensors);
  }
};
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
