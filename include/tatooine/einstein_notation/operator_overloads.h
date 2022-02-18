#ifndef TATOOINE_EINSTEIN_NOTATION_OPERATOR_OVERLOADS_H
#define TATOOINE_EINSTEIN_NOTATION_OPERATOR_OVERLOADS_H
//==============================================================================
#include <tatooine/einstein_notation/indexed_static_tensor.h>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <typename TensorLHS, index... IndicesLHS, typename TensorRHS,
          index... IndicesRHS>
auto constexpr contract(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
                        indexed_static_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contracted_tensor<indexed_static_tensor<TensorLHS, IndicesLHS...>,
                           indexed_static_tensor<TensorRHS, IndicesRHS...>>{
      lhs, rhs};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorLHS, index... IndicesLHS, typename TensorRHS,
          index... IndicesRHS>
auto operator*(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
               indexed_static_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contract(lhs, rhs);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndexedTensorLHS, typename TensorRHS, index... IndicesRHS,
          std::size_t... LHSSeq>
auto contract(contracted_tensor<IndexedTensorLHS...>          lhs,
              indexed_static_tensor<TensorRHS, IndicesRHS...> rhs,
              std::index_sequence<LHSSeq...> /*lhs_seq*/) {
  return contracted_tensor<IndexedTensorLHS...,
                           indexed_static_tensor<TensorRHS, IndicesRHS...>>{
      lhs.template at<LHSSeq>()..., rhs};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndexedTensorLHS, typename TensorRHS, index... IndicesRHS>
auto contract(contracted_tensor<IndexedTensorLHS...>          lhs,
              indexed_static_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contract(lhs, rhs,
                  std::make_index_sequence<sizeof...(IndexedTensorLHS)>{});
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndexedTensorLHS, typename TensorRHS, index... IndicesRHS>
auto operator*(contracted_tensor<IndexedTensorLHS...>          lhs,
               indexed_static_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contract(lhs, rhs);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <index... IndicesLHS, typename TensorLHS, typename... IndexedTensorRHS,
          std::size_t... RHSSeq>
auto contract(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
              contracted_tensor<IndexedTensorRHS...>          rhs,
              std::index_sequence<RHSSeq...> /*rhs_seq*/) {
  return contracted_tensor<indexed_static_tensor<TensorLHS, IndicesLHS...>,
                           IndexedTensorRHS...>{lhs,
                                                rhs.template at<RHSSeq>()...};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <index... IndicesLHS, typename TensorLHS, typename... IndexedTensorRHS>
auto contract(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
              contracted_tensor<IndexedTensorRHS...>          rhs) {
  return contract(lhs, rhs,
                  std::make_index_sequence<sizeof...(IndexedTensorRHS)>{});
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorLHS, index... IndicesLHS, typename... IndexedTensorRHS>
auto operator*(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
               contracted_tensor<IndexedTensorRHS...>          rhs) {
  return contract(lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename TensorLHS, index... IndicesLHS, typename TensorRHS,
          index... IndicesRHS>
auto operator+(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
               indexed_static_tensor<TensorRHS, IndicesRHS...> rhs) {
  return added_contracted_tensor{
      contracted_tensor<indexed_static_tensor<TensorLHS, IndicesLHS...>>{lhs},
      contracted_tensor<indexed_static_tensor<TensorRHS, IndicesRHS...>>{rhs}};
}
//------------------------------------------------------------------------------
template <typename... IndexedTensorLHS, typename TensorRHS, index... IndicesRHS>
auto operator+(contracted_tensor<IndexedTensorLHS...>          lhs,
               indexed_static_tensor<TensorRHS, IndicesRHS...> rhs) {
  return added_contracted_tensor{
      lhs,
      contracted_tensor<indexed_static_tensor<TensorRHS, IndicesRHS...>>{rhs}};
}
//------------------------------------------------------------------------------
template <typename TensorLHS, index... IndicesLHS, typename... TensorsRHS>
auto operator+(indexed_static_tensor<TensorLHS, IndicesLHS...> lhs,
               contracted_tensor<TensorsRHS...>                rhs) {
  return added_contracted_tensor{
      contracted_tensor<indexed_static_tensor<TensorLHS, IndicesLHS...>>{lhs},
      rhs};
}
//------------------------------------------------------------------------------
template <typename... TensorsLHS, typename... TensorsRHS>
auto operator+(contracted_tensor<TensorsLHS...> lhs,
               contracted_tensor<TensorsRHS...> rhs) {
  return added_contracted_tensor{lhs, rhs};
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... TensorsRHS,
          std::size_t... Seq>
auto add(added_contracted_tensor<ContractedTensorsLHS...> lhs,
         contracted_tensor<TensorsRHS...> rhs, std::index_sequence<Seq...>) {
  return added_contracted_tensor{lhs.template at<Seq>()..., rhs};
}
//------------------------------------------------------------------------------
template <typename... TensorsLHS, typename... ContractedTensorsRHS,
          std::size_t... Seq>
auto add(contracted_tensor<TensorsLHS...>                 lhs,
         added_contracted_tensor<ContractedTensorsRHS...> rhs,
         std::index_sequence<Seq...>) {
  return added_contracted_tensor{lhs, rhs.template at<Seq>()...};
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... ContractedTensorsRHS,
          std::size_t... Seq0, std::size_t... Seq1>
auto add(added_contracted_tensor<ContractedTensorsLHS...> lhs,
         added_contracted_tensor<ContractedTensorsRHS...> rhs,
         std::index_sequence<Seq0...>, std::index_sequence<Seq1...>) {
  return added_contracted_tensor{lhs.template at<Seq0>()...,
                                 rhs.template at<Seq1>()...};
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... TensorsRHS>
auto add(added_contracted_tensor<ContractedTensorsLHS...> lhs,
         contracted_tensor<TensorsRHS...>                 rhs) {
  return add(lhs, rhs,
             std::make_index_sequence<sizeof...(ContractedTensorsLHS)>{});
}
//------------------------------------------------------------------------------
template <typename... TensorsLHS, typename... ContractedTensorsRHS>
auto add(contracted_tensor<TensorsLHS...>                 lhs,
         added_contracted_tensor<ContractedTensorsRHS...> rhs) {
  return add(lhs, rhs,
             std::make_index_sequence<sizeof...(ContractedTensorsRHS)>{});
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... ContractedTensorsRHS>
auto add(added_contracted_tensor<ContractedTensorsLHS...> lhs,
         added_contracted_tensor<ContractedTensorsRHS...> rhs) {
  return add(lhs, rhs,
             std::make_index_sequence<sizeof...(ContractedTensorsLHS)>{},
             std::make_index_sequence<sizeof...(ContractedTensorsRHS)>{});
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... TensorsRHS>
auto operator+(added_contracted_tensor<ContractedTensorsLHS...> lhs,
               contracted_tensor<TensorsRHS...>                 rhs) {
  return add(lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename... TensorsLHS, typename... ContractedTensorsRHS>
auto operator+(contracted_tensor<TensorsLHS...>                 lhs,
               added_contracted_tensor<ContractedTensorsRHS...> rhs) {
  return add(lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... ContractedTensorsRHS>
auto operator+(added_contracted_tensor<ContractedTensorsLHS...> lhs,
               added_contracted_tensor<ContractedTensorsRHS...> rhs) {
  return add(lhs, rhs);
}
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
