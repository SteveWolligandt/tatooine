#ifndef TATOOINE_EINSTEIN_NOTATION_H
#define TATOOINE_EINSTEIN_NOTATION_H
//==============================================================================
#include <tatooine/count_types.h>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <std::size_t I>
struct index {
  static auto constexpr i = I;
};
using i_t = index<0>;
using j_t = index<1>;
using k_t = index<2>;
using l_t = index<3>;
using m_t = index<4>;
using n_t = index<5>;
using o_t = index<6>;

[[maybe_unused]] static auto constexpr inline i = i_t{};
[[maybe_unused]] static auto constexpr inline j = j_t{};
[[maybe_unused]] static auto constexpr inline k = k_t{};
[[maybe_unused]] static auto constexpr inline l = l_t{};
[[maybe_unused]] static auto constexpr inline m = m_t{};
[[maybe_unused]] static auto constexpr inline n = n_t{};
[[maybe_unused]] static auto constexpr inline o = o_t{};
//==============================================================================
template <typename IndexCounter, typename... FreeIndices>
struct free_indices_impl;
//------------------------------------------------------------------------------
template <typename... Indices>
using free_indices = typename free_indices_impl<count_types<Indices...>>::type;
//------------------------------------------------------------------------------
template <typename CurIndex, std::size_t N, typename... Counts,
          typename... FreeIndices>
struct free_indices_impl<
    type_counter_impl<type_number_pair<CurIndex, N>, Counts...>,
    FreeIndices...> {
  using type = std::conditional_t<
      (N == 1),
      typename free_indices_impl<type_counter_impl<Counts...>, FreeIndices...,
                                 CurIndex>::type,
      typename free_indices_impl<type_counter_impl<Counts...>,
                                 FreeIndices...>::type>;
};
//------------------------------------------------------------------------------
template <typename... FreeIndices>
struct free_indices_impl<type_counter_impl<>, FreeIndices...> {
  using type = static_type_set<FreeIndices...>;
};
//==============================================================================
template <typename IndexCounter, typename... FreeIndices>
struct contracted_indices_impl;
//------------------------------------------------------------------------------
template <typename... Indices>
using contracted_indices =
    typename contracted_indices_impl<count_types<Indices...>>::type;
//------------------------------------------------------------------------------
template <typename CurIndex, std::size_t N, typename... Counts,
          typename... FreeIndices>
struct contracted_indices_impl<
    type_counter_impl<type_number_pair<CurIndex, N>, Counts...>,
    FreeIndices...> {
  using type = std::conditional_t<
      (N != 1),
      typename contracted_indices_impl<type_counter_impl<Counts...>,
                                       FreeIndices..., CurIndex>::type,
      typename contracted_indices_impl<type_counter_impl<Counts...>,
                                       FreeIndices...>::type>;
};
//------------------------------------------------------------------------------
template <typename... FreeIndices>
struct contracted_indices_impl<type_counter_impl<>, FreeIndices...> {
  using type = static_type_set<FreeIndices...>;
};
//==============================================================================
template <typename Tensor, typename... Indices>
struct indexed_tensor {

};
//==============================================================================
template <typename... IndexedTensors>
struct contracted_tensor {
  using index_tuples_t = std::tuple<typename IndexedTensors::index_tuple...>;
  template <size_t I>
  using index_tuple_at = std::tuple_element_t<I, index_tuples_t>;
  template <size_t I, size_t J>
  using contracted_indices =
      binary_contraction<std::tuple_element_t<I, index_tuples_t>,
                         std::tuple_element_t<J, index_tuples_t>>;
  using contracted_indices =
      binary_contraction<std::tuple_element_t<I, index_tuples_t>,
                         std::tuple_element_t<J, index_tuples_t>>;
};
//==============================================================================
template <typename Tensor, typename... Indices>
struct tensor {
  using index_tuple = std::tuple<Indices...>;
  template <typename OtherIndexTuple>
  using mapping = index::mapping<index_tuple, OtherIndexTuple>;

  static auto constexpr rank() { return Tensor::rank(); }

  template <size_t I>
  static auto constexpr size() {
    return Tensor::template size<I>();
  }
  //============================================================================
  template <typename... IndexedTensors, size_t... FreeIndexSequence
            //, size_t... ContractedIndexSequence
            >
  auto assign(index::contracted_tensor<IndexedTensors...> /*unused*/,
              std::index_sequence<FreeIndexSequence...>
              //, std::index_sequence<ContractedIndexSequence...>
              ) -> tensor& {
    using contracted_tensor = index::contracted_tensor<IndexedTensors...>;
    using index_tuples      = typename contracted_tensor::index_tuples_t;
    //  using contractions    = typename contracted_tensor::contracted_indices;
    //  auto lhs_index_map    = index_pairs_as_array<mapping<lhs_index_tuple>>;
    //  auto rhs_index_map    = index_pairs_as_array<mapping<rhs_index_tuple>>;
    //
    //  auto lhs_index_array = std::array<size_t, LHSIndexedTensor::rank()>{};
    //  auto rhs_index_array = std::array<size_t, RHSIndexedTensor::rank()>{};
    //  for_loop(
    //      [&](auto const... free_indices) {
    //        auto const free_index_array = std::array{free_indices...};
    //        for (auto const& [l, rl] : lhs_index_map) {
    //          lhs_index_array[rl] = free_index_array[l];
    //        }
    //        for (auto const& [l, rr] : rhs_index_map) {
    //          rhs_index_array[rr] = free_index_array[l];
    //        }
    //
    //        for_loop(
    //            [&](auto const... contracted_indices) {
    //              auto const contracted_index_array =
    //                  std::array{contracted_indices...};
    //              (
    //                  [&] {
    //                    lhs_index_array[first_of_index_pair<std::tuple_element_t<
    //                        ContractedIndexSequence, contractions>>::value] =
    //                        contracted_index_array[ContractedIndexSequence];
    //                  }(),
    //                  ...);
    //              (
    //                  [&] {
    //                    rhs_index_array[second_of_index_pair<std::tuple_element_t<
    //                        ContractedIndexSequence, contractions>>::value] =
    //                        contracted_index_array[ContractedIndexSequence];
    //                  }(),
    //                  ...);
    //              std::cerr << "(";
    //              std::cerr << free_index_array.front();
    //              for (size_t i = 1; i < free_index_array.size(); ++i) {
    //                std::cerr << ", " << free_index_array[i];
    //              }
    //              std::cerr << ") += (";
    //              std::cerr << lhs_index_array.front();
    //              for (size_t i = 1; i < lhs_index_array.size(); ++i) {
    //                std::cerr << ", " << lhs_index_array[i];
    //              }
    //              std::cerr << ") * (";
    //              std::cerr << rhs_index_array.front();
    //              for (size_t i = 1; i < rhs_index_array.size(); ++i) {
    //                std::cerr << ", " << rhs_index_array[i];
    //              }
    //              std::cerr << ")\n";
    //            },
    //            LHSIndexedTensor::template size<
    //                first_of_index_pair<std::tuple_element_t<
    //                    ContractedIndexSequence, contractions>>::value>()...);
    //      },
    //      Tensor::template size<FreeIndexSequence>()...);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // template <typename OtherTensor, typename... OtherIndices,
  //          size_t... IndexSequence>
  // auto assign(index::tensor<OtherTensor, OtherIndices...> [>unused<],
  //            std::index_sequence<IndexSequence...>) -> auto& {
  //  using other_index_tuple = std::tuple<OtherIndices...>;
  //  using index_map_t       = mapping<other_index_tuple>;
  //
  //  // print_index_pairs<index_map_t>();
  //  static auto constexpr index_map = index_pairs_as_array<index_map_t>;
  //
  //  for_loop(
  //      [](auto const... indices) {
  //        auto const lhs_indices = std::array{indices...};
  //        auto const rhs_indices = [&] {
  //          auto rhs_indices = std::array<size_t, OtherTensor::rank()>{};
  //          for (auto const& [l, r] : index_map) {
  //            rhs_indices[r] = lhs_indices[l];
  //          }
  //          return rhs_indices;
  //        }();
  //
  //        // std::cerr << "lhs(";
  //        // std::cerr << lhs_indices.front();
  //        // for (size_t i = 1; i < lhs_indices.size(); ++i) {
  //        //  std::cerr << ", " << lhs_indices[i];
  //        //}
  //        // std::cerr << ") = rhs(";
  //        // std::cerr << rhs_indices.front();
  //        // for (size_t i = 1; i < rhs_indices.size(); ++i) {
  //        //  std::cerr << ", " << rhs_indices[i];
  //        //}
  //        // std::cerr << ")\n";
  //      },
  //      Tensor::template size<IndexSequence>()...);
  //
  //  return *this;
  //}
  //----------------------------------------------------------------------------
  // template <typename OtherTensor, typename... OtherIndices>
  // auto operator=(index::tensor<OtherTensor, OtherIndices...> other) ->
  // tensor& {
  //  return assign(other, std::make_index_sequence<Tensor::rank()>{});
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... IndexedTensors>
  auto operator=(index::contracted_tensor<IndexedTensors...> other) -> tensor& {
    // using contractions =
    //    typename index::contracted_tensor<LHSIndexedTensor,
    //                                      RHSIndexedTensor>::contracted_indices;
    return assign(
        other, std::make_index_sequence<Tensor::rank()>{}
        //, std::make_index_sequence<std::tuple_size_v<contractions>>{}
    );
  }
};
template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(tensor<TensorLHS, IndicesLHS...> /*unused*/,
               tensor<TensorRHS, IndicesRHS...> /*unused*/) {
  return contracted_tensor<tensor<TensorLHS, IndicesLHS...>,
                           tensor<TensorRHS, IndicesRHS...>>{};
}
template <typename... IndexedTensorLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(contracted_tensor<IndexedTensorLHS...> /*unused*/,
               tensor<TensorRHS, IndicesRHS...> /*unused*/) {
  return contracted_tensor<IndexedTensorLHS...,
                           tensor<TensorRHS, IndicesRHS...>>{};
}
template <typename... IndicesLHS, typename TensorLHS,
          typename... IndexedTensorRHS>
auto operator*(tensor<TensorLHS, IndicesLHS...> /*unused*/,
               contracted_tensor<IndexedTensorRHS...> /*unused*/) {
  return contracted_tensor<tensor<TensorLHS, IndicesLHS...>,
                           contracted_tensor<IndexedTensorRHS...>>{};
}
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
