#ifndef TATOOINE_EINSTEIN_NOTATION_TYPE_TRAITS_H
#define TATOOINE_EINSTEIN_NOTATION_TYPE_TRAITS_H
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <typename IndexAcc, typename... Ts>
struct indexed_tensors_to_index_list_impl;
//------------------------------------------------------------------------------
template <typename... AccumulatedIndices, typename Tensor, typename... Indices,
          typename... Ts>
struct indexed_tensors_to_index_list_impl<
    type_list<AccumulatedIndices...>, indexed_static_tensor<Tensor, Indices...>,
    Ts...> {
  using type = typename indexed_tensors_to_index_list_impl<
      type_list<AccumulatedIndices..., Indices...>, Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename... AccumulatedIndices, std::size_t I, typename... Ts>
struct indexed_tensors_to_index_list_impl<type_list<AccumulatedIndices...>,
                                          index_t<I>, Ts...> {
  using type = typename indexed_tensors_to_index_list_impl<
      type_list<AccumulatedIndices..., index_t<I>>, Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename... AccumulatedIndices>
struct indexed_tensors_to_index_list_impl<type_list<AccumulatedIndices...>> {
  using type = type_list<AccumulatedIndices...>;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using indexed_tensors_to_index_list =
    typename indexed_tensors_to_index_list_impl<type_list<>, Ts...>::type;
//==============================================================================
template <typename IndexCounter, typename... FreeIndices>
struct free_indices_impl;
//------------------------------------------------------------------------------
template <typename... Indices>
struct free_indices_aux;
//------------------------------------------------------------------------------
template <typename... Indices>
struct free_indices_aux<type_list<Indices...>> {
  using type = typename free_indices_impl<count_types<Indices...>>::type;
};
//------------------------------------------------------------------------------
template <typename... Indices>
using free_indices =
    typename free_indices_aux<indexed_tensors_to_index_list<Indices...>>::type;
//------------------------------------------------------------------------------
template <typename CurIndex, std::size_t N, typename... Counts,
          typename... FreeIndices>
struct free_indices_impl<type_list<type_number_pair<CurIndex, N>, Counts...>,
                         FreeIndices...> {
  using type = std::conditional_t<
      (N == 1),
      typename free_indices_impl<type_list<Counts...>, FreeIndices...,
                                 CurIndex>::type,
      typename free_indices_impl<type_list<Counts...>, FreeIndices...>::type>;
};
//------------------------------------------------------------------------------
template <typename... FreeIndices>
struct free_indices_impl<type_list<>, FreeIndices...> {
  using type = type_set<FreeIndices...>;
};
//==============================================================================
template <typename IndexCounter, typename... FreeIndices>
struct contracted_indices_impl;
//------------------------------------------------------------------------------
template <typename... Indices>
struct contracted_indices_aux;
//------------------------------------------------------------------------------
template <typename... Indices>
struct contracted_indices_aux<type_list<Indices...>> {
  using type = typename contracted_indices_impl<count_types<Indices...>>::type;
};
//------------------------------------------------------------------------------
template <typename... Indices>
using contracted_indices = typename contracted_indices_aux<
    indexed_tensors_to_index_list<Indices...>>::type;
//------------------------------------------------------------------------------
template <typename CurIndex, std::size_t N, typename... Counts,
          typename... FreeIndices>
struct contracted_indices_impl<
    type_list<type_number_pair<CurIndex, N>, Counts...>, FreeIndices...> {
  using type = std::conditional_t<
      (N != 1),
      typename contracted_indices_impl<type_list<Counts...>, FreeIndices...,
                                       CurIndex>::type,
      typename contracted_indices_impl<type_list<Counts...>,
                                       FreeIndices...>::type>;
};
//------------------------------------------------------------------------------
template <typename... FreeIndices>
struct contracted_indices_impl<type_list<>, FreeIndices...> {
  using type = type_set<FreeIndices...>;
};
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
