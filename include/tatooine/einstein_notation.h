#ifndef TATOOINE_EINSTEIN_NOTATION_H
#define TATOOINE_EINSTEIN_NOTATION_H
//==============================================================================
#include <tatooine/type_counter.h>
#include <tatooine/demangling.h>
#include <tatooine/for_loop.h>
#include <tatooine/type_traits.h>

#include <array>
#include <iostream>
#include <map>
#include <utility>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <std::size_t I>
struct index {
  static auto constexpr get() { return I; }
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
template <typename... IndexedTensors>
struct contracted_tensor;
//==============================================================================
template <typename Tensor, typename... Indices>
struct indexed_tensor {
  using indices = type_list<Indices...>;
  static auto index_map() {
    return index_map(std::make_index_sequence<rank()>{});
  }
  template <size_t... Seq>
  static auto index_map(std::index_sequence<Seq...>)
      -> std::map<std::size_t, std::size_t> {
    using map_t = std::map<std::size_t, std::size_t>;
    return map_t{typename map_t::value_type{Indices::get(), Seq}...};
  };
  static auto constexpr rank() { return Tensor::rank(); }
  template <std::size_t I>
  static auto constexpr size() {
    return Tensor::template size<I>();
  }
  private:
   template <size_t I, typename E, typename HeadIndex, typename... TailIndices>
   static auto constexpr size_() {
     if constexpr (is_same<E, HeadIndex>) {
       return Tensor::dimension(I);
     } else {
       return size_<I + 1, E, TailIndices...>();
     }
   }

  public:
  template <typename E>
  static auto constexpr size() {
    return size_<0, E, Indices...>();
  }

  template <typename E>
  static auto constexpr contains() -> bool{
    return type_list<Indices...>::template contains<E>;
  }
  //============================================================================
  template <typename... IndexedTensors, std::size_t... FreeIndexSequence,
            std::size_t... ContractedIndexSequence>
  auto assign(contracted_tensor<IndexedTensors...> other,
              std::index_sequence<FreeIndexSequence...>,
              std::index_sequence<ContractedIndexSequence...>)
      -> indexed_tensor& {
    using contracted_tensor  = contracted_tensor<IndexedTensors...>;
    using free_indices       = typename contracted_tensor::free_indices;
    using contracted_indices = typename contracted_tensor::contracted_indices;

    auto index_arrays =
        std::array{std::array<std::size_t, IndexedTensors::rank()>{}...};
    using map_t                  = std::map<std::size_t, std::size_t>;
    auto const free_iterator_map = map_t{map_t::value_type{
        FreeIndexSequence,
        free_indices::template at<FreeIndexSequence>::get()}...};
    auto const contracted_iterator_map = map_t{map_t::value_type{
        ContractedIndexSequence,
        contracted_indices::template at<ContractedIndexSequence>::get()}...};
    auto const tensor_index_maps = std::array{IndexedTensors::index_map()...};

    for_loop(
        [&](auto const... free_indices) {
          auto const free_index_array = std::array{free_indices...};
          std::size_t i = 0;
          for (auto& index_array : index_arrays) {
            for (auto const& [k, v] : tensor_index_maps[i]) {
              index_array[free_iterator_map.at(v)] =
                  free_index_array[free_iterator_map.at(v)];
            }
            ++i;
          }
          for (auto& index_array : index_arrays) {
            for (auto i : index_array) {
              std::cerr << i << ' ';
            }
            std::cerr << '\n';
          }
          for_loop(
              [&](auto const... contracted_indices) {
                auto const contracted_index_array =
                    std::array{contracted_indices...};
                //              (
                //                  [&] {
                //                    lhs_index_array[first_of_index_pair<std::tuple_element_t<
                //                        ContractedIndexSequence,
                //                        contracted_indices>>::value] =
                //                        contracted_index_array[ContractedIndexSequence];
                //                  }(),
                //                  ...);
                //              (
                //                  [&] {
                //                    rhs_index_array[second_of_index_pair<std::tuple_element_t<
                //                        ContractedIndexSequence,
                //                        contracted_indices>>::value] =
                //                        contracted_index_array[ContractedIndexSequence];
                //                  }(),
                //                  ...);
                //              std::cerr << "(";
                //              std::cerr << free_index_array.front();
                //              for (std::size_t i = 1; i <
                //              free_index_array.size();
                //              ++i) {
                //                std::cerr << ", " <<
                //                free_index_array[i];
                //              }
                //              std::cerr << ") += (";
                //              std::cerr << lhs_index_array.front();
                //              for (std::size_t i = 1; i <
                //              lhs_index_array.size();
                //              ++i) {
                //                std::cerr << ", " <<
                //                lhs_index_array[i];
                //              }
                //              std::cerr << ") * (";
                //              std::cerr << rhs_index_array.front();
                //              for (std::size_t i = 1; i <
                //              rhs_index_array.size();
                //              ++i) {
                //                std::cerr << ", " <<
                //                rhs_index_array[i];
                //              }
                //              std::cerr << ")\n";
              },
              contracted_tensor::template size<
                  typename contracted_indices::template at<
                      ContractedIndexSequence>>()...);
        },
        Tensor::dimension(FreeIndexSequence)...);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // template <typename OtherTensor, typename... OtherIndices,
  //          std::size_t... IndexSequence>
  // auto assign(tensor<OtherTensor, OtherIndices...> [>unused<],
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
  //          auto rhs_indices = std::array<std::size_t, OtherTensor::rank()>{};
  //          for (auto const& [l, r] : index_map) {
  //            rhs_indices[r] = lhs_indices[l];
  //          }
  //          return rhs_indices;
  //        }();
  //
  //        // std::cerr << "lhs(";
  //        // std::cerr << lhs_indices.front();
  //        // for (std::size_t i = 1; i < lhs_indices.size(); ++i) {
  //        //  std::cerr << ", " << lhs_indices[i];
  //        //}
  //        // std::cerr << ") = rhs(";
  //        // std::cerr << rhs_indices.front();
  //        // for (std::size_t i = 1; i < rhs_indices.size(); ++i) {
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
  // auto operator=(tensor<OtherTensor, OtherIndices...> other) ->
  // tensor& {
  //  return assign(other, std::make_index_sequence<Tensor::rank()>{});
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... IndexedTensors>
  auto operator=(contracted_tensor<IndexedTensors...> other)
      -> indexed_tensor& {
    return assign(other, std::make_index_sequence<Tensor::rank()>{},
                  std::make_index_sequence<
                      contracted_tensor<IndexedTensors...>::rank()>{});
  }
};
//==============================================================================
template <typename IndexAcc, typename... Ts>
struct indexed_tensors_to_index_list_impl;
//------------------------------------------------------------------------------
template <typename... AccumulatedIndices, typename Tensor, typename... Indices,
          typename... Ts>
struct indexed_tensors_to_index_list_impl<type_list<AccumulatedIndices...>,
                                          indexed_tensor<Tensor, Indices...>,
                                          Ts...> {
  using type = typename indexed_tensors_to_index_list_impl<
      type_list<AccumulatedIndices..., Indices...>, Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename... AccumulatedIndices, size_t I, typename... Ts>
struct indexed_tensors_to_index_list_impl<type_list<AccumulatedIndices...>,
                                          index<I>, Ts...> {
  using type = typename indexed_tensors_to_index_list_impl<
      type_list<AccumulatedIndices..., index<I>>, Ts...>::type;
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
template <typename... IndexedTensors>
struct contracted_tensor {
  using indices_per_tensor = type_list<typename IndexedTensors::indices...>;
  template <std::size_t I>
  using indices_of_tensor = type_list_at<indices_per_tensor, I>;
  using free_indices =
      tatooine::einstein_notation::free_indices<IndexedTensors...>;
  using contracted_indices =
      tatooine::einstein_notation::contracted_indices<IndexedTensors...>;
  static auto constexpr rank() { return free_indices::size; }
  template <typename E>
  static auto constexpr size() {
    std::size_t s = 0;
    (
        [&] {
          if constexpr (IndexedTensors::template contains<E>()) {
            s = IndexedTensors::template size<E>();
          }
        }(),
        ...);
    return s;
  }
};
//==============================================================================
template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(indexed_tensor<TensorLHS, IndicesLHS...> /*unused*/,
               indexed_tensor<TensorRHS, IndicesRHS...> /*unused*/) {
  return contracted_tensor<indexed_tensor<TensorLHS, IndicesLHS...>,
                           indexed_tensor<TensorRHS, IndicesRHS...>>{};
}
template <typename... IndexedTensorLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(contracted_tensor<IndexedTensorLHS...> /*unused*/,
               indexed_tensor<TensorRHS, IndicesRHS...> /*unused*/) {
  return contracted_tensor<IndexedTensorLHS...,
                           indexed_tensor<TensorRHS, IndicesRHS...>>{};
}
template <typename... IndicesLHS, typename TensorLHS,
          typename... IndexedTensorRHS>
auto operator*(indexed_tensor<TensorLHS, IndicesLHS...> /*unused*/,
               contracted_tensor<IndexedTensorRHS...> /*unused*/) {
  return contracted_tensor<indexed_tensor<TensorLHS, IndicesLHS...>,
                           contracted_tensor<IndexedTensorRHS...>>{};
}
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
