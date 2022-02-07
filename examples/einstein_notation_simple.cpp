#include <tatooine/for_loop.h>

#include <iostream>
#include <tuple>
//==============================================================================
using namespace tatooine;
//==============================================================================
template <size_t I, size_t CurNum, size_t... RestNums>
struct ith_num {
  static auto constexpr value = ith_num<I - 1, RestNums...>::value;
};
template <size_t CurNum, size_t... RestNums>
struct ith_num<0, CurNum, RestNums...> {
  static auto constexpr value = CurNum;
};
template <size_t I, size_t... Nums>
[[maybe_unused]] static auto constexpr ith_num_v = ith_num<I, Nums...>::value;
//==============================================================================
template <size_t I, typename CurType, typename... RestTypes>
struct ith_type {
  using type = typename ith_type<I - 1, RestTypes...>::type;
};
template <typename CurType, typename... RestTypes>
struct ith_type<0, CurType, RestTypes...> {
  using type = CurType;
};
template <size_t I, typename... Types>
using ith_type_t = typename ith_type<I, Types...>::type;
//==============================================================================
namespace index {
//==============================================================================
template <size_t... Indices>
struct tuple {
  static auto constexpr size = sizeof...(Indices);
  static auto constexpr as_array() { return std::array{Indices...}; }
};

template <size_t I, size_t Index, size_t... Indices>
struct get_impl {
  static auto constexpr value = get_impl<I - 1, Indices...>::value;
};
template <size_t Index, size_t... Indices>
struct get_impl<0, Index, Indices...> {
  static auto constexpr value = Index;
};
template <size_t I, size_t... Indices>
static auto constexpr get(tuple<Indices...>) {
  return get_impl<I, Indices...>::value;
}

template <size_t I, size_t J>
struct index_pair {
  static auto constexpr i = I;
  static auto constexpr j = J;
};

template <typename Pair>
struct first_of_index_pair;
template <size_t I, size_t J>
struct first_of_index_pair<index_pair<I, J>> {
  static auto constexpr value = I;
};

template <typename Pair>
struct second_of_index_pair;
template <size_t I, size_t J>
struct second_of_index_pair<index_pair<I, J>> {
  static auto constexpr value = J;
};

template <typename T>
struct index_pairs_as_array_impl;
template <typename... Pairs>
struct index_pairs_as_array_impl<std::tuple<Pairs...>> {
  static auto constexpr value =
      std::array{std::pair{first_of_index_pair<Pairs>::value,
                           second_of_index_pair<Pairs>::value}...};
};
template <typename T>
static auto constexpr index_pairs_as_array =
    index_pairs_as_array_impl<T>::value;

template <typename IndexPairTuple, size_t I = 0>
auto print_index_pairs() {
  std::cerr
      << "("
      << first_of_index_pair<std::tuple_element_t<I, IndexPairTuple>>::value
      << ", ";
  std::cerr
      << second_of_index_pair<std::tuple_element_t<I, IndexPairTuple>>::value
      << ") ";
  if constexpr (I < std::tuple_size_v<IndexPairTuple> - 1) {
    print_index_pairs<IndexPairTuple, I + 1>();
  } else {
    std::cerr << '\n';
  }
}
template <typename IndexTuple, size_t I = 0>
auto print_indices() {
  std::cerr << get<I>(IndexTuple{}) << ' ';
  if constexpr (I < IndexTuple::size - 1) {
    print_indices<IndexTuple, I + 1>();
  } else {
    std::cerr << '\n';
  }
}
template <std::size_t I>
struct base_t {
  static auto constexpr i = I;
};
[[maybe_unused]] static auto constexpr inline i = base_t<0>{};
[[maybe_unused]] static auto constexpr inline j = base_t<1>{};
[[maybe_unused]] static auto constexpr inline k = base_t<2>{};
[[maybe_unused]] static auto constexpr inline l = base_t<3>{};

//==============================================================================
// LHS_I < size(LHS)- 1
// RHS_I < size(RHS)- 1
// Different Types
// increase LHS_I , do not store new type
template <typename LHS, size_t LHS_N, typename LHS_Type_I, size_t LHS_I,
          typename... CollectedIndices>
struct unary_contraction {
  // using type = typename unary_contraction<
  //  LHS, LHS_N, std::tuple_element_t<LHS_I + 1, LHS>, LHS_I + 1,
  //  RHS, RHS_N, RHS_Type_I, RHS_I,
  //  CollectedIndices...>::type;
};

//==============================================================================
// LHS_I < size(LHS)- 1
// RHS_I < size(RHS)- 1
// Different Types
// increase LHS_I , do not store new type
template <typename LHS, size_t LHS_N, typename LHS_Type_I, size_t LHS_I,
          typename RHS, size_t RHS_N, typename RHS_Type_I, size_t RHS_I,
          typename... CollectedIndices>
struct binary_contraction_impl {
  using type = typename binary_contraction_impl<
      LHS, LHS_N, std::tuple_element_t<LHS_I + 1, LHS>, LHS_I + 1, RHS, RHS_N,
      RHS_Type_I, RHS_I, CollectedIndices...>::type;
};
//==============================================================================
// LHS_I < size(LHS)- 1
// RHS_I < size(RHS)- 1
// Same Types
// increase LHS_I , store new type
template <typename LHS, size_t LHS_N, size_t LHS_I, typename RHS, size_t RHS_N,
          size_t RHS_I, typename SameType, typename... CollectedIndices>
struct binary_contraction_impl<LHS, LHS_N, SameType, LHS_I, RHS, RHS_N,
                               SameType, RHS_I, CollectedIndices...> {
  using type = typename binary_contraction_impl<
      LHS, LHS_N, std::tuple_element_t<LHS_I + 1, LHS>, LHS_I + 1, RHS, RHS_N,
      SameType, RHS_I, CollectedIndices..., index_pair<LHS_I, RHS_I>>::type;
};
//==============================================================================
// LHS_I == size(LHS)- 1
// RHS_I < size(RHS)- 1
// Different Types
// LHS_I = 0, increase RHS_I, do not store new type
template <typename LHS, size_t LHS_N, typename LHS_Type_I, typename RHS,
          size_t RHS_N, typename RHS_Type_I, size_t RHS_I,
          typename... CollectedIndices>
struct binary_contraction_impl<LHS, LHS_N, LHS_Type_I, LHS_N - 1, RHS, RHS_N,
                               RHS_Type_I, RHS_I, CollectedIndices...> {
  using type =
      typename binary_contraction_impl<LHS, LHS_N, std::tuple_element_t<0, LHS>,
                                       0, RHS, RHS_N,
                                       std::tuple_element_t<RHS_I + 1, RHS>,
                                       RHS_I + 1, CollectedIndices...>::type;
};
//==============================================================================
// LHS_I == size(LHS)- 1
// RHS_I < size(RHS)- 1
// Same Types
// LHS_I = 0, increase RHS_I, store new type
template <typename LHS, size_t LHS_N, typename RHS, size_t RHS_N, size_t RHS_I,
          typename SameType, typename... CollectedIndices>
struct binary_contraction_impl<LHS, LHS_N, SameType, LHS_N - 1, RHS, RHS_N,
                               SameType, RHS_I, CollectedIndices...> {
  using type = typename binary_contraction_impl<
      LHS, LHS_N, std::tuple_element_t<0, LHS>, 0, RHS, RHS_N,
      std::tuple_element_t<RHS_I + 1, RHS>, RHS_I + 1, CollectedIndices...,
      index_pair<LHS_N - 1, RHS_I>>::type;
};
//==============================================================================
// LHS_I == size(LHS)- 1
// RHS_I == size(RHS)- 1
// Different Types
// end of recursion, do not store new type
template <typename LHS, size_t LHS_N, typename LHS_Type_I, typename RHS,
          size_t RHS_N, typename RHS_Type_I, typename... CollectedIndices>
struct binary_contraction_impl<LHS, LHS_N, LHS_Type_I, LHS_N - 1, RHS, RHS_N,
                               RHS_Type_I, RHS_N - 1, CollectedIndices...> {
  using type = std::tuple<CollectedIndices...>;
};
//==============================================================================
// LHS_I == size(LHS)- 1
// RHS_I == size(RHS)- 1
// Same Types
// end of recursion, store new type
template <typename LHS, size_t LHS_N, typename RHS, size_t RHS_N,
          typename SameType, typename... CollectedIndices>
struct binary_contraction_impl<LHS, LHS_N, SameType, LHS_N - 1, RHS, RHS_N,
                               SameType, RHS_N - 1, CollectedIndices...> {
  using type =
      std::tuple<CollectedIndices..., index_pair<LHS_N - 1, RHS_N - 1>>;
};
//==============================================================================
template <typename LHS, typename RHS>
using binary_contraction = typename binary_contraction_impl<
    LHS, std::tuple_size_v<LHS>, std::tuple_element_t<0, LHS>, 0, RHS,
    std::tuple_size_v<RHS>, std::tuple_element_t<0, RHS>, 0>::type;
//==============================================================================
/// LHS_I < LHS_N-1
/// RHS_I < RHS_N-1
/// Different indices.
template <typename LHS, size_t LHS_N, size_t LHS_I, typename LHS_Cur,
          typename RHS, size_t RHS_N, size_t RHS_I, typename RHS_Cur,
          typename... Maps>
struct mapping_impl {
  static auto constexpr LHS_J = LHS_I;
  static auto constexpr RHS_J = RHS_I + 1;

  using LHS_Next = std::tuple_element_t<LHS_J, LHS>;
  using RHS_Next = std::tuple_element_t<RHS_J, RHS>;

  using type = typename mapping_impl<LHS, LHS_N, LHS_J, LHS_Next, RHS, RHS_N,
                                     RHS_J, RHS_Next, Maps...>::type;
};
//==============================================================================
/// LHS_I == LHS_N-1
/// RHS_I < RHS_N-1
/// Different indices.
template <typename LHS, size_t LHS_N, size_t LHS_I, typename LHS_Cur,
          typename RHS, size_t RHS_N, typename RHS_Cur, typename... Maps>
struct mapping_impl<LHS, LHS_N, LHS_I, LHS_Cur, RHS, RHS_N, RHS_N - 1, RHS_Cur,
                    Maps...> {
  static auto constexpr LHS_J = LHS_I + 1;
  static auto constexpr RHS_J = 0;

  using LHS_Next = std::tuple_element_t<LHS_J, LHS>;
  using RHS_Next = std::tuple_element_t<RHS_J, RHS>;

  using type = typename mapping_impl<LHS, LHS_N, LHS_J, LHS_Next, RHS, RHS_N,
                                     RHS_J, RHS_Next, Maps...>::type;
};
//==============================================================================
/// LHS_I == LHS_N - 1
/// RHS_I == RHS_N - 1
/// Different indices.
/// End of loop.
template <typename LHS, size_t LHS_N, typename LHS_Cur, typename RHS,
          size_t RHS_N, typename RHS_Cur, typename... Maps>
struct mapping_impl<LHS, LHS_N, LHS_N - 1, LHS_Cur, RHS, RHS_N, RHS_N - 1,
                    RHS_Cur, Maps...> {
  using type = std::tuple<Maps...>;
};
//==============================================================================
/// LHS_I < LHS_N-1
/// RHS_I < RHS_N-1
/// Same indices.
template <typename LHS, size_t LHS_N, size_t LHS_I, typename RHS, size_t RHS_N,
          size_t RHS_I, typename SameType, typename... Maps>
struct mapping_impl<LHS, LHS_N, LHS_I, SameType, RHS, RHS_N, RHS_I, SameType,
                    Maps...> {
  static auto constexpr LHS_J = LHS_I;
  static auto constexpr RHS_J = RHS_I + 1;

  using LHS_Next = std::tuple_element_t<LHS_J, LHS>;
  using RHS_Next = std::tuple_element_t<RHS_J, RHS>;

  using NewMap = index_pair<LHS_I, RHS_I>;

  using type = typename mapping_impl<LHS, LHS_N, LHS_J, LHS_Next, RHS, RHS_N,
                                     RHS_J, RHS_Next, Maps..., NewMap>::type;
};
//==============================================================================
/// LHS_I == LHS_N-1
/// RHS_I < RHS_N-1
/// Same indices.
template <typename LHS, size_t LHS_N, size_t LHS_I, typename RHS, size_t RHS_N,
          typename SameType, typename... Maps>
struct mapping_impl<LHS, LHS_N, LHS_I, SameType, RHS, RHS_N, RHS_N - 1,
                    SameType, Maps...> {
  static auto constexpr LHS_J = LHS_I + 1;
  static auto constexpr RHS_J = 0;

  using LHS_Next = std::tuple_element_t<LHS_J, LHS>;
  using RHS_Next = std::tuple_element_t<RHS_J, RHS>;

  using NewMap = index_pair<LHS_I, RHS_N - 1>;

  using type = typename mapping_impl<LHS, LHS_N, LHS_J, LHS_Next, RHS, RHS_N,
                                     RHS_J, RHS_Next, Maps..., NewMap>::type;
};
//==============================================================================
/// LHS_I == LHS_N-1
/// RHS_I == RHS_N-1
/// Same indices.
/// End of loop.
template <typename LHS, size_t LHS_N, typename RHS, size_t RHS_N,
          typename SameType, typename... Maps>
struct mapping_impl<LHS, LHS_N, LHS_N - 1, SameType, RHS, RHS_N, RHS_N - 1,
                    SameType, Maps...> {
  using NewMap = index_pair<LHS_N - 1, RHS_N - 1>;
  using type   = std::tuple<Maps..., NewMap>;
};

template <typename LHS, typename RHS>
using mapping = typename mapping_impl<
    LHS, std::tuple_size_v<LHS>, 0, std::tuple_element_t<0, LHS>, RHS,
    std::tuple_size_v<RHS>, 0, std::tuple_element_t<0, RHS>>::type;
//==============================================================================
template <typename IndexedTensorLHS, typename IndexedTensorRHS>
struct contracted_tensor {
  using LHSIndexTuple      = typename IndexedTensorLHS::index_tuple;
  using RHSIndexTuple      = typename IndexedTensorRHS::index_tuple;
  using contracted_indices = binary_contraction<LHSIndexTuple, RHSIndexTuple>;
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
  template <typename LHSIndexedTensor, typename RHSIndexedTensor,
            size_t... FreeIndexSequence, size_t... ContractedIndexSequence>
  auto assign(
      index::contracted_tensor<LHSIndexedTensor, RHSIndexedTensor> /*unused*/,
      std::index_sequence<FreeIndexSequence...>,
      std::index_sequence<ContractedIndexSequence...>) -> tensor& {
    using contracted_tensor =
        index::contracted_tensor<LHSIndexedTensor, RHSIndexedTensor>;
    using lhs_index_tuple = typename LHSIndexedTensor::index_tuple;
    using rhs_index_tuple = typename RHSIndexedTensor::index_tuple;
    using contractions    = typename contracted_tensor::contracted_indices;
    auto lhs_index_map    = index_pairs_as_array<mapping<lhs_index_tuple>>;
    auto rhs_index_map    = index_pairs_as_array<mapping<rhs_index_tuple>>;

    auto lhs_index_array = std::array<size_t, LHSIndexedTensor::rank()>{};
    auto rhs_index_array = std::array<size_t, RHSIndexedTensor::rank()>{};
    for_loop(
        [&](auto const... free_indices) {
          auto const free_index_array = std::array{free_indices...};
          for (auto const& [l, rl] : lhs_index_map) {
            lhs_index_array[rl] = free_index_array[l];
          }
          for (auto const& [l, rr] : rhs_index_map) {
            rhs_index_array[rr] = free_index_array[l];
          }

          for_loop(
              [&](auto const... contracted_indices) {
                auto const contracted_index_array =
                    std::array{contracted_indices...};
                (
                    [&] {
                      lhs_index_array[first_of_index_pair<std::tuple_element_t<
                          ContractedIndexSequence, contractions>>::value] =
                          contracted_index_array[ContractedIndexSequence];
                    }(),
                    ...);
                (
                    [&] {
                      rhs_index_array[second_of_index_pair<std::tuple_element_t<
                          ContractedIndexSequence, contractions>>::value] =
                          contracted_index_array[ContractedIndexSequence];
                    }(),
                    ...);
                std::cerr << "(";
                std::cerr << free_index_array.front();
                for (size_t i = 1; i < free_index_array.size(); ++i) {
                  std::cerr << ", " << free_index_array[i];
                }
                std::cerr << ") += (";
                std::cerr << lhs_index_array.front();
                for (size_t i = 1; i < lhs_index_array.size(); ++i) {
                  std::cerr << ", " << lhs_index_array[i];
                }
                std::cerr << ") * (";
                std::cerr << rhs_index_array.front();
                for (size_t i = 1; i < rhs_index_array.size(); ++i) {
                  std::cerr << ", " << rhs_index_array[i];
                }
                std::cerr << ")\n";
              },
              LHSIndexedTensor::template size<
                  first_of_index_pair<std::tuple_element_t<
                      ContractedIndexSequence, contractions>>::value>()...);
        },
        Tensor::template size<FreeIndexSequence>()...);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherTensor, typename... OtherIndices,
            size_t... IndexSequence>
  auto assign(index::tensor<OtherTensor, OtherIndices...> /*unused*/,
              std::index_sequence<IndexSequence...>) -> auto& {
    using other_index_tuple = std::tuple<OtherIndices...>;
    using index_map_t       = mapping<other_index_tuple>;

    // print_index_pairs<index_map_t>();
    static auto constexpr index_map = index_pairs_as_array<index_map_t>;

    for_loop(
        [](auto const... indices) {
          auto const lhs_indices = std::array{indices...};
          auto const rhs_indices = [&] {
            auto rhs_indices = std::array<size_t, OtherTensor::rank()>{};
            for (auto const& [l, r] : index_map) {
              rhs_indices[r] = lhs_indices[l];
            }
            return rhs_indices;
          }();

          // std::cerr << "lhs(";
          // std::cerr << lhs_indices.front();
          // for (size_t i = 1; i < lhs_indices.size(); ++i) {
          //  std::cerr << ", " << lhs_indices[i];
          //}
          // std::cerr << ") = rhs(";
          // std::cerr << rhs_indices.front();
          // for (size_t i = 1; i < rhs_indices.size(); ++i) {
          //  std::cerr << ", " << rhs_indices[i];
          //}
          // std::cerr << ")\n";
        },
        Tensor::template size<IndexSequence>()...);

    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename... OtherIndices>
  auto operator=(index::tensor<OtherTensor, OtherIndices...> other) -> tensor& {
    return assign(other, std::make_index_sequence<Tensor::rank()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename LHSIndexedTensor, typename RHSIndexedTensor>
  auto operator=(
      index::contracted_tensor<LHSIndexedTensor, RHSIndexedTensor> other)
      -> tensor& {
    using contractions =
        typename index::contracted_tensor<LHSIndexedTensor,
                                          RHSIndexedTensor>::contracted_indices;
    return assign(other, std::make_index_sequence<Tensor::rank()>{},
                  std::make_index_sequence<std::tuple_size_v<contractions>>{});
  }
};
//==============================================================================
}  // namespace index
//==============================================================================
template <typename Tensor, size_t I>
struct tensor_size;

template <size_t... Dimensions>
struct tensor {
  using this_type = tensor;
  static auto constexpr rank() { return sizeof...(Dimensions); }
  template <size_t I>
  static auto size() {
    return tensor_size<tensor, I>::value;
  }
  template <typename... Indices>
  auto operator()(Indices... /*unused*/) {
    return index::tensor<this_type, Indices...>{};
  }
};

template <size_t I, size_t... Dimensions>
struct tensor_size<tensor<Dimensions...>, I> : ith_num<I, Dimensions...> {};

template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(index::tensor<TensorLHS, IndicesLHS...> /*unused*/,
               index::tensor<TensorRHS, IndicesRHS...> /*unused*/) {
  return index::contracted_tensor<index::tensor<TensorLHS, IndicesLHS...>,
                                  index::tensor<TensorRHS, IndicesRHS...>>{};
}

auto main() -> int {
  using namespace index;
  [[maybe_unused]] auto T3   = ::tensor<2>{};
  [[maybe_unused]] auto T33  = ::tensor<2, 2>{};
  [[maybe_unused]] auto T333 = ::tensor<2, 2, 2>{};
  // T33(i, j) = T33(i, k) * T33(j, k);
  std::cerr << "A(i, l) = B(i, j, k) * C(l, j, k)\n";
  T33(i, l) = T333(i, j, k) * T333(l, j, k);
  // T3(i) = T333(i, j, k) * T33(j, k);
}
