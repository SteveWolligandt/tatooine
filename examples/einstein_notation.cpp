#include <tatooine/for_loop.h>

#include <iostream>
#include <tuple>
using namespace tatooine;
namespace index {
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


template <typename IndexTuple, size_t I = 0>
auto print_indices() {
  std::cout << "("
            << first_of_index_pair<std::tuple_element_t<I, IndexTuple>>::value
            << ", ";
  std::cout << second_of_index_pair<std::tuple_element_t<I, IndexTuple>>::value
            << ") ";
  if constexpr (I < std::tuple_size_v<IndexTuple> - 1) {
    print_indices<IndexTuple, I + 1>();
  }
}
template <std::size_t I>
struct base_t {
  static auto constexpr i = I;
};
static auto constexpr i = base_t<0>{};
static auto constexpr j = base_t<1>{};
static auto constexpr k = base_t<2>{};
static auto constexpr l = base_t<3>{};

// LHS_I < size(LHS)- 1
// RHS_I < size(RHS)- 1
// Different Types
// increase LHS_I , do not store new type
template <typename LHS, typename LHS_Type_I, size_t LHS_I,
          typename RHS, typename RHS_Type_I, size_t RHS_I,
          typename... CollectedIndices>
struct contraction {
  using type = typename contraction<
    LHS, std::tuple_element_t<LHS_I + 1, LHS>, LHS_I + 1,
    RHS, RHS_Type_I, RHS_I,
    CollectedIndices...>::type;
};

// LHS_I < size(LHS)- 1
// RHS_I < size(RHS)- 1
// Same Types
// increase LHS_I , store new type
template <typename LHS, size_t LHS_I,
          typename RHS, size_t RHS_I,
          typename SameType,
          typename... CollectedIndices>
struct contraction<LHS, SameType, LHS_I,
                   RHS, SameType, RHS_I,
                   CollectedIndices...> {
  using type = typename contraction<
    LHS, std::tuple_element_t<LHS_I + 1, LHS>, LHS_I + 1,
    RHS, SameType, RHS_I, CollectedIndices...,
    index_pair<LHS_I, RHS_I>>::type;
};

// LHS_I == size(LHS)- 1
// RHS_I < size(RHS)- 1
// Different Types
// LHS_I = 0, increase RHS_I, do not store new type
template <typename LHS, typename LHS_Type_I,
          typename RHS, typename RHS_Type_I, size_t RHS_I,
          typename... CollectedIndices>
struct contraction<LHS, LHS_Type_I, std::tuple_size_v<LHS> - 1,
                   RHS, RHS_Type_I, RHS_I,
                   CollectedIndices...> {
  using type = typename contraction<
      LHS, std::tuple_element_t<0, LHS>, 0,
      RHS, std::tuple_element_t<RHS_I + 1, RHS>, RHS_I + 1,
      CollectedIndices...>::type;
};

// LHS_I == size(LHS)- 1
// RHS_I < size(RHS)- 1
// Same Types
// LHS_I = 0, increase RHS_I, store new type
template <typename LHS,
          typename RHS, size_t RHS_I,
          typename SameType,
          typename... CollectedIndices>
struct contraction<LHS, SameType, std::tuple_size_v<LHS> - 1,
                   RHS, SameType, RHS_I,
                   CollectedIndices...> {
  using type = typename contraction<
    LHS, std::tuple_element_t<0, LHS>, 0,
    RHS, std::tuple_element_t<RHS_I + 1, RHS>, RHS_I + 1,
    CollectedIndices..., index_pair<std::tuple_size_v<LHS> - 1, RHS_I>>::type;
};

// LHS_I == size(LHS)- 1
// RHS_I == size(RHS)- 1
// Different Types
// end of recursion, do not store new type
template <typename LHS, typename LHS_Type_I,
          typename RHS, typename RHS_Type_I,
          typename... CollectedIndices>
struct contraction<LHS, LHS_Type_I, std::tuple_size_v<LHS> - 1,
                   RHS, RHS_Type_I, std::tuple_size_v<RHS> - 1,
                   CollectedIndices...> {
  using type = std::tuple<CollectedIndices...>;
};

// LHS_I == size(LHS)- 1
// RHS_I == size(RHS)- 1
// Same Types
// end of recursion, store new type
template <typename LHS,
          typename RHS,
          typename SameType,
          typename... CollectedIndices>
struct contraction<LHS, SameType, std::tuple_size_v<LHS> - 1,
                   RHS, SameType, std::tuple_size_v<RHS> - 1,
                   CollectedIndices...> {
  using type = std::tuple<CollectedIndices...,
                          index_pair<std::tuple_size_v<LHS> - 1,
                                     std::tuple_size_v<RHS> - 1>>;
};

template <typename IndexedTensorLHS, typename IndexedTensorRHS>
struct contracted_tensor {
  using LHSIndexTuple      = typename IndexedTensorLHS::index_tuple;
  using RHSIndexTuple      = typename IndexedTensorRHS::index_tuple;
  using contracted_indices = typename contraction<
      LHSIndexTuple, std::tuple_element_t<0, LHSIndexTuple>, 0, RHSIndexTuple,
      std::tuple_element_t<0, RHSIndexTuple>, 0>::type;
};
template <typename Tensor, typename... Indices>
struct tensor {
  using index_tuple = std::tuple<Indices...>;
  template <typename LHS, typename RHS>
  auto operator=(index::contracted_tensor<LHS, RHS>) -> tensor& {}
};
}  // namespace index

template <size_t I, size_t CurNum, size_t... RestNums>
struct ith_num {
  static auto constexpr value = ith_num<I + 1, RestNums...>::value;
};
template <size_t CurNum, size_t... RestNums>
struct ith_num<0, CurNum, RestNums...> {
  static auto constexpr value = CurNum;
};

template <typename Tensor, size_t I>
struct tensor_size;

template <size_t... Dimensions>
struct tensor {
  using this_t = tensor;
  static auto constexpr rank() { return sizeof...(Dimensions); }
  template <size_t I>
  static auto size() {
    return tensor_size<tensor, I>::value;
  }
  template <typename... Indices>
  auto operator()(Indices...) {
    return index::tensor<this_t, Indices...>{};
  }
};

template <size_t I, size_t... Dimensions>
struct tensor_size<tensor<Dimensions...>, I> : ith_num<I, Dimensions...> {};


template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(index::tensor<TensorLHS, IndicesLHS...>,
               index::tensor<TensorRHS, IndicesRHS...>) {
  return index::contracted_tensor<index::tensor<TensorLHS, IndicesLHS...>,
                                  index::tensor<TensorRHS, IndicesRHS...>>{};
}

auto main() -> int {
  using namespace index;
  auto T33  = tensor<3, 3>{};
  auto T333 = tensor<3, 3, 3>{};
  T33(i, k) * T333(i, k, j);
}
