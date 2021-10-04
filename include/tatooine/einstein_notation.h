#ifndef TATOOINE_EINSTEIN_NOTATION_H
#define TATOOINE_EINSTEIN_NOTATION_H
//==============================================================================
#include <tatooine/type_counter.h>
#include <tatooine/demangling.h>
#include <tatooine/for_loop.h>
#include <tatooine/type_traits.h>
#include <tatooine/blas.h>

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
using p_t = index<7>;
using q_t = index<8>;
using r_t = index<9>;
using s_t = index<10>;
using t_t = index<11>;
using u_t = index<12>;
using v_t = index<13>;
using w_t = index<14>;

[[maybe_unused]] static auto constexpr inline i = i_t{};
[[maybe_unused]] static auto constexpr inline j = j_t{};
[[maybe_unused]] static auto constexpr inline k = k_t{};
[[maybe_unused]] static auto constexpr inline l = l_t{};
[[maybe_unused]] static auto constexpr inline m = m_t{};
[[maybe_unused]] static auto constexpr inline n = n_t{};
[[maybe_unused]] static auto constexpr inline o = o_t{};
[[maybe_unused]] static auto constexpr inline p = p_t{};
[[maybe_unused]] static auto constexpr inline q = q_t{};
[[maybe_unused]] static auto constexpr inline r = r_t{};
[[maybe_unused]] static auto constexpr inline s = s_t{};
[[maybe_unused]] static auto constexpr inline t = t_t{};
[[maybe_unused]] static auto constexpr inline u = u_t{};
[[maybe_unused]] static auto constexpr inline v = v_t{};
[[maybe_unused]] static auto constexpr inline w = w_t{};

template <typename T>
struct is_index_impl : std::false_type{};
template <std::size_t N>
struct is_index_impl<index<N>> : std::true_type{};
template <typename... Ts>
static auto constexpr is_index = (is_index_impl<Ts>::value && ...);
//==============================================================================
template <typename... IndexedTensors>
struct contracted_tensor;
//==============================================================================
template <typename... ContractedTensors>
struct added_contracted_tensor;
//==============================================================================
template <typename Tensor, typename... Indices>
struct indexed_tensor {
  using tensor_t = std::decay_t<Tensor>;
 private:
  Tensor m_tensor;

 public:
  explicit indexed_tensor(Tensor t) : m_tensor{t} {}

  auto tensor() const -> auto const& { return m_tensor; }
  auto tensor()       -> auto&       { return m_tensor; }

  using indices = type_list<Indices...>;
  template <size_t I>
  using index_at = typename indices::template at<I>;

  static auto index_map() {
    return index_map(std::make_index_sequence<rank()>{});
  }
  template <size_t... Seq>
  static auto constexpr index_map(std::index_sequence<Seq...> /*seq*/) {
    return std::array{Indices::get()...};
  };
  static auto constexpr rank() { return tensor_t::rank(); }
  template <std::size_t I>
  static auto constexpr size() {
    return tensor_t::template size<I>();
  }
  private:
   template <size_t I, typename E, typename HeadIndex, typename... TailIndices>
   static auto constexpr size_() {
     if constexpr (is_same<E, HeadIndex>) {
       return tensor_t::dimension(I);
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
  template <typename... ContractedTensors, std::size_t... Seq,
            typename Tensor_                                       = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto assign(added_contracted_tensor<ContractedTensors...> other,
              std::index_sequence<Seq...>/*seq*/) {
    ([&] { *this += other.template at<Seq>(); }(), ...);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors, std::size_t... FreeIndexSequence,
            std::size_t... ContractedIndexSequence,
            std::size_t... ContractedTensorsSequence, typename Tensor_ = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto add(contracted_tensor<IndexedTensors...> other,
           std::index_sequence<FreeIndexSequence...> /*seq*/,
           std::index_sequence<ContractedIndexSequence...> /*seq*/,
           std::index_sequence<ContractedTensorsSequence...> /*seq*/) {
    using map_t              = std::map<std::size_t, std::size_t>;
    using contracted_tensor  = contracted_tensor<IndexedTensors...>;
    using free_indices       = typename contracted_tensor::free_indices;
    using contracted_indices = typename contracted_tensor::contracted_indices;

    auto const free_indices_map = map_t{
        map_t::value_type{free_indices::template at<FreeIndexSequence>::get(),
                          FreeIndexSequence}...};
    auto const contracted_indices_map = map_t{map_t::value_type{
        contracted_indices::template at<ContractedIndexSequence>::get(),
        ContractedIndexSequence,
    }...};
    auto const tensor_index_maps = std::tuple{IndexedTensors::index_map()...};
    auto index_arrays =
        std::tuple{make_array<std::size_t, IndexedTensors::rank()>()...};

    for_loop(
        [&](auto const... free_indices) {
          // setup indices of single tensors for free indices
          {
            auto const free_index_array = std::array{free_indices...};
            (
                [&] {
                  auto& index_array =
                      std::get<ContractedTensorsSequence>(index_arrays);
                  auto const& tensor_index_map =
                      std::get<ContractedTensorsSequence>(tensor_index_maps);
                  auto index_arr_it        = begin(index_array);
                  auto tensor_index_map_it = begin(tensor_index_map);

                  for (; tensor_index_map_it != end(tensor_index_map);
                       ++tensor_index_map_it, ++index_arr_it) {
                    if (free_indices_map.contains(*tensor_index_map_it)) {
                      *index_arr_it = free_index_array[free_indices_map.at(
                          *tensor_index_map_it)];
                    }
                  }
                }(),
                ...);
          }
          if constexpr (contracted_indices::empty){
            m_tensor(free_indices...) +=
                (other.template at<ContractedTensorsSequence>().tensor()(
                     std::get<ContractedTensorsSequence>(index_arrays)) *
                 ...);
          } else {
            for_loop(
                [&](auto const... contracted_indices) {
                  // setup indices of single tensors for contracted indices
                  {
                    auto const contracted_index_array =
                        std::array{contracted_indices...};
                    (
                        [&] {
                          auto& index_array =
                              std::get<ContractedTensorsSequence>(index_arrays);
                          auto const& tensor_index_map =
                              std::get<ContractedTensorsSequence>(
                                  tensor_index_maps);
                          auto index_arr_it        = begin(index_array);
                          auto tensor_index_map_it = begin(tensor_index_map);

                          for (; tensor_index_map_it != end(tensor_index_map);
                               ++tensor_index_map_it, ++index_arr_it) {
                            if (contracted_indices_map.contains(
                                    *tensor_index_map_it)) {
                              *index_arr_it = contracted_index_array
                                  [contracted_indices_map.at(
                                      *tensor_index_map_it)];
                            }
                          }
                        }(),
                        ...);
                  }

                  m_tensor(free_indices...) +=
                      (other.template at<ContractedTensorsSequence>().tensor()(
                           std::get<ContractedTensorsSequence>(index_arrays)) *
                       ...);
                },
                contracted_tensor::template size<
                    typename contracted_indices::template at<
                        ContractedIndexSequence>>()...);
          }
        },
        tensor_t::dimension(FreeIndexSequence)...);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors, typename Tensor_ = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto operator+=(contracted_tensor<IndexedTensors...> other)
      -> indexed_tensor& {
    add(other, std::make_index_sequence<rank()>{},
        std::make_index_sequence<
            contracted_tensor<IndexedTensors...>::contracted_indices::size>{},
        std::make_index_sequence<sizeof...(IndexedTensors)>{});
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors, typename Tensor_ = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto assign(contracted_tensor<IndexedTensors...> other) {
    m_tensor = tensor_t{tag::fill{0}};
    add(other, std::make_index_sequence<rank()>{},
        std::make_index_sequence<
            contracted_tensor<IndexedTensors...>::contracted_indices::size>{},
        std::make_index_sequence<sizeof...(IndexedTensors)>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... IndexedTensors, typename Tensor_ = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto operator=(contracted_tensor<IndexedTensors...> other)
      -> indexed_tensor& {
    assign(other);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... ContractedTensors, typename Tensor_ = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto operator=(added_contracted_tensor<ContractedTensors...> other)
      -> indexed_tensor& {
    m_tensor = tensor_t{tag::fill{0}};
    assign(other, std::make_index_sequence<sizeof...(ContractedTensors)>{});
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensors, typename... Is, typename Tensor_ = Tensor,
            enable_if<!is_const<std::remove_reference_t<Tensor_>>> = true>
  auto operator=(indexed_tensor<Tensors, Is...> other) -> indexed_tensor& {
    m_tensor = tensor_t{tag::fill{0}};
    *this += contracted_tensor{other};
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// A(i,k) = B(i,j) * C(j, k)
  template <typename LHS, typename RHS, typename I, typename J,typename K,
            size_t ThisRank                         = rank(),
            enable_if<ThisRank == rank(), ThisRank> = true>
  auto operator=(
      contracted_tensor<indexed_tensor<LHS, I, J>,
                        indexed_tensor<RHS, J, K>>
          other) -> indexed_tensor& {
    if constexpr (is_same<I, index_at<0>> && is_same<K, index_at<1>>) {
      using comp_type = typename tensor_t::value_type;
      static_assert(is_same<comp_type, typename std::decay_t<LHS>::value_type>);
      static_assert(is_same<comp_type, typename std::decay_t<RHS>::value_type>);
      blas::gemm(comp_type(1), other.template at<0>().tensor(),
                 other.template at<1>().tensor(), comp_type(0), m_tensor);
    } else {
      assign(other);
    }
    return *this;
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
template <typename... IndexedTensors>
struct contracted_tensor {
  using real_t = common_type<typename IndexedTensors::tensor_t::value_type...>;
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

 private:
  std::tuple<IndexedTensors...> m_tensors;

 public:
  contracted_tensor(IndexedTensors... tensors) : m_tensors{tensors...}{}

  template <std::size_t I>
  auto at() const {
    return std::get<I>(m_tensors);
  }
  template <std::size_t I>
  auto at() {
    return std::get<I>(m_tensors);
  }
  constexpr auto num_tensors() {
    return sizeof...(IndexedTensors);
  }

  template <std::size_t... ContractedIndexSequence,
            std::size_t... ContractedTensorsSequence,
            bool free_indices_empty       = free_indices::empty,
            enable_if<free_indices_empty> = true>
  auto to_scalar(
      std::index_sequence<ContractedIndexSequence...> /*seq*/,
      std::index_sequence<ContractedTensorsSequence...> /*seq*/) const {
    using map_t              = std::map<std::size_t, std::size_t>;
    using contracted_tensor  = contracted_tensor<IndexedTensors...>;
    using contracted_indices = typename contracted_tensor::contracted_indices;

    auto const contracted_indices_map = map_t{map_t::value_type{
        contracted_indices::template at<ContractedIndexSequence>::get(),
        ContractedIndexSequence,
    }...};
    auto const tensor_index_maps = std::tuple{IndexedTensors::index_map()...};
    auto       index_arrays =
        std::tuple{make_array<std::size_t, IndexedTensors::rank()>()...};
    real_t acc = 0;

    for_loop(
        [&](auto const... contracted_indices) {
          // setup indices of single tensors for contracted indices
          {
            auto const contracted_index_array =
                std::array{contracted_indices...};
            (
                [&] {
                  auto& index_array =
                      std::get<ContractedTensorsSequence>(index_arrays);
                  auto const& tensor_index_map =
                      std::get<ContractedTensorsSequence>(tensor_index_maps);
                  auto index_arr_it        = begin(index_array);
                  auto tensor_index_map_it = begin(tensor_index_map);

                  for (; tensor_index_map_it != end(tensor_index_map);
                       ++tensor_index_map_it, ++index_arr_it) {
                    if (contracted_indices_map.contains(*tensor_index_map_it)) {
                      *index_arr_it =
                          contracted_index_array[contracted_indices_map.at(
                              *tensor_index_map_it)];
                    }
                  }
                }(),
                ...);
          }

          acc += (at<ContractedTensorsSequence>().tensor()(
                      std::get<ContractedTensorsSequence>(index_arrays)) *
                  ...);
        },
        contracted_tensor::template size<
            typename contracted_indices::template at<
                ContractedIndexSequence>>()...);
    return acc;
  }
  //----------------------------------------------------------------------------
  operator real_t() const {
    if constexpr (free_indices::empty){
      return to_scalar(std::make_index_sequence<contracted_indices::size>{},
                       std::make_index_sequence<sizeof...(IndexedTensors)>{});
    } else {
      return real_t(0) / real_t(0);
    }
  }
};
//==============================================================================
template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto contract(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
              indexed_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contracted_tensor<indexed_tensor<TensorLHS, IndicesLHS...>,
                           indexed_tensor<TensorRHS, IndicesRHS...>>{lhs, rhs};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
               indexed_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contract(lhs, rhs);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndexedTensorLHS, typename TensorRHS,
          typename... IndicesRHS, std::size_t... LHSSeq>
auto contract(contracted_tensor<IndexedTensorLHS...>   lhs,
              indexed_tensor<TensorRHS, IndicesRHS...> rhs,
              std::index_sequence<LHSSeq...> /*lhs_seq*/) {
  return contracted_tensor<IndexedTensorLHS...,
                           indexed_tensor<TensorRHS, IndicesRHS...>>{
      lhs.template at<LHSSeq>()..., rhs};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndexedTensorLHS, typename TensorRHS,
          typename... IndicesRHS>
auto contract(contracted_tensor<IndexedTensorLHS...>   lhs,
              indexed_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contract(lhs, rhs,
                  std::make_index_sequence<sizeof...(IndexedTensorLHS)>{});
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndexedTensorLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator*(contracted_tensor<IndexedTensorLHS...>   lhs,
               indexed_tensor<TensorRHS, IndicesRHS...> rhs) {
  return contract(lhs, rhs);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndicesLHS, typename TensorLHS,
          typename... IndexedTensorRHS, std::size_t... RHSSeq>
auto contract(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
              contracted_tensor<IndexedTensorRHS...>   rhs,
              std::index_sequence<RHSSeq...> /*rhs_seq*/) {
  return contracted_tensor<indexed_tensor<TensorLHS, IndicesLHS...>,
                           IndexedTensorRHS...>{lhs,
                                                rhs.template at<RHSSeq>()...};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... IndicesLHS, typename TensorLHS,
          typename... IndexedTensorRHS>
auto contract(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
              contracted_tensor<IndexedTensorRHS...> rhs) {
  return contract(lhs, rhs,
                  std::make_index_sequence<sizeof...(IndexedTensorRHS)>{});
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorLHS, typename... IndicesLHS,
          typename... IndexedTensorRHS>
auto operator*(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
               contracted_tensor<IndexedTensorRHS...>   rhs) {
  return contract(lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename TensorLHS, typename... IndicesLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator+(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
               indexed_tensor<TensorRHS, IndicesRHS...> rhs) {
  return added_contracted_tensor{
      contracted_tensor<indexed_tensor<TensorLHS, IndicesLHS...>>{lhs},
      contracted_tensor<indexed_tensor<TensorRHS, IndicesRHS...>>{rhs}};
}
//------------------------------------------------------------------------------
template <typename ... IndexedTensorLHS, typename TensorRHS,
          typename... IndicesRHS>
auto operator+(contracted_tensor<IndexedTensorLHS...> lhs,
               indexed_tensor<TensorRHS, IndicesRHS...> rhs) {
  return added_contracted_tensor{
      lhs, contracted_tensor<indexed_tensor<TensorRHS, IndicesRHS...>>{rhs}};
}
//------------------------------------------------------------------------------
template <typename TensorLHS, typename... IndicesLHS,
          typename... TensorsRHS>
auto operator+(indexed_tensor<TensorLHS, IndicesLHS...> lhs,
               contracted_tensor<TensorsRHS...> rhs) {
  return added_contracted_tensor{
      contracted_tensor<indexed_tensor<TensorLHS, IndicesLHS...>>{lhs},
      rhs};
}
//------------------------------------------------------------------------------
template <typename ...TensorsLHS,
          typename... TensorsRHS>
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
auto add(contracted_tensor<TensorsLHS...>       lhs,
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
  return added_contracted_tensor{lhs.template at<Seq0>()..., rhs.template at<Seq1>()...};
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... TensorsRHS>
auto add(added_contracted_tensor<ContractedTensorsLHS...> lhs,
         contracted_tensor<TensorsRHS...>                 rhs) {
  return add(lhs, rhs, std::make_index_sequence<sizeof...(ContractedTensorsLHS)>{});
}
//------------------------------------------------------------------------------
template <typename... TensorsLHS, typename... ContractedTensorsRHS>
auto add(contracted_tensor<TensorsLHS...> lhs,
         added_contracted_tensor<ContractedTensorsRHS...>     rhs) {
  return add(lhs, rhs,
             std::make_index_sequence<sizeof...(ContractedTensorsRHS)>{});
}
//------------------------------------------------------------------------------
template <typename... ContractedTensorsLHS, typename... ContractedTensorsRHS>
auto add(added_contracted_tensor<ContractedTensorsLHS...> lhs,
         added_contracted_tensor<ContractedTensorsRHS...>     rhs) {
  return add(lhs, rhs,
             std::make_index_sequence<sizeof...(ContractedTensorsLHS)>{},
             std::make_index_sequence<sizeof...(ContractedTensorsRHS)>{});
}
//------------------------------------------------------------------------------
template <typename ...ContractedTensorsLHS,
          typename... TensorsRHS>
auto operator+(added_contracted_tensor<ContractedTensorsLHS...> lhs,
               contracted_tensor<TensorsRHS...> rhs) {
  return add(lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename ...TensorsLHS,
          typename... ContractedTensorsRHS>
auto operator+(contracted_tensor<TensorsLHS...> lhs,
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
