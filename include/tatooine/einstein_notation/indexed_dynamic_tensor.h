#ifndef TATOOINE_EINSTEIN_NOTATION_INDEXED_DYNAMIC_TENSOR_H
#define TATOOINE_EINSTEIN_NOTATION_INDEXED_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/einstein_notation/index.h>
#include <tatooine/make_array.h>
#include <tatooine/tensor_concepts.h>
#include <tatooine/type_list.h>

#include <blas.hh>
#include <map>
#include <tuple>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <typename... IndexedTensors>
struct contracted_tensor;
//==============================================================================
template <typename... ContractedTensors>
struct added_contracted_tensor;
//==============================================================================
template <dynamic_tensor Tensor, index... Indices>
struct indexed_dynamic_tensor {
  using tensor_type = std::decay_t<Tensor>;

 private:
  Tensor m_tensor;

 public:
  explicit indexed_dynamic_tensor(Tensor t) : m_tensor{t} {}

  auto tensor() const -> auto const& { return m_tensor; }
  auto tensor() -> auto& { return m_tensor; }

  using indices = type_list<Indices...>;
  template <std::size_t I>
  using index_at = typename indices::template at<I>;

  static auto index_map() {
    return index_map(std::make_index_sequence<rank()>{});
  }
  template <std::size_t... Seq>
  static auto constexpr index_map(std::index_sequence<Seq...> /*seq*/) {
    return std::array{Indices::get()...};
  }
  static auto constexpr rank() { return tensor_type::rank(); }
  template <std::size_t I>
  static auto constexpr size() {
    return tensor_type::template size<I>();
  }

 private:
  template <std::size_t I, typename E, typename HeadIndex,
            typename... TailIndices>
  static auto constexpr size_() {
    if constexpr (is_same<E, HeadIndex>) {
      return tensor_type::dimension(I);
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
  static auto constexpr contains() -> bool {
    return type_list<Indices...>::template contains<E>;
  }
  //============================================================================
  template <typename... ContractedTensors, std::size_t... Seq>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto assign(
      added_contracted_tensor<ContractedTensors...> other,
      std::index_sequence<Seq...> /*seq*/) {
    ([&] { *this += other.template at<Seq>(); }(), ...);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors, std::size_t... FreeIndexSequence,
            std::size_t... ContractedIndexSequence,
            std::size_t... ContractedTensorsSequence>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto add(
      contracted_tensor<IndexedTensors...> other,
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
    auto       index_arrays =
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
          if constexpr (contracted_indices::empty) {
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
        tensor_type::dimension(FreeIndexSequence)...);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto operator+=(
      contracted_tensor<IndexedTensors...> other) -> indexed_dynamic_tensor& {
    add(other, std::make_index_sequence<rank()>{},
        std::make_index_sequence<
            contracted_tensor<IndexedTensors...>::contracted_indices::size>{},
        std::make_index_sequence<sizeof...(IndexedTensors)>{});
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto assign(
      contracted_tensor<IndexedTensors...> other) {
    m_tensor = tensor_type::zeros();
    add(other, std::make_index_sequence<rank()>{},
        std::make_index_sequence<
            contracted_tensor<IndexedTensors...>::contracted_indices::size>{},
        std::make_index_sequence<sizeof...(IndexedTensors)>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... IndexedTensors>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto operator=(
      contracted_tensor<IndexedTensors...> other) -> indexed_dynamic_tensor& {
    assign(other);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... ContractedTensors>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto operator=(
      added_contracted_tensor<ContractedTensors...> other)
      -> indexed_dynamic_tensor& {
    m_tensor = tensor_type::zeros();
    assign(other, std::make_index_sequence<sizeof...(ContractedTensors)>{});
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensors, typename... Is>
  requires(!is_const<std::remove_reference_t<Tensor>>) auto operator=(
      indexed_dynamic_tensor<Tensors, Is...> other) -> indexed_dynamic_tensor& {
    m_tensor = tensor_type::zeros();
    *this += contracted_tensor{other};
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// A(i,k) = B(i,j) * C(j, k)
  template <typename LHS, typename RHS, typename I, typename J, typename K>
  auto operator=(contracted_tensor<indexed_dynamic_tensor<LHS, I, J>,
                                   indexed_dynamic_tensor<RHS, J, K>>
                     other) -> indexed_dynamic_tensor& {
    if constexpr (is_same<I, index_at<0>> && is_same<K, index_at<1>>) {
      using comp_type = typename tensor_type::value_type;
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
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
