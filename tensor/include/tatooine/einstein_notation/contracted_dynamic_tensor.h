#ifndef TATOOINE_EINSTEIN_NOTATION_CONTRACTED_DYNAMIC_TENSOR_H
#define TATOOINE_EINSTEIN_NOTATION_CONTRACTED_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/einstein_notation/type_traits.h>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <typename... IndexedTensors>
struct contracted_dynamic_tensor {
  using value_type =
      common_type<typename IndexedTensors::tensor_type::value_type...>;
  using indices_per_tensor = type_list<typename IndexedTensors::indices...>;
  template <std::size_t I>
  using indices_of_tensor = type_list_at<indices_per_tensor, I>;
  using free_indices =
      tatooine::einstein_notation::free_indices<IndexedTensors...>;
  using contracted_indices =
      tatooine::einstein_notation::contracted_indices<IndexedTensors...>;
  //----------------------------------------------------------------------------
  static auto constexpr rank() { return free_indices::size; }
  //----------------------------------------------------------------------------
  template <typename E, std::size_t ...Is>
  auto constexpr dimension(std::index_sequence<Is...>/*seq*/) const {
    std::size_t s = 0;
    (
        [&](auto const& indexed_tensor) {
          if constexpr (IndexedTensors::template contains<E>()) {
            s = indexed_tensor.template dimension<E>();
          }
        }(at<Is>()),
        ...);
    return s;
  }
  //----------------------------------------------------------------------------
  template <typename E>
  auto constexpr dimension() const {
    return dimension<E>(std::make_index_sequence<sizeof...(IndexedTensors)>{});
  }

 private:
  std::tuple<IndexedTensors...> m_tensors;

 public:
  contracted_dynamic_tensor(IndexedTensors... tensors) : m_tensors{tensors...} {}

  template <std::size_t I>
  auto at() const {
    return std::get<I>(m_tensors);
  }
  template <std::size_t I>
  auto at() {
    return std::get<I>(m_tensors);
  }
  constexpr auto num_tensors() { return sizeof...(IndexedTensors); }

  template <std::size_t... ContractedIndexSequence,
            std::size_t... ContractedTensorsSequence>
  requires free_indices::empty auto to_scalar(
      std::index_sequence<ContractedIndexSequence...> /*seq*/,
      std::index_sequence<ContractedTensorsSequence...> /*seq*/) const {
    using map_t              = std::map<std::size_t, std::size_t>;
    using contracted_dynamic_tensor  = contracted_dynamic_tensor<IndexedTensors...>;
    using contracted_indices = typename contracted_dynamic_tensor::contracted_indices;

    auto const contracted_indices_map = map_t{map_t::value_type{
        contracted_indices::template at<ContractedIndexSequence>::get(),
        ContractedIndexSequence,
    }...};
    auto const tensor_index_maps = std::tuple{IndexedTensors::index_map()...};
    auto       index_arrays =
        std::tuple{make_array<std::size_t, IndexedTensors::rank()>()...};
    value_type acc = 0;

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
        contracted_dynamic_tensor::template dimension<
            typename contracted_indices::template at<
                ContractedIndexSequence>>()...);
    return acc;
  }
  //----------------------------------------------------------------------------
  operator value_type() const {
    if constexpr (free_indices::empty) {
      return to_scalar(std::make_index_sequence<contracted_indices::dimension>{},
                       std::make_index_sequence<sizeof...(IndexedTensors)>{});
    } else {
      return value_type(0) / value_type(0);
    }
  }
};
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
