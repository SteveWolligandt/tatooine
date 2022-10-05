#ifndef TATOOINE_EINSTEIN_NOTATION_INDEXED_DYNAMIC_TENSOR_H
#define TATOOINE_EINSTEIN_NOTATION_INDEXED_DYNAMIC_TENSOR_H
//==============================================================================
#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
#include <tatooine/blas.h>
#endif
#include <tatooine/einstein_notation/index.h>
#include <tatooine/for_loop.h>
#include <tatooine/make_array.h>
#include <tatooine/tensor_concepts.h>
#include <tatooine/tensor_type_traits.h>
#include <tatooine/type_list.h>
#include <tatooine/type_set.h>

#include <map>
#include <tuple>
//==============================================================================
namespace tatooine::einstein_notation {
//==============================================================================
template <typename... IndexedTensors>
struct contracted_dynamic_tensor;
//==============================================================================
template <typename... ContractedTensors>
struct added_contracted_dynamic_tensor;
//==============================================================================
template <dynamic_tensor Tensor, index... Indices>
struct indexed_dynamic_tensor {
  using tensor_type = std::decay_t<Tensor>;
  using value_type = tensor_value_type<tensor_type>;

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
  static auto constexpr rank() { return sizeof...(Indices); }
  auto dimension(std::size_t const i) {
    return m_tensor.dimension(i);
  }

 private:
  template <std::size_t I, index E, index HeadIndex,
            index... TailIndices>
  auto dimension_() const {
    if constexpr (is_same<E, HeadIndex>) {
      return m_tensor.dimension(I);
    } else {
      return dimension_<I + 1, E, TailIndices...>();
    }
  }

 public:
  template <index E>
  auto dimension() const {
    return dimension_<0, E, Indices...>();
  }

  template <index E>
  static auto constexpr contains() -> bool {
    return type_list<Indices...>::template contains<E>;
  }
  //============================================================================
  template <typename... ContractedTensors, std::size_t... Seq>
  auto assign(added_contracted_dynamic_tensor<ContractedTensors...> other,
              std::index_sequence<Seq...> /*seq*/)
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    ([&] { *this += other.template at<Seq>(); }(), ...);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors, std::size_t... FreeIndexSequence,
            std::size_t... ContractedIndexSequence,
            std::size_t... ContractedTensorsSequence>
  auto add(contracted_dynamic_tensor<IndexedTensors...> other,
           std::index_sequence<FreeIndexSequence...> /*seq*/,
           std::index_sequence<ContractedIndexSequence...> /*seq*/,
           std::index_sequence<ContractedTensorsSequence...> /*seq*/)
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    using map_t              = std::map<std::size_t, std::size_t>;
    using contracted_dynamic_tensor  = contracted_dynamic_tensor<IndexedTensors...>;
    using free_indices       = typename contracted_dynamic_tensor::free_indices;
    using contracted_indices = typename contracted_dynamic_tensor::contracted_indices;

    auto const free_indices_map = map_t{
        map_t::value_type{free_indices::template at<FreeIndexSequence>::get(),
                          FreeIndexSequence}...};
    auto c = std::array{ContractedTensorsSequence...};
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

                  auto const f = std::array{free_indices...};
                  auto const g = std::array{
                      std::get<ContractedTensorsSequence>(index_arrays)...};
                  m_tensor(free_indices...) +=
                      (other.template at<ContractedTensorsSequence>().tensor()(
                           std::get<ContractedTensorsSequence>(index_arrays)) *
                       ...);
                },
                other.template dimension<typename contracted_indices::template at<ContractedIndexSequence>>()...);
          }
        },
        m_tensor.dimension(FreeIndexSequence)...);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors>
  auto operator+=(contracted_dynamic_tensor<IndexedTensors...> other)
      -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    add(other, std::make_index_sequence<rank()>{},
        std::make_index_sequence<contracted_dynamic_tensor<
            IndexedTensors...>::contracted_indices::size>{},
        std::make_index_sequence<rank()>{});
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors, index T, index... Ts>
  auto resize_internal_tensor(contracted_dynamic_tensor<IndexedTensors...> other,
                       type_set_impl<T, Ts...> const /*ts*/,
                       std::vector<std::size_t>& size) {
    size.push_back(other.template dimension<T>());
    resize_internal_tensor(other, type_set_impl<Ts...>{}, size);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors>
  auto resize_internal_tensor(
      contracted_dynamic_tensor<IndexedTensors...> /*other*/,
      type_set_impl<> const /*ts*/, std::vector<std::size_t>& size) {
    m_tensor = tensor_type::zeros(size);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors>
  auto resize_internal_tensor(
      contracted_dynamic_tensor<IndexedTensors...> other) {
    auto size = std::vector<std::size_t>{};
    resize_internal_tensor(
        other,
        typename contracted_dynamic_tensor<IndexedTensors...>::free_indices{},
        size);
  }
  //----------------------------------------------------------------------------
  template <typename... IndexedTensors>
  auto assign(contracted_dynamic_tensor<IndexedTensors...> other) 
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    resize_internal_tensor(
        other);
    add(other, std::make_index_sequence<rank()>{},
        std::make_index_sequence<
            contracted_dynamic_tensor<IndexedTensors...>::contracted_indices::size>{},
        std::make_index_sequence<rank()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... IndexedTensors>
  auto operator=(contracted_dynamic_tensor<IndexedTensors...> other)
      -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    assign(other);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... ContractedTensors>
  auto operator=(added_contracted_dynamic_tensor<ContractedTensors...> other)
      -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    m_tensor = tensor_type::zeros();
    assign(other, std::make_index_sequence<sizeof...(ContractedTensors)>{});
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensors, typename... Is>
  auto operator=(indexed_dynamic_tensor<Tensors, Is...> other)
      -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>>)
  {
    m_tensor = tensor_type::zeros();
    *this += contracted_dynamic_tensor{other};
    return *this;
  }
#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \f$\mA(i,k) = \mB(i,j) \mC(j, k) + \mA(i,k)\f$
  template <typename LHS, typename RHS, typename I, typename J, typename K>
  auto operator+=(contracted_dynamic_tensor<indexed_dynamic_tensor<LHS, I, J>,
                                            indexed_dynamic_tensor<RHS, J, K>>
                     other) -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>> &&
           is_same<value_type, tensor_value_type<LHS>> &&
           is_same<value_type, tensor_value_type<RHS>> &&
           is_same<I, index_at<0>> &&
           is_same<K, index_at<1>>)
  {
    assert(m_tensor.dimension(0) == other.template at<0>().tensor().dimension(0));
    assert(m_tensor.dimension(1) == other.template at<1>().tensor().dimension(1));
    blas::gemm(value_type(1),
               other.template at<0>().tensor(),
               other.template at<1>().tensor(), value_type(1), m_tensor);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \f$\mA(i,k) = \mB(i,j) \mC(k, j)\f$
  template <typename LHS, typename RHS, typename I, typename J, typename K>
  auto operator=(contracted_dynamic_tensor<indexed_dynamic_tensor<LHS, I, J>,
                                           indexed_dynamic_tensor<RHS, K, J>>
                     other) -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>> &&
           is_same<value_type, tensor_value_type<LHS>> &&
           is_same<value_type, tensor_value_type<RHS>> &&
           is_same<I, index_at<0>> &&
           is_same<K, index_at<1>>)
  {
    m_tensor.resize(other.template at<0>().tensor().dimension(0),
                    other.template at<1>().tensor().dimension(1));
    blas::gemm(blas::op::no_transpose, blas::op::transpose, value_type(1),
               other.template at<0>().tensor(),
               other.template at<1>().tensor(), value_type(0), m_tensor);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \f$\mA(i,k) = \mB(j,i) \mC(j, k)\f$
  template <typename LHS, typename RHS, typename I, typename J, typename K>
  auto operator=(contracted_dynamic_tensor<indexed_dynamic_tensor<LHS, J, I>,
                                           indexed_dynamic_tensor<RHS, J, K>>
                     other) -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>> &&
           is_same<value_type, tensor_value_type<LHS>> &&
           is_same<value_type, tensor_value_type<RHS>> &&
           is_same<I, index_at<0>> &&
           is_same<K, index_at<1>>)
  {
    m_tensor.resize(other.template at<0>().tensor().dimension(0),
                    other.template at<1>().tensor().dimension(1));
    blas::gemm(blas::op::transpose, blas::op::no_transpose, value_type(1),
               other.template at<0>().tensor(),
               other.template at<1>().tensor(), value_type(0), m_tensor);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \f$\mA(i,k) = \mB(i,j) \mC(j, k)\f$
  template <typename LHS, typename RHS, typename I, typename J, typename K>
  auto operator=(contracted_dynamic_tensor<indexed_dynamic_tensor<LHS, I, J>,
                                           indexed_dynamic_tensor<RHS, J, K>>
                     other) -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>> &&
           is_same<value_type, tensor_value_type<LHS>> &&
           is_same<value_type, tensor_value_type<RHS>> &&
           is_same<I, index_at<0>> &&
           is_same<K, index_at<1>>)
  {
    m_tensor.resize(other.template at<0>().tensor().dimension(0),
                    other.template at<1>().tensor().dimension(1));
    blas::gemm(value_type(1),
               other.template at<0>().tensor(),
               other.template at<1>().tensor(), value_type(0), m_tensor);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \f$\va(i) = \mB(i,j) \vc(j)\f$
  template <typename LHS, typename RHS, typename I, typename J>
  auto operator=(contracted_dynamic_tensor<indexed_dynamic_tensor<LHS, I, J>,
                                           indexed_dynamic_tensor<RHS, J>>
                     other) -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>> &&
           is_same<value_type, tensor_value_type<LHS>> &&
           is_same<value_type, tensor_value_type<RHS>> &&
           is_same<I, index_at<0>>)
  {
    m_tensor.resize(other.template at<0>().tensor().dimension(0));
    blas::gemv(value_type(1), other.template at<0>().tensor(),
               other.template at<1>().tensor(), value_type(0), m_tensor);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \f$\va(i) = \mB(i,j) \vc(j) + \va(i)\f$
  template <typename LHS, typename RHS, typename I, typename J>
  auto operator+=(contracted_dynamic_tensor<indexed_dynamic_tensor<LHS, I, J>,
                                            indexed_dynamic_tensor<RHS, J>>
                      other)
      -> indexed_dynamic_tensor&
  requires(!is_const<std::remove_reference_t<Tensor>> &&
           is_same<value_type, tensor_value_type<LHS>> &&
           is_same<value_type, tensor_value_type<RHS>> &&
           is_same<I, index_at<0>>)
  {
    assert(m_tensor.dimension(0) ==
           other.template at<0>().tensor().dimension(0));
    blas::gemm(value_type(1), other.template at<0>().tensor(),
               other.template at<1>().tensor(), value_type(1), m_tensor);
    return *this;
  }
#endif
};
//==============================================================================
}  // namespace tatooine::einstein_notation
//==============================================================================
#endif
