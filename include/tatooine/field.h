#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H
//==============================================================================
#include <tatooine/crtp.h>
#include <tatooine/grid.h>
#include <tatooine/tensor.h>
#include <tatooine/tensor_type.h>
#include <tatooine/type_traits.h>

#include <vector>
//==============================================================================
namespace tatooine::polymorphic {
//==============================================================================
template <typename Real, size_t N, typename Tensor>
struct field {
  //============================================================================
  // typedefs
  //============================================================================
  using real_t   = Real;
  using this_t   = field<Real, N, Tensor>;
  using pos_t    = vec<Real, N>;
  using time_t   = Real;
  using tensor_t = Tensor;
  //============================================================================
  // static methods
  //============================================================================
  static constexpr auto is_field() { return true; }
  static constexpr auto is_scalarfield() { return is_arithmetic<Tensor>; }
  static constexpr auto is_vectorfield() { return tensor_rank() == 1; }
  static constexpr auto is_matrixfield() { return tensor_rank() == 2; }
  static constexpr auto num_dimensions() { return N; }
  //----------------------------------------------------------------------------
  static constexpr auto num_tensor_components() {
    if constexpr (is_scalarfield()) {
      return 1;
    } else {
      return tensor_t::num_components();
    }
  }
  //----------------------------------------------------------------------------
  static constexpr auto tensor_rank() {
    if constexpr (is_scalarfield()) {
      return 0;
    } else {
      return tensor_t::rank();
    }
  }
  //----------------------------------------------------------------------------
  template <size_t tensor_rank           = tensor_rank,
            enable_if<(tensor_rank > 0)> = true>
  static constexpr auto tensor_dimension(size_t i) {
    return tensor_t::dimension(i);
  }
  //============================================================================
  // ctors
  //============================================================================
  constexpr field()                 = default;
  constexpr field(field const&)     = default;
  constexpr field(field&&) noexcept = default;
  //============================================================================
  // assign ops
  //============================================================================
  constexpr auto operator=(field const&) -> field& = default;
  constexpr auto operator=(field&&) noexcept -> field& = default;
  //============================================================================
  // dtor
  //============================================================================
  virtual ~field() = default;
  //============================================================================
  // virtual methods
  //============================================================================
#ifdef __cpp_concepts
  [[nodiscard]] constexpr virtual auto evaluate(pos_t const& x,
                                                Real const   t) const
      -> tensor_t = 0;
  [[nodiscard]] constexpr virtual auto in_domain(pos_t const&, Real const) const
      -> bool = 0;
#else
  [[nodiscard]] virtual auto evaluate(pos_t const& x, Real const t) const
      -> tensor_t = 0;
  [[nodiscard]] virtual auto in_domain(pos_t const&, Real const) const
      -> bool = 0;
#endif
  //============================================================================
  // methods
  //============================================================================
  constexpr auto operator()(pos_t const& x, Real const t) const -> tensor_t {
    return evaluate(x, t);
  }
};  // field
template <typename Real, size_t N, size_t R = N, size_t C = N>
using matrixfield = field<Real, N, mat<Real, R, C>>;
template <typename Real, size_t N, size_t C = N>
using vectorfield = field<Real, N, vec<Real, C>>;
template <typename Real, size_t N>
using scalarfield = field<Real, N, Real>;
//==============================================================================
}  // namespace tatooine::polymorphic
//==============================================================================

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N, typename Tensor>
using field_list =
    std::vector<std::unique_ptr<polymorphic::field<Real, N, Tensor>>>;
template <typename Real, size_t N, size_t C = N>
using vectorfield_list = field_list<Real, N, vec<Real, C>>;
//==============================================================================
template <typename DerivedField, typename Real, size_t N, typename Tensor>
struct field : polymorphic::field<Real, N, Tensor> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t   = field<DerivedField, Real, N, Tensor>;
  using parent_t = polymorphic::field<Real, N, Tensor>;
  using pos_t    = typename parent_t::pos_t;
  using tensor_t = typename parent_t::tensor_t;
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived() -> DerivedField& {
    return static_cast<DerivedField&>(*this);
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived() const -> DerivedField const& {
    return static_cast<DerivedField const&>(*this);
  }
  //============================================================================
  // ctors
  //============================================================================
  field()                 = default;
  field(field const&)     = default;
  field(field&&) noexcept = default;
  //============================================================================
  // assign ops
  //============================================================================
  auto operator=(field const&) -> field& = default;
  auto operator=(field&&) noexcept -> field& = default;
  //============================================================================
  // dtor
  //============================================================================
  virtual ~field() = default;
  //============================================================================
  // methods
  //============================================================================
  [[nodiscard]] auto evaluate(pos_t const& x, Real const t) const
      -> tensor_t override {
    return as_derived().evaluate(x, t);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, Real const t) const
      -> bool override {
    return as_derived().in_domain(x, t);
  }
};  // field
//==============================================================================
template <typename V, typename Real, size_t N, size_t R = N, size_t C = N>
using matrixfield = field<V, Real, N, mat<Real, R, C>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t N, size_t C = N>
using vectorfield = field<V, Real, N, vec<Real, N>>;
//------------------------------------------------------------------------------
template <typename V, typename Real, size_t N>
using scalarfield = field<V, Real, N, Real>;
//==============================================================================
// type traits
//==============================================================================
template <typename T, typename = void>
struct is_field : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_field_v = is_field<T>::value;
//------------------------------------------------------------------------------
template <typename T>
struct is_field<T> : std::integral_constant<bool, T::is_field()> {};
//==============================================================================
template <typename T, typename = void>
struct is_scalarfield : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_scalarfield_v = is_scalarfield<T>::value;
//------------------------------------------------------------------------------
template <typename T>
struct is_scalarfield<T> : std::integral_constant<bool, T::is_scalarfield()> {};
//==============================================================================
template <typename T, typename = void>
struct is_vectorfield : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_vectorfield_v = is_vectorfield<T>::value;
//------------------------------------------------------------------------------
template <typename T>
struct is_vectorfield<T> : std::integral_constant<bool, T::is_vectorfield()> {};
//==============================================================================
template <typename T, typename = void>
struct is_matrixfield : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr bool is_matrixfield_v = is_matrixfield<T>::value;
//------------------------------------------------------------------------------
template <typename T>
struct is_matrixfield<T> : std::integral_constant<bool, T::is_matrixfield()> {};
//==============================================================================
// free functions
//==============================================================================
#ifdef __cpp_concpets
template <typename V, arithmetic VReal, size_t N, typename Tensor,
          indexable_space... SpatialDimensions, arithmetc T>
#else
template <typename V, typename VReal, size_t N, typename Tensor,
          typename... SpatialDimensions, typename T,
          enable_if<is_arithmetic<VReal, T>> = true>
#endif
auto sample_to_raw(field<V, VReal, N, Tensor> const& f,
                   grid<SpatialDimensions...> const& discretized_domain,
                   T const t, size_t padding = 0, VReal padval = 0) {
  auto const         nan = VReal(0) / VReal(0);
  std::vector<VReal> raw_data;
  raw_data.reserve(discretized_domain.num_vertices() *
                   (f.num_tensor_components() + padding));
  for (auto x : discretized_domain.vertices()) {
    if (f.in_domain(x, t)) {
      auto sample = f(x, t);
      if constexpr (f.is_scalarfield()) {
        raw_data.push_back(static_cast<VReal>(sample));
      } else {
        for (size_t i = 0; i < f.num_tensor_components(); ++i) {
          raw_data.push_back(static_cast<VReal>(sample[i]));
        }
      }
      for (size_t i = 0; i < padding; ++i) {
        raw_data.push_back(padval);
      }
    } else {
      for (size_t i = 0; i < f.num_tensor_components() + padding; ++i) {
        raw_data.push_back(nan);
      }
    }
  }
  return raw_data;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concpets
template <typename V, arithmetic VReal, arithmetic TReal, size_t N,
          typename Tensor, indexable_space... SpatialDimensions,
          indexable_space TemporalDimension>
#else
template <typename V, typename VReal, typename TReal, size_t N, typename Tensor,
          typename... SpatialDimensions, typename TemporalDimension,
          enable_if<is_arithmetic<VReal, TReal>> = true>
#endif
auto sample_to_raw(field<V, VReal, N, Tensor> const& f,
                   grid<SpatialDimensions...> const& discretized_domain,
                   TemporalDimension const& temporal_domain, size_t padding = 0,
                   VReal padval = 0) {
  auto const         nan = VReal(0) / VReal(0);
  std::vector<VReal> raw_data;
  raw_data.reserve(discretized_domain.num_vertices() * temporal_domain.size() *
                   (f.num_tensor_components() + padding));
  for (auto t : temporal_domain) {
    for (auto v : discretized_domain.vertices()) {
      auto const x = v.position();
      if (f.in_domain(x, t)) {
        auto sample = f(x, t);
        if constexpr (f.is_scalarfield()) {
          raw_data.push_back(static_cast<VReal>(sample));
        } else {
          for (size_t i = 0; i < f.num_tensor_components(); ++i) {
            raw_data.push_back(static_cast<VReal>(sample[i]));
          }
        }
        for (size_t i = 0; i < padding; ++i) {
          raw_data.push_back(padval);
        }
      } else {
        for (size_t i = 0; i < f.num_tensor_components() + padding; ++i) {
          raw_data.push_back(nan);
        }
      }
    }
  }
  return raw_data;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concpets
template <arithmetic VReal, arithmetic TReal, size_t N, typename Tensor,
          indexable_space... SpatialDimensions>
#else
template <typename VReal, typename TReal, size_t N, typename Tensor,
          typename... SpatialDimensions,
          enable_if<is_arithmetic<VReal, TReal>> = true>
#endif
auto sample_to_vector(polymorphic::field<VReal, N, Tensor> const& f,
                      grid<SpatialDimensions...> const& discretized_domain,
                      TReal                             t) {
  using V                   = polymorphic::field<VReal, N, Tensor>;
  using tensor_t            = typename V::tensor_t;
  auto const            nan = VReal(0) / VReal(0);
  std::vector<tensor_t> data;
  data.reserve(discretized_domain.num_vertices());
  for (auto x : discretized_domain.vertices()) {
    if (f.in_domain(x, t)) {
      data.push_back(f(x, t));
    } else {
      if constexpr (rank<tensor_t>() == 0) {
        data.push_back(nan);
      } else {
        data.push_back(tensor_t{tag::fill{nan}});
      }
    }
  }
  return data;
}
//------------------------------------------------------------------------------
#ifdef __cpp_concpets
template <typename V, arithmetic VReal, arithmetic TReal size_t N,
          typename Tensor, indexable_space... SpatialDimensions>
#else
template <typename V, typename VReal, typename TReal, size_t N, typename Tensor,
          typename... SpatialDimensions,
          enable_if<is_arithmetic<VReal, TReal>> = true>
#endif
auto discretize(field<V, VReal, N, Tensor> const& f,
                grid<SpatialDimensions...>&       discretized_domain,
                std::string const& property_name, TReal const t) -> auto& {
  auto const ood_tensor = [&f] {
    if constexpr (is_scalarfield_v<V>) {
      return VReal(0) / VReal(0);
    } else {
      return Tensor{tag::fill{VReal(0) / VReal(0)}};
    }
  }();
  auto& discretized_field = [&]() -> decltype(auto) {
    if constexpr (is_scalarfield_v<V>) {
      return discretized_domain.first.template add_vertex_property<VReal>(
          property_name);
    } else if constexpr (is_vectorfield_v<V>) {
      return discretized_domain
          .template add_vertex_property<vec<VReal, V::tensor_t::dimension(0)>>(
              property_name);
    } else if constexpr (is_matrixfield_v<V>) {
      return discretized_domain.template add_vertex_property<
          mat<VReal, V::tensor_t::dimension(0), V::tensor_t::dimension(1)>>(
          property_name);
    } else {
      return discretized_domain.template add_vertex_property<Tensor>(
          property_name);
    }
  }();
  discretized_domain.loop_over_vertex_indices([&](auto const... is) {
    auto const x = discretized_domain.vertex_at(is...);
    if (f.in_domain(x, t)) {
      discretized_field(is...) = f(x, t);
    } else {
      discretized_field(is...) = ood_tensor;
    }
  });
  return discretized_field;
}
//------------------------------------------------------------------------------
//#ifdef __cpp_concpets
// template <typename V, arithmetic VReal, arithmetic TReal, size_t N,
//          typename Tensor, indexable_space... SpatialDimensions,
//          indexable_space TemporalDomain>
//#else
// template <typename V, typename VReal, typename TReal, size_t N,
//          typename Tensor, typename... SpatialDimensions,
//          typename TemporalDomain,
//          enable_if<is_arithmetic<VReal, TReal>> = true>
//#endif
// auto resample(field<V, VReal, N, Tensor> const& f,
//              grid<SpatialDimensions...>&              discretized_domain,
//              TemporalDomain const& temporal_domain) -> auto& {
//  auto const ood_tensor = [] {
//    if constexpr (f.is_scalarfield()) {
//      return VReal(0) / VReal(0);
//    } else {
//      return Tensor{tag::fill{VReal(0) / VReal(0)}};
//    }
//  }();
//  auto& discretized_field = [&]() -> decltype(auto) {
//    if constexpr (f.is_scalarfield()) {
//      return gn.first.template
//      add_contiguous_vertex_property<VReal>(gn.second);
//    } else if constexpr (f.is_vectorfield()) {
//      return gn.first
//          .template add_contiguous_vertex_property<vec<VReal,
//          f.dimension(0)>>(
//              gn.second);
//    } else if constexpr (f.is_matrixfield()) {
//      return gn.first
//          .template add_contiguous_vertex_property<mat<VReal, f.dimension(0),
//          f.dimension(1)>>(
//              gn.second);
//    } else {
//      return gn.first.template add_contiguous_vertex_property<
//          Tensor>(gn.second);
//    }
//  }();
//  for (auto const t : temporal_domain) {
//    discretized_domain.loop_over_vertex_indices([&](auto const... is) {
//      auto const x = discretized_domain.vertex_at(is...);
//      if (f.in_domain(x, t)) {
//        discretized_field.data_at(is...) = f(x, t);
//      } else {
//        discretized_field.data_at(is...) = ood_tensor;
//      }
//    });
//  }
//  return discretized_field;
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field_operations.h>
#include <tatooine/field_type_traits.h>
#endif
