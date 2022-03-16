#ifndef TATOOINE_DISCRETIZE_FIELD_H
#define TATOOINE_DISCRETIZE_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/rectilinear_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <
    typename V, arithmetic VReal, std::size_t NumDimensions, typename Tensor,
    detail::rectilinear_grid::dimension... SpatialDimensions, arithmetic T>
auto sample_to_raw(
    field<V, VReal, NumDimensions, Tensor> const& f,
    rectilinear_grid<SpatialDimensions...> const& discretized_domain, T const t,
    std::size_t padding = 0, VReal padval = 0) {
  auto const         nan = VReal(0) / VReal(0);
  std::vector<VReal> raw_data;
  auto               vs = discretized_domain.vertices();
  raw_data.reserve(vs.size() * (f.num_tensor_components() + padding));
  for (auto v : vs) {
    auto const x      = vs[v];
    auto       sample = f(x, t);
    if constexpr (f.is_scalarfield()) {
      raw_data.push_back(static_cast<VReal>(sample));
    } else {
      for (std::size_t i = 0; i < f.num_tensor_components(); ++i) {
        raw_data.push_back(static_cast<VReal>(sample[i]));
      }
    }
    for (std::size_t i = 0; i < padding; ++i) {
      raw_data.push_back(padval);
    }
  }
  return raw_data;
}
//------------------------------------------------------------------------------
template <typename V, arithmetic VReal, arithmetic TReal,
          std::size_t NumDimensions, typename Tensor,
          detail::rectilinear_grid::dimension... SpatialDimensions,
          detail::rectilinear_grid::dimension TemporalDimension>
auto sample_to_raw(
    field<V, VReal, NumDimensions, Tensor> const& f,
    rectilinear_grid<SpatialDimensions...> const& discretized_domain,
    TemporalDimension const& temporal_domain, std::size_t padding = 0,
    VReal padval = 0) {
  auto const         nan = VReal(0) / VReal(0);
  std::vector<VReal> raw_data;
  auto               vs = discretized_domain.vertices();
  raw_data.reserve(vs.size() * temporal_domain.size() *
                   (f.num_tensor_components() + padding));
  for (auto t : temporal_domain) {
    for (auto v : vs) {
      auto const x      = v.position();
      auto       sample = f(x, t);
      if constexpr (f.is_scalarfield()) {
        raw_data.push_back(static_cast<VReal>(sample));
      } else {
        for (std::size_t i = 0; i < f.num_tensor_components(); ++i) {
          raw_data.push_back(static_cast<VReal>(sample[i]));
        }
      }
      for (std::size_t i = 0; i < padding; ++i) {
        raw_data.push_back(padval);
      }
    }
  }
  return raw_data;
}
//------------------------------------------------------------------------------
template <arithmetic VReal, arithmetic TReal, std::size_t NumDimensions,
          typename Tensor,
          detail::rectilinear_grid::dimension... SpatialDimensions>
auto sample_to_vector(
    polymorphic::field<VReal, NumDimensions, Tensor> const& f,
    rectilinear_grid<SpatialDimensions...> const& discretized_domain, TReal t) {
  using V           = polymorphic::field<VReal, NumDimensions, Tensor>;
  using tensor_type = typename V::tensor_type;
  auto const               nan = VReal(0) / VReal(0);
  std::vector<tensor_type> data;
  auto                     vs = discretized_domain.vertices();
  data.reserve(vs.size());
  for (auto v : vs) {
    auto const x = vs[v];
    data.push_back(f(x, t));
  }
  return data;
}
//------------------------------------------------------------------------------
template <arithmetic VReal, arithmetic TReal, std::size_t NumDimensions,
          typename Tensor,
          detail::rectilinear_grid::dimension... SpatialDimensions,
          typename ExecutionPolicy>
auto discretize(polymorphic::field<VReal, NumDimensions, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name, TReal const t,
                ExecutionPolicy const execution_policy) -> auto& {
  auto& discretized_field = [&]() -> decltype(auto) {
    if constexpr (is_arithmetic<Tensor>) {
      return discretized_domain.template insert_vertex_property<VReal>(
          property_name);
    } else if constexpr (static_vec<Tensor>) {
      return discretized_domain
          .template vertex_property<vec<VReal, Tensor::dimension(0)>>(
              property_name);
    } else if constexpr (static_mat<Tensor>) {
      return discretized_domain.template vertex_property<
          mat<VReal, Tensor::dimension(0), Tensor::dimension(1)>>(
          property_name);
    } else {
      return discretized_domain.template vertex_property<Tensor>(property_name);
    }
  }();
  discretized_domain.vertices().iterate_indices(
      [&](auto const... is) {
        auto const x             = discretized_domain.vertex_at(is...);
        discretized_field(is...) = f(x, t);
      },
      execution_policy);
  return discretized_field;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic VReal, arithmetic TReal, std::size_t NumDimensions,
          typename Tensor,
          detail::rectilinear_grid::dimension... SpatialDimensions>
auto discretize(polymorphic::field<VReal, NumDimensions, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name, TReal const t) -> auto& {
  return discretize(f, discretized_domain, property_name, t,
                    execution_policy::sequential);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic VReal, std::size_t NumDimensions, typename Tensor,
          detail::rectilinear_grid::dimension... SpatialDimensions>
auto discretize(polymorphic::field<VReal, NumDimensions, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const& property_name) -> auto& {
  return discretize(f, discretized_domain, property_name, 0,
                    execution_policy::sequential);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic VReal, std::size_t NumDimensions, typename Tensor,
          typename ExecutionPolicy,
          detail::rectilinear_grid::dimension... SpatialDimensions>
auto discretize(polymorphic::field<VReal, NumDimensions, Tensor> const& f,
                rectilinear_grid<SpatialDimensions...>& discretized_domain,
                std::string const&                      property_name,
                ExecutionPolicy const execution_policy) -> auto& {
  return discretize(f, discretized_domain, property_name, 0, execution_policy);
}
//------------------------------------------------------------------------------
/// Discretizes to a cutting plane of a field.
/// \param basis spatial basis of cutting plane
template <typename V, arithmetic VReal, arithmetic TReal,
          std::size_t NumDimensions, typename Tensor, typename BasisReal,
          typename X0Real, typename X1Real>
auto discretize(field<V, VReal, NumDimensions, Tensor> const& f,
                vec<X0Real, NumDimensions> const&             x0,
                mat<BasisReal, NumDimensions, 2> const&       basis,
                vec<X1Real, 2> const& spatial_size, std::size_t const res0,
                std::size_t const res1, std::string const& property_name,
                TReal const t) {
  auto const cell_extent =
      vec<VReal, 2>{spatial_size(0) / (res0 - 1), spatial_size(1) / (res1 - 1)};
  std::cerr << x0 << '\n';
  std::cerr << spatial_size << '\n';
  std::cerr << cell_extent << '\n';
  auto discretized_domain = rectilinear_grid{
      linspace<VReal>{0, length(basis * vec{spatial_size(0), 0}), res0},
      linspace<VReal>{0, length(basis * vec{0, spatial_size(1)}), res1}};

  auto& discretized_field = [&]() -> decltype(auto) {
    if constexpr (is_scalarfield<V>()) {
      return discretized_domain.template insert_chunked_vertex_property<VReal>(
          property_name);
    } else if constexpr (is_vectorfield<V>()) {
      return discretized_domain
          .template vertex_property<vec<VReal, V::tensor_type::dimension(0)>>(
              property_name);
    } else if constexpr (is_matrixfield<V>()) {
      return discretized_domain.template vertex_property<mat<
          VReal, V::tensor_type::dimension(0), V::tensor_type::dimension(1)>>(
          property_name);
    } else {
      return discretized_domain.template vertex_property<Tensor>(property_name);
    }
  }();
  for (std::size_t i1 = 0; i1 < res1; ++i1) {
    for (std::size_t i0 = 0; i0 < res0; ++i0) {
      auto const x = x0 + basis * vec{cell_extent(0) * i0, cell_extent(1) * i1};
      discretized_field(i0, i1) = f(x, t);
    }
  }
  return discretized_domain;
}
//------------------------------------------------------------------------------
/// Discretizes to a cutting plane of a field.
/// \param basis spatial basis of cutting plane
template <typename V, arithmetic VReal, arithmetic TReal,
          std::size_t NumDimensions, typename Tensor, typename BasisReal,
          typename X0Real>
auto discretize(field<V, VReal, NumDimensions, Tensor> const& f,
                mat<BasisReal, NumDimensions, 2> const&       basis,
                vec<X0Real, NumDimensions> const& x0, std::size_t const res0,
                std::size_t const res1, std::string const& property_name,
                TReal const t) {
  return discretize(f, x0, basis, vec<BasisReal, 2>::ones(), res0, res1,
                    property_name, t);
}
//------------------------------------------------------------------------------
/// Discretizes to a cutting plane of a field.
/// \param basis spatial basis of cutting plane
template <typename V, arithmetic VReal, arithmetic TReal,
          std::size_t NumDimensions, typename Tensor, typename BasisReal,
          typename X0Real>
auto discretize(field<V, VReal, NumDimensions, Tensor> const& f,
                vec<BasisReal, NumDimensions> const&          extent0,
                vec<BasisReal, NumDimensions> const&          extent1,
                vec<X0Real, NumDimensions> const& /*x0*/,
                std::size_t const res0, std::size_t const res1,
                std::string const& property_name, TReal const t) {
  auto basis   = mat<BasisReal, NumDimensions, 2>{};
  basis.col(0) = extent0;
  basis.col(1) = extent1;
  return discretize(f, basis, res0, res1, property_name, t);
}
#endif
