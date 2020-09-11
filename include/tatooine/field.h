#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H
//╔════════════════════════════════════════════════════════════════════════════╗
#include <tatooine/crtp.h>
#include <tatooine/grid.h>
#include <tatooine/tensor.h>
#include <tatooine/tensor_type.h>
#include <tatooine/type_traits.h>

#include <vector>
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine::parent {
//╒══════════════════════════════════════════════════════════════════════════╕
template <typename Real, size_t N, size_t... TensorDims>
struct field {
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ typedefs                                                             │
  //├──────────────────────────────────────────────────────────────────────┤
  using real_t   = Real;
  using this_t   = field<Real, N, TensorDims...>;
  using pos_t    = vec<Real, N>;
  using time_t   = Real;
  using tensor_t = std::conditional_t<sizeof...(TensorDims) == 0, Real,
                                      tensor_type<Real, TensorDims...>>;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ static methods                                                       │
  //├──────────────────────────────────────────────────────────────────────┤
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  static constexpr auto num_dimensions() { return N; }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  static constexpr auto num_tensor_dimensions() {
    return sizeof...(TensorDims);
  }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  template <size_t _num_tensor_dims = sizeof...(TensorDims),
            std::enable_if_t<(_num_tensor_dims > 0)>...>
  static constexpr auto tensor_dimension(size_t i) {
    return tensor_t::dimension(i);
  }
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
  field()                 = default;
  field(field const&)     = default;
  field(field&&) noexcept = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  auto operator=(field const&) -> field& = default;
  auto operator=(field&&) noexcept -> field& = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  virtual ~field() = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ virtual methods                                                      │
  //├──────────────────────────────────────────────────────────────────────┤
  [[nodiscard]] virtual auto evaluate(pos_t const& x, Real t = 0) const
      -> tensor_t                                                        = 0;
  [[nodiscard]] virtual auto in_domain(pos_t const&, Real) const -> bool = 0;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ methods                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  auto operator()(pos_t const& x, Real t) const -> tensor_t {
    return evaluate(x, t);
  }
};  // field
template <typename Real, size_t N, size_t C = N>
using vectorfield = field<Real, N, C>;
//╘══════════════════════════════════════════════════════════════════════════╛
}  // namespace tatooine::parent
//╚════════════════════════════════════════════════════════════════════════════╝
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine {
template <typename Real, size_t N, size_t... TensorDims>
using field_list =
    std::vector<std::unique_ptr<parent::field<Real, N, TensorDims...>>>;
template <typename Real, size_t N, size_t D = N>
using vectorfield_list = field_list<Real, N, D>;
template <typename Derived, typename Real, size_t N, size_t... TensorDims>
//╒══════════════════════════════════════════════════════════════════════════╕
struct field : parent::field<Real, N, TensorDims...>, crtp<Derived> {
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ typedefs                                                             │
  //├──────────────────────────────────────────────────────────────────────┤
  using this_t        = field<Derived, Real, N, TensorDims...>;
  using parent_crtp_t = crtp<Derived>;
  using parent_t      = parent::field<Real, N, TensorDims...>;
  using pos_t         = typename parent_t::pos_t;
  using tensor_t      = typename parent_t::tensor_t;
  using parent_crtp_t::as_derived;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
  field()                 = default;
  field(field const&)     = default;
  field(field&&) noexcept = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  auto operator=(field const&) -> field& = default;
  auto operator=(field&&) noexcept -> field& = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ dtor                                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
  ~field() override = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ methods                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  [[nodiscard]] auto evaluate(pos_t const& x, Real t) const
      -> tensor_t override {
    return as_derived().evaluate(x, t);
  }
  //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
  [[nodiscard]] auto in_domain(pos_t const& x, Real t) const -> bool override {
    return as_derived().in_domain(x, t);
  }
};  // field
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename V, typename Real, size_t N, size_t C = N>
using vectorfield = field<V, Real, N, C>;
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename V, typename Real, size_t N>
using scalarfield = field<V, Real, N>;
//╘══════════════════════════════════════════════════════════════════════════╛
//╒══════════════════════════════════════════════════════════════════════════╕
//│ type traits                                                              │
//╞══════════════════════════════════════════════════════════════════════════╡
template <typename T>
struct is_field : std::false_type {};
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename T>
static constexpr bool is_field_v = is_field<T>::value;
template <typename Real, size_t N, size_t... TensorDims>
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
struct is_field<parent::field<Real, N, TensorDims...>> : std::true_type {};
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
template <typename Derived, typename Real, size_t N, size_t... TensorDims>
struct is_field<field<Derived, Real, N, TensorDims...>> : std::true_type {};
////╒══════════════════════════════════════════════════════════════════════════╕
////│ free functions                                                           │
////╞══════════════════════════════════════════════════════════════════════════╡
template <real_number OutReal, typename V, real_number VReal, size_t N,
          size_t... TensorDims, indexable_space... SpatialDimensions>
auto sample_to_raw(field<V, VReal, N, TensorDims...> const& f,
                   grid<SpatialDimensions...> const& g, real_number auto t,
                   size_t padding = 0, VReal padval = 0) {
  auto const           nan = OutReal(0) / OutReal(0);
  std::vector<OutReal> raw_data;
  auto const           num_comps = std::max<size_t>(1, (TensorDims * ...));
  raw_data.reserve(g.num_vertices() * (num_comps + padding));
  for (auto v : g.vertices()) {
    auto const x = v.position();
    if (f.in_domain(x, t)) {
      auto sample = f(x, t);
      if constexpr (sizeof...(TensorDims) == 0) {
        raw_data.push_back(static_cast<OutReal>(sample));
      } else {
        for (size_t i = 0; i < num_comps; ++i) {
          raw_data.push_back(static_cast<OutReal>(sample[i]));
        }
      }
      for (size_t i = 0; i < padding; ++i) { raw_data.push_back(padval); }
    } else {
      for (size_t i = 0; i < num_comps + padding; ++i) {
        raw_data.push_back(nan);
      }
    }
  }
  return raw_data;
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename OutReal, typename V, real_number VReal, real_number TReal,
          size_t N, size_t... TensorDims, indexable_space... SpatialDimensions,
          indexable_space TemporalDimension>
auto sample_to_raw(field<V, VReal, N, TensorDims...> const& f,
                   grid<SpatialDimensions...> const&        g,
                   TemporalDimension const& temporal_domain, size_t padding = 0,
                   OutReal padval = 0) {
  auto const           nan = OutReal(0) / OutReal(0);
  std::vector<OutReal> raw_data;
  auto const           num_comps = std::max<size_t>(1, (TensorDims * ...));
  raw_data.reserve(g.num_vertices() * temporal_domain.size() *
                   (num_comps + padding));
  for (auto t : temporal_domain) {
    for (auto v : g.vertices()) {
      auto const x = v.position();
      if (f.in_domain(x, t)) {
        auto sample = f(x, t);
        if constexpr (sizeof...(TensorDims) == 0) {
          raw_data.push_back(static_cast<OutReal>(sample));
        } else {
          for (size_t i = 0; i < num_comps; ++i) {
            raw_data.push_back(static_cast<OutReal>(sample[i]));
          }
        }
        for (size_t i = 0; i < padding; ++i) { raw_data.push_back(padval); }
      } else {
        for (size_t i = 0; i < num_comps + padding; ++i) {
          raw_data.push_back(nan);
        }
      }
    }
  }
  return raw_data;
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename OutReal, typename V, real_number VReal, size_t N,
          size_t... TensorDims, indexable_space... SpatialDimensions>
auto resample(field<V, VReal, N, TensorDims...> const& f,
              grid<SpatialDimensions...> const&        spatial_domain,
              real_number auto const                   t) {
  auto const ood_tensor = [] {
    if constexpr (sizeof...(TensorDims) == 0) {
      return OutReal(0) / OutReal(0);
    } else {
      return tensor<OutReal, TensorDims...>{tag::fill{OutReal(0) / OutReal(0)}};
    }
  }();
  std::pair gn{spatial_domain.copy_without_properties(), "resampled"};
  auto&     prop = [&]() -> decltype(auto) {
    if constexpr (sizeof...(TensorDims) == 0) {
      return gn.first.template add_chunked_vertex_property<OutReal>(
          gn.second, std::vector<size_t>(N, 10));
    } else if constexpr (sizeof...(TensorDims) == 1) {
      return gn.first
          .template add_chunked_vertex_property<vec<OutReal, TensorDims...>>(
              gn.second, std::vector<size_t>(N, 10));
    } else if constexpr (sizeof...(TensorDims) == 2) {
      return gn.first
          .template add_chunked_vertex_property<mat<OutReal, TensorDims...>>(
              gn.second, std::vector<size_t>(N, 10));
    } else {
      return gn.first
          .template add_chunked_vertex_property<tensor<OutReal, TensorDims...>>(
              gn.second, std::vector<size_t>(N, 10));
    }
  }();
  spatial_domain.loop_over_vertex_indices([&](auto const... is) {
    auto const x = spatial_domain.vertex_at(is...);
    if (f.in_domain(x, t)) {
      prop.data_at(is...) = f(x, t);
    } else {
      prop.data_at(is...) = ood_tensor;
    }
  });
  return gn;
}
//├──────────────────────────────────────────────────────────────────────────┤
template <typename OutReal, typename V, real_number VReal, real_number TReal,
          size_t N, size_t... TensorDims, indexable_space... SpatialDimensions,
          indexable_space TemporalDomain>
auto resample(field<V, VReal, N, TensorDims...> const& f,
              grid<SpatialDimensions...> const&        spatial_domain,
              TemporalDomain const&                    temporal_domain) {
  auto const ood_tensor = [] {
    if constexpr (sizeof...(TensorDims) == 0) {
      return OutReal(0) / OutReal(0);
    } else {
      return tensor<OutReal, TensorDims...>{tag::fill{OutReal(0) / OutReal(0)}};
    }
  }();
  std::pair gn{spatial_domain + temporal_domain, "resampled"};
  auto&     prop = [&]() -> decltype(auto) {
    if constexpr (sizeof...(TensorDims) == 0) {
      return gn.first.template add_chunked_vertex_property<OutReal>(
          gn.second, std::vector<size_t>(N, 10));
    } else if constexpr (sizeof...(TensorDims) == 1) {
      return gn.first
          .template add_chunked_vertex_property<vec<OutReal, TensorDims...>>(
              gn.second, std::vector<size_t>(N, 10));
    } else if constexpr (sizeof...(TensorDims) == 2) {
      return gn.first
          .template add_chunked_vertex_property<mat<OutReal, TensorDims...>>(
              gn.second, std::vector<size_t>(N, 10));
    } else {
      return gn.first
          .template add_chunked_vertex_property<tensor<OutReal, TensorDims...>>(
              gn.second, std::vector<size_t>(N, 10));
    }
  }();
  for (auto const t : temporal_domain) {
    spatial_domain.loop_over_vertex_indices([&](auto const... is) {
      auto const x = spatial_domain.vertex_at(is...);
      if (f.in_domain(x, t)) {
        prop.data_at(is...) = f(x, t);
      } else {
        prop.data_at(is...) = ood_tensor;
      }
    });
  }
  return gn;
}
}  // namespace tatooine
//╚════════════════════════════════════════════════════════════════════════════╝
#include "differentiated_field.h"
#include "field_operations.h"
#endif
