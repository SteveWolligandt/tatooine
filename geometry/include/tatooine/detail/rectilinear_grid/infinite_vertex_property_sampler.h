#include <tatooine/detail/rectilinear_grid/vertex_property_sampler.h>
#include <tatooine/field.h>
#include <tatooine/variadic_helpers.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point_range... Dimensions>
requires(sizeof...(Dimensions) > 1) class rectilinear_grid;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename VertexPropSampler, std::size_t... RepeatedDims>
struct infinite_vertex_property_sampler
    : tatooine::field<
          infinite_vertex_property_sampler<VertexPropSampler, RepeatedDims...>,
          typename VertexPropSampler::real_type,
          VertexPropSampler::num_dimensions(),
          typename VertexPropSampler::tensor_type> {
  VertexPropSampler const& m_sampler;
  infinite_vertex_property_sampler(VertexPropSampler const& sampler)
      : m_sampler{sampler} {}

  using parent_type = tatooine::field<infinite_vertex_property_sampler,
                                      typename VertexPropSampler::real_type,
                                      VertexPropSampler::num_dimensions(),
                                      typename VertexPropSampler::tensor_type>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;

 private:
  static constexpr auto non_repeated_dimensions__() {
    auto constexpr rs = repeated_dimensions;
    auto non          = std::array<std::size_t, num_non_repeated_dimensions>{};
    auto idx          = std::size_t(0);
    for (std::size_t i = 0; i < parent_type::num_dimensions(); ++i) {
      bool b = true;
      for (auto r : rs) {
        if (r == i) {
          b = false;
          break;
        }
      }
      if (b) {
        non[idx++] = i;
      }
    }

    return non;
  }

 public:
  static auto constexpr num_non_repeated_dimensions =
      parent_type::num_dimensions() - sizeof...(RepeatedDims);
  static auto constexpr num_repeated_dimensions =
      parent_type::num_dimensions() - num_non_repeated_dimensions;
  static constexpr auto repeated_dimensions     = std::array{RepeatedDims...};
  static constexpr auto non_repeated_dimensions = non_repeated_dimensions__();
  template <std::size_t... i>
  auto clamp_pos(pos_type x, std::index_sequence<i...>) const {
    (
        [&] {
          auto constexpr dim = repeated_dimensions[i];
          auto const front = m_sampler.grid().template dimension<dim>().front();
          auto const back  = m_sampler.grid().template dimension<dim>().back();
          auto const extent = back - front;
          if (x(dim) < front) {
            x(dim) += gcem::ceil((front - x(dim)) / extent) * extent;
          }
          if (x(dim) > back) {
            x(dim) -= extent;
            x(dim) -= gcem::ceil((x(dim) - back) / extent) * extent;
          }
        }(),
        ...);
    return x;
  }
  auto clamp_pos(pos_type const& x) const {
    return clamp_pos(x, std::make_index_sequence<num_repeated_dimensions>{});
  }
  [[nodiscard]] auto evaluate(pos_type const& x, real_type const t) const
      -> tensor_type {
    if (is_inside(x)) {
      return m_sampler(clamp_pos(x), t);
    }
    return parent_type::ood_tensor();
  }
  //----------------------------------------------------------------------------
  template <std::size_t... i>
  auto constexpr is_inside(pos_type const& x, std::index_sequence<i...>) const
      -> bool {
    return ([&] {
      auto constexpr dim = non_repeated_dimensions[i];
      auto const front   = m_sampler.grid().template dimension<dim>().front();
      auto const back    = m_sampler.grid().template dimension<dim>().back();
      return front <= x(dim) && x(dim) <= back;
    }() && ...);
  }
  auto constexpr is_inside(pos_type const& x) const -> bool {
    return is_inside(x,
                     std::make_index_sequence<num_non_repeated_dimensions>{});
  }
};
//------------------------------------------------------------------------------
template <std::size_t... RepeatedDims, typename GridVertexProperty>
auto make_infinite(
    vertex_property_sampler<GridVertexProperty, interpolation::linear,
                            interpolation::linear> const& v) {
  return infinite_vertex_property_sampler<
      vertex_property_sampler<GridVertexProperty, interpolation::linear,
                              interpolation::linear>,
      RepeatedDims...>{v};
}
//------------------------------------------------------------------------------
template <std::size_t... RepeatedDims, typename GridVertexProperty>
auto make_infinite(
    vertex_property_sampler<GridVertexProperty, interpolation::linear,
                            interpolation::linear, interpolation::linear> const&
        v) {
  return infinite_vertex_property_sampler<
      vertex_property_sampler<GridVertexProperty, interpolation::linear,
                              interpolation::linear, interpolation::linear>,
      RepeatedDims...>{v};
}
//------------------------------------------------------------------------------
template <std::size_t... DimsToRepeat, typename Dim0, typename Dim1,
          typename ValueType>
auto repeat_for_infinite(
    typed_vertex_property_interface<tatooine::rectilinear_grid<Dim0, Dim1>,
                                    ValueType, true>& prop) {
  auto const s = prop.grid().size();
  // borders
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                variadic::contains<0, DimsToRepeat...>) {
    for (std::size_t i = 0;
         i < (variadic::contains<1, DimsToRepeat...> ? s[1] - 1 : s[1]); ++i) {
      prop(s[0] - 1, i) = prop(0, i);
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                variadic::contains<1, DimsToRepeat...>) {
    for (std::size_t i = 0;
         i < (variadic::contains<1, DimsToRepeat...> ? s[0] - 1 : s[0]); ++i) {
      prop(i, s[1] - 1) = prop(i, 0);
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<0, DimsToRepeat...> &&
                 variadic::contains<1, DimsToRepeat...>)) {
    // corner
    prop(s[0] - 1, s[1] - 1) = prop(0, 0);
  }
}
//------------------------------------------------------------------------------
template <std::size_t... DimsToRepeat, typename Dim0, typename Dim1,
          typename Dim2, typename ValueType>
auto repeat_for_infinite(
    typed_vertex_property_interface<
        tatooine::rectilinear_grid<Dim0, Dim1, Dim2>, ValueType, true>& prop) {
  auto const s = prop.grid().size();
  // planes
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                variadic::contains<0, DimsToRepeat...>) {
    for (std::size_t i = 0;
         i < (variadic::contains<1, DimsToRepeat...> ? s[1] - 1 : s[1]); ++i) {
      for (std::size_t j = 0;
           j < (variadic::contains<2, DimsToRepeat...> ? s[2] - 1 : s[2]);
           ++j) {
        prop(s[0] - 1, i, j) = prop(0, i, j);
      }
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                variadic::contains<1, DimsToRepeat...>) {
    for (std::size_t i = 0;
         i < (variadic::contains<0, DimsToRepeat...> ? s[0] - 1 : s[0]); ++i) {
      for (std::size_t j = 0;
           j < (variadic::contains<2, DimsToRepeat...> ? s[2] - 1 : s[2]);
           ++j) {
        prop(i, s[1] - 1, j) = prop(i, 0, j);
      }
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                variadic::contains<2, DimsToRepeat...>) {
    for (std::size_t i = 0;
         i < (variadic::contains<0, DimsToRepeat...> ? s[0] - 1 : s[0]); ++i) {
      for (std::size_t j = 0;
           j < (variadic::contains<1, DimsToRepeat...> ? s[1] - 1 : s[1]);
           ++j) {
        prop(i, j, s[2] - 1) = prop(i, j, 0);
      }
    }
  }
  // edges
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<1, DimsToRepeat...> &&
                 variadic::contains<2, DimsToRepeat...>)) {
    for (std::size_t i = 0;
         i < (variadic::contains<0, DimsToRepeat...> ? s[0] - 1 : s[0]); ++i) {
      prop(i, s[1] - 1, s[2] - 1) = prop(i, 0, 0);
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<0, DimsToRepeat...> &&
                 variadic::contains<2, DimsToRepeat...>)) {
    for (std::size_t i = 0;
         i < (variadic::contains<1, DimsToRepeat...> ? s[1] - 1 : s[1]); ++i) {
      prop(s[0] - 1, i, s[2] - 1) = prop(0, i, 0);
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<0, DimsToRepeat...> &&
                 variadic::contains<1, DimsToRepeat...>)) {
    for (std::size_t i = 0;
         i < (variadic::contains<2, DimsToRepeat...> ? s[2] - 1 : s[2]); ++i) {
      prop(s[0] - 1, s[1] - 1, i) = prop(0, 0, i);
    }
  }
  // corner
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<0, DimsToRepeat...> &&
                 variadic::contains<1, DimsToRepeat...> &&
                 variadic::contains<2, DimsToRepeat...>)) {
    prop(s[0] - 1, s[1] - 1, s[2] - 1) = prop(0, 0, 0);
  }
}
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
