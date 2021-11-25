#include <tatooine/field.h>
#include <tatooine/variadic_helpers.h>
#include <tatooine/rectilinear_grid_vertex_property_sampler.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename VertexPropSampler, std::size_t... RepeatedDims>
struct infinite_rectilinear_grid_vertex_property_sampler
    : field<infinite_rectilinear_grid_vertex_property_sampler<VertexPropSampler, RepeatedDims...>,
            typename VertexPropSampler::real_t,
            VertexPropSampler::num_dimensions(),
            typename VertexPropSampler::tensor_t> {
  VertexPropSampler const& m_sampler;
  infinite_rectilinear_grid_vertex_property_sampler(VertexPropSampler const& sampler) : m_sampler{sampler} {}
  using parent_t = field<infinite_rectilinear_grid_vertex_property_sampler, typename VertexPropSampler::real_t,
                         VertexPropSampler::num_dimensions(),
                         typename VertexPropSampler::tensor_t>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;

 private:
  static constexpr auto non_repeated_dimensions__() {
    auto constexpr num_non_repeated =
        parent_t::num_dimensions() - sizeof...(RepeatedDims);
    auto constexpr rs = repeated_dimensions;
    auto non          = std::array<std::size_t, num_non_repeated>{};
    auto idx          = std::size_t(0);
    for (std::size_t i = 0; i < parent_t::num_dimensions(); ++i) {
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
  static constexpr auto repeated_dimensions     = std::array{RepeatedDims...};
  static constexpr auto non_repeated_dimensions = non_repeated_dimensions__();
  template <std::size_t... i>
  auto clamp_pos(pos_t x, std::index_sequence<i...>) const {
    (
        [&] {
          auto const front  = m_sampler.grid().template front<i>();
          auto const back   = m_sampler.grid().template back<i>();
          auto const extent = back - front;
          while (x(i) < front) {
            x(i) += extent;
          }
          while (x(i) > back) {
            x(i) -= extent;
          }
        }(),
        ...);
    return x;
  }
  auto clamp_pos(pos_t const& x) const {
    return clamp_pos(x, std::make_index_sequence<2>{});
  }
  [[nodiscard]] auto evaluate(pos_t const& x, real_t const t) const
      -> tensor_t {
    if (!is_inside(x)) {
      return parent_t::ood_tensor();
    }
    return m_sampler(clamp_pos(x), t);
  }
  //----------------------------------------------------------------------------
  template <std::size_t... i>
  auto constexpr is_inside(pos_t const& x, std::index_sequence<i...>) const
      -> bool {
    bool inside = true;
    (
        [&] {
          auto constexpr dim = non_repeated_dimensions[i];
          auto const front   = m_sampler.grid().template front<dim>();
          auto const back    = m_sampler.grid().template back<dim>();
          if (x(dim) < front) {
            inside = false;
          }
          if (x(dim) > back) {
            inside = false;
          }
        }(),
        ...);
    return inside;
  }
  auto constexpr is_inside(pos_t const& x) const -> bool {
    return is_inside(
        x, std::make_index_sequence<non_repeated_dimensions.size()>{});
  }
};
//------------------------------------------------------------------------------
template <std::size_t... RepeatedDims, typename GridVertexProperty>
auto make_infinite(rectilinear_grid_vertex_property_sampler<
                   GridVertexProperty, interpolation::linear,
                   interpolation::linear> const& v) {
  return infinite_rectilinear_grid_vertex_property_sampler<
      rectilinear_grid_vertex_property_sampler<
          GridVertexProperty, interpolation::linear, interpolation::linear>,
      RepeatedDims...>{v};
}
//------------------------------------------------------------------------------
template <std::size_t... DimsToRepeat, typename Dim0, typename Dim1,
          typename ValueType>
auto repeat_for_infinite(typed_rectilinear_grid_vertex_property_interface<
                         rectilinear_grid<Dim0, Dim1>, ValueType, true>& prop) {
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
    typed_rectilinear_grid_vertex_property_interface<
        rectilinear_grid<Dim0, Dim1, Dim2>, ValueType, true>& prop) {
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
        prop(i, j, s[1] - 1) = prop(i, j, 0);
      }
    }
  }
  // edges
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<1, DimsToRepeat...> &&
                 variadic::contains<2, DimsToRepeat...>)) {
    for (std::size_t i = 0;
         i < (variadic::contains<0, DimsToRepeat...> ? s[0] - 1 : s[0]); ++i) {
      prop(i, s[0] - 1, s[1] - 1) = prop(i, 0, 0);
    }
  }
  if constexpr (sizeof...(DimsToRepeat) == 0 ||
                (variadic::contains<0, DimsToRepeat...> &&
                 variadic::contains<2, DimsToRepeat...>)) {
    for (std::size_t i = 0;
         i < (variadic::contains<1, DimsToRepeat...> ? s[1] - 1 : s[1]); ++i) {
      prop(s[0] - 1, i, s[1] - 1) = prop(0, i, 0);
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
}  // namespace tatooine
//==============================================================================
