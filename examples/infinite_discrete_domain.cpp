#include <tatooine/rectilinear_grid.h>
#include <tatooine/trace_pathline.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
template <typename VertexPropSampler, std::size_t... RepeatedDims>
struct infinite_field : field<infinite_field<VertexPropSampler, RepeatedDims...>,
                              typename VertexPropSampler::real_t,
                              VertexPropSampler::num_dimensions(),
                              typename VertexPropSampler::tensor_t> {
  VertexPropSampler const& m_sampler;
  infinite_field(VertexPropSampler const& sampler) : m_sampler{sampler} {}
  using parent_t = field<infinite_field, typename VertexPropSampler::real_t,
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
    ([&] {
      auto const front  = m_sampler.grid().template front<i>();
      auto const back   = m_sampler.grid().template back<i>();
      auto const extent = back - front;
      while (x(i) < front) {
        x(i) += extent;
      }
      while (x(i) > back) {
        x(i) -= extent;
      }
    }(), ...);
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
    ([&] {
      auto constexpr  dim   = non_repeated_dimensions[i];
      auto const      front = m_sampler.grid().template front<dim>();
      auto const      back  = m_sampler.grid().template back<dim>();
      if (x(dim) < front) {
        inside = false;
      }
      if (x(dim) > back) {
        inside = false;
      }
    }(), ...);
    return inside;
  }
  auto constexpr is_inside(pos_t const& x) const -> bool {
    return is_inside(x, std::make_index_sequence<non_repeated_dimensions.size()>{});
  }
};
template <std::size_t... RepeatedDims, typename V>
auto make_infinite(V const& v) {
  return infinite_field<V, RepeatedDims...>{v};
}
int main() {
  auto discretized_domain = NonuniformRectilinearGrid<2>{
      std::vector{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0},
      std::vector{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                  5.0}};

  auto& discretized_data =
      discretized_domain.insert_contiguous_vertex_property<vec2>("foo");
  auto const s = discretized_domain.size();
  for_loop(
      [&](auto const... is) {
        discretized_data((is + 1)...) = vec2::randu(-1,1);
      },
      s[0] - 2, s[1] - 2);

  // borders
  for (std::size_t i = 1; i < s[0] - 1; ++i) {
    discretized_data(i, 0)        = discretized_data(i, s[1] - 2);
    discretized_data(i, s[1] - 1) = discretized_data(i, 1);
  }
  for (std::size_t i = 1; i < s[1] - 1; ++i) {
    discretized_data(0, i)        = discretized_data(s[0] - 2, i);
    discretized_data(s[0] - 1, i) = discretized_data(1, i);
  }
  // corners
  discretized_data(       0, 0       ) = discretized_data(s[0] - 2, s[1] - 2);
  discretized_data(s[0] - 1, s[1] - 1) = discretized_data(       1,        1);
  discretized_data(       0, s[1] - 1) = discretized_data(s[0] - 2,        1);
  discretized_data(s[0] - 1,        0) = discretized_data(       1, s[1] - 2);

  auto v = discretized_data.linear_sampler();

  discretized_domain.write("ghost_cells.vtk");

  auto w = make_infinite<0,1>(v);
  trace_pathline(w, vec2{0, 0}, 0, 100).write_vtk("streamline.vtk");

  auto resampled=rectilinear_grid{linspace{-10.0,10.0,1000}, linspace{-10.0,10.0,1000}};
  discretize(w, resampled, "resampled");
  resampled.write("resampled_ghost_cells.vtk");
}
