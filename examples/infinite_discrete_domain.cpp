#include <tatooine/infinite_rectilinear_grid_vertex_property_sampler.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/trace_flow.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
int main() {
  auto discretized_domain = NonuniformRectilinearGrid<2>{std::vector{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                                                         std::vector{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}};

  discretized_domain.dimension<0>().push_back(
      2 * discretized_domain.back<0>() -
      discretized_domain.at<0>(discretized_domain.size<0>() - 2));
  discretized_domain.dimension<1>().push_back(
      2 * discretized_domain.back<1>() -
      discretized_domain.at<1>(discretized_domain.size<1>() - 2));

  auto& discretized_data =
      discretized_domain.insert_contiguous_vertex_property<vec2>("foo");
  auto const s = discretized_domain.size();
  for_loop(
      [&](auto const... is) { discretized_data((is)...) = vec2::randu(-1, 1); },
      s);

  repeat_for_infinite<0>(discretized_data);

  auto v = discretized_data.linear_sampler();

  std::cerr << "writing ghost_cells.vtk...";
  discretized_domain.write("ghost_cells.vtk");
  std::cerr << "done!\n";
  //
  auto w = make_infinite<0>(v);
  write_vtk(
      std::vector{
          trace_flow(w, vec2{0, 0}, 0, 100), trace_flow(w, vec2{1, 0}, 0, 100),
          trace_flow(w, vec2{2, 0}, 0, 100), trace_flow(w, vec2{3, 0}, 0, 100),
          trace_flow(w, vec2{4, 0}, 0, 100), trace_flow(w, vec2{5, 0}, 0, 100),
          trace_flow(w, vec2{0, 1}, 0, 100), trace_flow(w, vec2{1, 1}, 0, 100),
          trace_flow(w, vec2{2, 1}, 0, 100), trace_flow(w, vec2{3, 1}, 0, 100),
          trace_flow(w, vec2{4, 1}, 0, 100), trace_flow(w, vec2{5, 1}, 0, 100),
          trace_flow(w, vec2{0, 2}, 0, 100), trace_flow(w, vec2{1, 2}, 0, 100),
          trace_flow(w, vec2{2, 2}, 0, 100), trace_flow(w, vec2{3, 2}, 0, 100),
          trace_flow(w, vec2{4, 2}, 0, 100), trace_flow(w, vec2{5, 2}, 0, 100),
          trace_flow(w, vec2{0, 3}, 0, 100), trace_flow(w, vec2{1, 3}, 0, 100),
          trace_flow(w, vec2{2, 3}, 0, 100), trace_flow(w, vec2{3, 3}, 0, 100),
          trace_flow(w, vec2{4, 3}, 0, 100), trace_flow(w, vec2{5, 3}, 0, 100),
          trace_flow(w, vec2{0, 4}, 0, 100), trace_flow(w, vec2{1, 4}, 0, 100),
          trace_flow(w, vec2{2, 4}, 0, 100), trace_flow(w, vec2{3, 4}, 0, 100),
          trace_flow(w, vec2{4, 4}, 0, 100), trace_flow(w, vec2{5, 4}, 0, 100),
          trace_flow(w, vec2{0, 5}, 0, 100), trace_flow(w, vec2{1, 5}, 0, 100),
          trace_flow(w, vec2{2, 0}, 0, 100), trace_flow(w, vec2{3, 5}, 0, 100),
          trace_flow(w, vec2{4, 5}, 0, 100), trace_flow(w, vec2{5, 5}, 0, 100),
          trace_flow(w, vec2{0, 6}, 0, 100), trace_flow(w, vec2{1, 6}, 0, 100),
          trace_flow(w, vec2{2, 6}, 0, 100), trace_flow(w, vec2{3, 6}, 0, 100),
          trace_flow(w, vec2{4, 6}, 0, 100), trace_flow(w, vec2{5, 6}, 0, 100),
          trace_flow(w, vec2{0, 7}, 0, 100), trace_flow(w, vec2{1, 7}, 0, 100),
          trace_flow(w, vec2{2, 7}, 0, 100), trace_flow(w, vec2{3, 7}, 0, 100),
          trace_flow(w, vec2{4, 7}, 0, 100), trace_flow(w, vec2{5, 7}, 0, 100)},
      "streamline.vtk");

  auto resampled = rectilinear_grid{
      linspace{discretized_domain.front<0>() - discretized_domain.back<0>() +
                   discretized_domain.front<0>(),
               discretized_domain.back<0>() + discretized_domain.back<0>() -
                   discretized_domain.front<0>(),
               1000},
      linspace{discretized_domain.front<1>() - discretized_domain.back<1>() +
                   discretized_domain.front<1>(),
               discretized_domain.back<1>() + discretized_domain.back<1>() -
                   discretized_domain.front<1>(),
               1000}};
  discretize(w, resampled, "infinite", execution_policy::sequential);
  discretize(v, resampled, "raw", execution_policy::sequential);

  resampled.write("resampled_ghost_cells.vtk");
}
