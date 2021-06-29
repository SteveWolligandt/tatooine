#include <tatooine/grid.h>
#include <tatooine/random.h>
#include <tatooine/tensor_operations.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto  g       = grid{5, 5};
  auto& vectors = g.insert_vec2_vertex_property("vectors");

  auto rand = random::uniform {-0.5, 0.5};
  g.vertices().iterate_indices([&](auto const... is) {
    vectors(is...) = normalize(vec2{rand(), rand()});
  });
  g.write_vtk("random_vectorfield.vtk");
  auto integral_curves = std::vector<line2>{};
  auto ode_solver = ode::vclibs::rungekutta43<double, 2>{};
  auto v = vectors.sampler();
  auto rand2 = random::uniform {0.3, 0.7};
  size_t const num_lines       = 1;
  integral_curves.reserve(num_lines);
  for (size_t i = 0; i < num_lines; ++i) {
    auto& integral_curve = integral_curves.emplace_back();
    auto& param          = integral_curve.parameterization();
    auto& tangents = integral_curve.insert_vec2_vertex_property("tangents");
    auto const x0  = vec2{0.5, 0.5};
    //auto const x0  = vec2{rand2(), rand2()};
    ode_solver.solve(v, x0, 0, -100,
                     [&](auto const& x, auto const t, auto const dx) {
                       auto const v = integral_curve.push_front(x);
                       param[v]     = t;
                       tangents[v]  = dx;
                     });
    ode_solver.solve(v, x0, 0, 100,
                     [&](auto const& x, auto const t, auto const dx) {
                       auto const v = integral_curve.push_back(x);
                       param[v]     = t;
                       tangents[v]  = dx;
                     });
    //integral_curve = integral_curve.resample<interpolation::cubic>(linspace{
    //    param[line2::vertex_handle{0}],
    //    param[line2::vertex_handle{integral_curve.num_vertices() - 1}], 100});
    integral_curve.write_vtk("integral_curve.vtk");
  }
  //write_vtk(integral_curves, "integral_curves.vtk");
}
