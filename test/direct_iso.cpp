#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/direct_iso.h>
#include <tatooine/grid.h>
#include <tatooine/line.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/spacetime_vectorfield.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("direct_iso_grid_vertex_sampler") {
  auto v = analytical::fields::numerical::doublegyre {};
  auto vst = spacetime_vectorfield{v};
  auto mag_vst = length(vst);

  //auto g =
  //    grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}, linspace{0.0, 10.0, 11}};
  //auto& s       = discretize(mag_vst, g, "s", 0);
  //g.write_vtk("discretized_iso_volume.vtk");
  //auto iso = 1.05;
  //parameterized_line<real_t, 3, interpolation::linear> l;
  //l.push_back(vec3{4, 2, -2}, 0);
  //l.push_back(vec3{4, 2, 12}, 1);
  //parameterized_line<real_t, 3, interpolation::linear> l2;
  //l2.push_back(vec3{1, 0.5, 0}, 0);
  //l2.push_back(vec3{1, 0.5, 10}, 1);

  auto g =
      grid{linspace{0.0, 2.0, 10}, linspace{0.0, 2.0, 10}, linspace{0.0, 2.0, 10}};
  auto& s = g.add_scalar_vertex_property("s");
  auto rand = random_uniform{-1.0, 1.0, 1234ul};
  g.loop_over_vertex_indices([&](auto const ix, auto const iy, auto const iz) {
    //if (iy == 1) {
    //  s(ix, iy, iz) = -1;
    //} else {
    //  s(ix, iy, iz) = 1;
    //}
    s(ix, iy, iz) = rand();
  });
  auto                                                 iso = 0;
  parameterized_line<real_t, 3, interpolation::linear> l;
  l.push_back(vec3{-2, 1, -2}, 0);
  l.push_back(vec3{-2, 1, 2}, 1);
  parameterized_line<real_t, 3, interpolation::linear> l2;
  l2.push_back(vec3{1, 1, 1}, 0);
  l2.push_back(vec3{1, 1, 1}, 1);

  auto  sampler = s.linear_sampler();
  size_t i = 0;
  auto const ts = linspace{0.0, 1.0, 100};
  //auto const t = ts.back();
   for (auto const t : ts)
  {
    std::cout << t << '\n';
    rendering::perspective_camera cam{l(t), l2(t), vec3{0, 1, 0},
                                      60,   5,           5};
    //rendering::perspective_camera cam{vec3{11, 6, 11}, vec3{1, 0.4, 5}, vec3{0, 1, 0},
    //                                  30,   10,           10};
    
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << i;
    direct_iso(cam, sampler, iso,
               [](auto const&) {
                 return vec3{1, 0, 0};
               })
        .vertex_property<vec3>("rendering")
        .write_png("direct_iso." + ss.str() + ".png");
    ++i;
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
