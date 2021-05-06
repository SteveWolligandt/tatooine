#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/chrono.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/rendering/raytracing/render.h>
#include <tatooine/streamsurface.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
using mesh_t = triangular_mesh<real_t, 3>;
using cam_t  = rendering::perspective_camera<real_t>;
//==============================================================================
auto main(int const argc, char const** argv) -> int {
  auto const resolution_x = std::size_t(100);
  auto const resolution_y = std::size_t(100);
  auto const fov          = 30.0;
  auto const near         = 0.1;
  auto const far          = 100.0;
  auto const eye          = vec3{4.106644042631196, 2.3456801502763103, 15.671478391087849};
  auto const lookat       = vec3{-0.13426537562576765, -0.28287604628028545, 5.681056140184295};
  auto const cam =
      cam_t{eye, lookat, fov, near, far, resolution_x, resolution_y};

  analytical::fields::numerical::doublegyre dg;
  auto                                      stdg = spacetime(dg);
  parameterized_line<real_t, 3, interpolation::cubic> seed{
      std::pair{vec3{0.1, 0.1, 0.0}, 0.0},
      std::pair{vec3{1, 0.5, 0.0}, 0.5},
      std::pair{vec3{1.9, 0.1, 0.0}, 1.0}};
  streamsurface surf{stdg, seed};
  auto const    mesh = surf.discretize(100, 0.1, 0, 10);
  mesh.build_hierarchy();
  mesh.write_vtk("streamsurface.vtk");
  //mesh.hierarchy().write_vtk("streamsurface_hierarchy.vtk");

  auto const [rendering_duration, image] =
      measure([&] { return rendering::raytracing::render(cam, mesh); });

  write_png(image.vertex_property<vec3>("image"), argv[1]);
}
