#include<tatooine/triangular_mesh.h>
#include<tatooine/rendering/perspective_camera.h>
#include<tatooine/grid.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
using mesh_t = triangular_mesh<real_t, 3>;
using cam_t  = rendering::perspective_camera<real_t>;
//==============================================================================
auto main(int const argc, char const** argv) -> int {
  auto const resolution_x = std::size_t(800);
  auto const resolution_y = std::size_t(600);
  auto const        cam = cam_t{vec3{0, 1, -10}, vec3::zeros(), 60, 0.01, 1000,
                         resolution_x,    resolution_y};
  grid image{resolution_x, resolution_y};
  auto& rendered_mesh = image.add_vertex_property<vec3>("rendered mesh");
  mesh_t const mesh{argv[1]};
  //
  //#pragma omp parallel for collapse(2)
  //  for (std::size_t y = 0; y < resolution_y; ++y) {
  //    for (std::size_t x = 0; x < resolution_x; ++x) {
  //      //auto intersection = mesh.check_intersection(cam.ray(x, y));
  //      //if (intersection) {
  //      //  rendered_mesh(x, y) = vec3::ones();
  //      //} else {
  //      //  rendered_mesh(x, y) = vec3::zeros();
  //      //}
  //    }
  //  }
  //  write_png(rendered_mesh, argv[2]);
}
