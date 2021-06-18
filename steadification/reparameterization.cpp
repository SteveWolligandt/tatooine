#include <Tatooine/doublegyre.h>
#include <Tatooine/interpolators.h>
#include <Tatooine/regular_grid.h>
#include <Tatooine/spacetime_vectorfield.h>
#include <Tatooine/streamsurface.h>
#include "shaders.h"

using namespace tatooine;
using namespace tatooine::analytical;

int main() {
  doublegyre                          dg;
  SpacetimeVectorfield                dgst{dg};
  Line<3, double, LinearInterpolator> seed{{{0.4, 0.5, 0}, 0},
                                           {{0.6, 0.5, 0}, 1}};
  auto                                surf = make_streamsurface(dgst, seed);
  auto        mesh        = surf.discretize(0, 5, 0.05, 10, -10);
  const auto& uv_prop     = mesh.vertex_property<Vec<double, 2>>("uv");
  auto&       new_uv_prop = mesh.add_vertex_property<Vec<double, 2>>("uv_new");

  double              max_width = -std::numeric_limits<double>::max();
  std::vector<double> widths;
  for (const auto& front : mesh.fronts) {
    double width = 0;
    for (const auto& [subfront, range] : front) {
      for (auto it = begin(subfront); it != prev(end(subfront)); ++it)
        width += norm(mesh[*next(it)] - mesh[*it]);
    }
    max_width = std::max(width, max_width);
    widths.push_back(width);
  }

  auto width_it = begin(widths);
  for (const auto& front : mesh.fronts) {
    auto scale_factor = *width_it / max_width;
    for (const auto& [subfront, range] : front)
      for (auto v : subfront)
        new_uv_prop[v] = {uv_prop[v](0) * scale_factor, uv_prop[v](1)};
    ++width_it;
  }
  mesh.write_vtk("reparam.vtk");

  size_t tex_width = 1000, tex_height = 1000;
  size_t window_width = 2000, window_height = 1000;
  regular_grid_sampler<2, double, double, interpolation::linear,
                       interpolation::linear>
      noise(tex_width, tex_height);
  noise.fill_random();

  rendering::gl::Window             w("foo", window_width, window_height);
  rendering::gl::OrthographicCamera cam(0, 2, 0, 1, -10000, 100000, window_width,
                                window_height);
  // cam.transform().look_at({1, 1, 1}, {0, 0, 0}, {0, 1, 0});
  MeshViewerShader shader;
  shader.bind();
  shader.set_projection(cam.projection_matrix());
  shader.set_modelview(cam.view_matrix());

  rendering::gl::Texture2D<float, R> noise_tex(tex_width, tex_height, noise.data());
  noise_tex.bind();

  using vbo_t = rendering::gl::VertexBuffer<vec3f, vec2f>;

  rendering::gl::IndexBuffer ibo(mesh.num_faces() * 3);
  vbo_t              vbo(mesh.num_vertices());

  {
    auto   map = ibo.map();
    size_t i   = 0;
    for (auto f : mesh.faces()) {
      map[i++] = mesh[f][0].i;
      map[i++] = mesh[f][1].i;
      map[i++] = mesh[f][2].i;
    }
  }

  {
    auto map = vbo.map();
    for (auto v : mesh.vertices())
      map[v.i] = {{(float)mesh[v](0), (float)mesh[v](1), (float)mesh[v](2)},
                  {(float)new_uv_prop[v](0), (float)new_uv_prop[v](1)}};
  }

  rendering::gl::VertexArray vao;

  vao.bind();
  vbo.bind();
  ibo.bind();
  vbo.activate_attributes();

  w.set_render_function([&] {
      rendering::gl::clear_color_depth_buffer();
      rendering::gl::viewport(0, 0, window_width, window_height);
    vao.draw_triangles(mesh.num_faces() * 3);
  });

  w.start_rendering();
}
