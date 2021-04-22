#ifndef TATOOINE_RENDERING_RAYTRACING_RENDER_H
#define TATOOINE_RENDERING_RAYTRACING_RENDER_H
//==============================================================================
#include <tatooine/grid.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/triangular_mesh.h>
//==============================================================================
namespace tatooine::rendering::raytracing {
//==============================================================================
template <typename Real>
auto render(camera<Real> const& cam, triangular_mesh<Real, 3> const& mesh,
            vec<Real, 3> const& bg_color  = vec<Real, 3>::ones(),
            std::string const&  prop_name = "image") {
  grid  image{cam.plane_width(), cam.plane_height()};
  auto& rendered_mesh = image.template add_vertex_property<vec3>(prop_name);
  mesh.build_hierarchy();

  constexpr std::array offsets{vec2{0, 0}, vec2{-0.25, -0.25},
                               vec2{0.25, -0.25}, vec2{-0.25, 0.25},
                               vec2{0.25, 0.25}};
  Real const           shininess = 10;
  auto const           L         = normalize(cam.lookat() - cam.eye());
#pragma omp parallel for collapse(2)
  for (std::size_t y = 0; y < cam.plane_height(); ++y) {
    for (std::size_t x = 0; x < cam.plane_width(); ++x) {
      rendered_mesh(x, y) = vec<Real, 3>::zeros();
      for (auto const& offset : offsets) {
        auto const cur_ray      = cam.ray(x + offset(0), y + offset(1));
        auto const intersection = mesh.check_intersection(cur_ray);
        if (intersection) {
          auto const N = normalize(intersection->normal);
          auto const V = normalize(cur_ray.direction());

          auto const diffuse_color = intersection->normal * 0.5 + 0.5;
          auto       luminance     = diffuse_color;

          auto const illuminance = std::abs(dot(L, N));
          if (illuminance > 0) {
            auto       brdf     = diffuse_color;
            auto const half_dir = normalize(V + L);
            auto const spec_dot = std::max(dot(half_dir, N), 0.0);
            brdf += std::pow(spec_dot, shininess);
            luminance += brdf * illuminance;
          }

          rendered_mesh(x, y) += luminance;
        } else {
          rendered_mesh(x, y) += bg_color;
        }
      }
      rendered_mesh(x, y) /= offsets.size();
    }
  }
  return image;
}
//==============================================================================
}  // namespace tatooine::rendering::raytracing
//==============================================================================
#endif
