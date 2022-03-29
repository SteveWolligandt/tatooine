#ifndef TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
#define TATOOINE_RENDERING_INTERACTIVE_ELLIPSE_H
//==============================================================================
#include <tatooine/autonomous_particle.h>
#include <tatooine/geometry/ellipse.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/rendering/interactive/shaders.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename Ellipse>
requires(is_derived_from_hyper_ellipse<Ellipse> &&
         Ellipse::num_dimensions() == 2)
struct renderer<Ellipse> {
  using renderable_type = Ellipse;
  using real_type       = typename Ellipse::real_type;
  struct geometry : gl::indexeddata<Vec2<GLfloat>> {
    static auto get() -> auto& {
      static auto instance = geometry{};
      return instance;
    }
    explicit geometry(std::size_t const num_vertices = 128) {
      vertexbuffer().resize(num_vertices);
      {
        auto ts = linspace<float>{0, 2 * M_PI, num_vertices + 1};
        ts.pop_back();
        auto vb_map = vertexbuffer().wmap();
        auto i      = std::size_t{};
        for (auto const t : ts) {
          vb_map[i++] = Vec2<GLfloat>{std::cos(t), std::sin(t)};
        }
      }
      indexbuffer().resize(num_vertices);
      {
        auto data = indexbuffer().wmap();
        for (std::size_t i = 0; i < num_vertices; ++i) {
          data[i] = i;
        }
      }
    }
  };
  //==============================================================================
  using shader = shaders::colored_pass_through_2d;
  //==============================================================================
  int           line_width = 1;
  Vec4<GLfloat> color      = {0, 0, 0, 1};
  //==============================================================================
  renderer(renderable_type const& /*ell*/) {}
  //------------------------------------------------------------------------------
  static auto set_projection_matrix(Mat4<GLfloat> const& P) {
    shader::get().set_projection_matrix(P);
  }
  //------------------------------------------------------------------------------
  auto properties(renderable_type const& /*ell*/) {
    ImGui::Text("Ellipse");
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data());
  }
  //==============================================================================
  static auto construct_model_matrix(Mat2<real_type> const& S,
                                     Vec2<real_type> const& center) {
    auto constexpr O                   = GLfloat(0);
    auto constexpr I                   = GLfloat(1);
    static auto constexpr ell_is_float = is_same<GLfloat, real_type>;
    if constexpr (ell_is_float) {
      return Mat4<GLfloat>{{S(0, 0), S(0, 1), O, center(0)},
                           {S(1, 0), S(1, 1), O, center(1)},
                           {O, O, I, O},
                           {O, O, O, I}};
    } else {
      return Mat4<GLfloat>{
          {GLfloat(S(0, 0)), GLfloat(S(0, 1)), O, GLfloat(center(0))},
          {GLfloat(S(1, 0)), GLfloat(S(1, 1)), O, GLfloat(center(1))},
          {O, O, I, O},
          {O, O, O, I}};
    }
  }
  //==============================================================================
  auto update(auto const dt, auto& ell, camera auto const& cam) {
    auto& shader        = shader::get();
    using cam_real_type = typename std::decay_t<decltype(cam)>::real_type;
    static auto constexpr ell_is_float = is_same<GLfloat, real_type>;
    static auto constexpr cam_is_float = is_same<GLfloat, cam_real_type>;

    auto V = [&] {
      if constexpr (cam_is_float) {
        return cam.view_matrix();
      } else {
        return Mat4<GLfloat>{cam.view_matrix()};
      }
    }();
    shader.set_model_view_matrix(V *
                                 construct_model_matrix(ell.S(), ell.center()));
    shader.set_color(color(0), color(1), color(2), color(3));
  }
  //----------------------------------------------------------------------------
  auto render() {
    shader::get().bind();
    gl::line_width(line_width);
    geometry::get().draw_line_loop();
  }
};
//==============================================================================
template <range EllipseRange>
requires(is_derived_from_hyper_ellipse<
           std::ranges::range_value_t<EllipseRange>> &&
         std::ranges::range_value_t<EllipseRange>::num_dimensions() == 2)
struct renderer<EllipseRange> {
  using renderable_type = EllipseRange;
  using ellipse_type =  std::ranges::range_value_t<EllipseRange>;
  using real_type =
      typename ellipse_type::real_type;
  //==============================================================================
  using geometry = typename renderer<ellipse_type>::geometry;
  using shader   = typename renderer<ellipse_type>::shader;
  //==============================================================================
  int           line_width = 1;
  Vec4<GLfloat> color      = {0, 0, 0, 1};
  //==============================================================================
  renderer(renderable_type const& ell) {}
  //------------------------------------------------------------------------------
  static auto set_projection_matrix(Mat4<GLfloat> const& P) {
    shader::get().set_projection_matrix(P);
  }
  //------------------------------------------------------------------------------
  auto properties(renderable_type const& ell) {
    ImGui::Text("Ellipse");
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data());
  }
  //==============================================================================
  static auto construct_model_matrix(Mat2<real_type> const& S,
                                     Vec2<real_type> const& center) {
    auto constexpr O                   = GLfloat(0);
    auto constexpr I                   = GLfloat(1);
    static auto constexpr ell_is_float = is_same<GLfloat, real_type>;
    if constexpr (ell_is_float) {
      return Mat4<GLfloat>{{S(0, 0), S(0, 1), O, center(0)},
                           {S(1, 0), S(1, 1), O, center(1)},
                           {O, O, I, O},
                           {O, O, O, I}};
    } else {
      return Mat4<GLfloat>{
          {GLfloat(S(0, 0)), GLfloat(S(0, 1)), O, GLfloat(center(0))},
          {GLfloat(S(1, 0)), GLfloat(S(1, 1)), O, GLfloat(center(1))},
          {O, O, I, O},
          {O, O, O, I}};
    }
  }
  //==============================================================================
  auto render(auto const& ellipses, camera auto const& cam) {
    auto& shader = shader::get();
    shader.bind();
    using cam_real_type = typename std::decay_t<decltype(cam)>::real_type;
    static auto constexpr cam_is_float = is_same<GLfloat, cam_real_type>;

    gl::line_width(line_width);
    shader.set_color(color(0), color(1), color(2), color(3));
    auto const V = [&] {
      if constexpr (cam_is_float) {
        return cam.view_matrix();
      } else {
        return Mat4<GLfloat>{cam.view_matrix()};
      }
    }();
    for (auto const& ell : ellipses) {
      shader.set_model_view_matrix(
          V * construct_model_matrix(ell.S(), ell.center()));
      geometry::get().draw_line_loop();
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
