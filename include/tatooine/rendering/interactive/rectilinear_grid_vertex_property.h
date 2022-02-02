#ifndef TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID_VERTEX_PROPERTY_H
#define TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID_VERTEX_PROPERTY_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::detail::interactive {
//==============================================================================
template <typename T, typename Axis0, typename Axis1, bool B>
struct renderer<
    tatooine::detail::rectilinear_grid::typed_vertex_property_interface<
        tatooine::rectilinear_grid<Axis0, Axis1>, T, B>> {
  using prop_type =
      tatooine::detail::rectilinear_grid::typed_vertex_property_interface<
          tatooine::rectilinear_grid<Axis0, Axis1>, T, B>;
  //==============================================================================
  struct scale : gl::indexeddata<Vec2<GLfloat>> {
    struct viridis_t {
      gl::tex1rgb32f tex;
      viridis_t(){
        auto s = color_scales::viridis<GLfloat>{};
        tex.upload_data(s.data(), 256);
        tex.set_wrap_mode(gl::WrapMode::CLAMP_TO_EDGE);
      }
    };
    static auto viridis() -> auto& {
      static auto instance = viridis_t{};
      return instance;
    }
  };
  struct geometry : gl::indexeddata<Vec2<GLfloat>> {
    static auto get() -> auto& {
      static auto instance = geometry{};
      return instance;
    }
    explicit geometry() {
      vertexbuffer().resize(4);
      {
        auto vb_map = vertexbuffer().wmap();
        vb_map[0] = Vec2<GLfloat>{0,0};
        vb_map[1] = Vec2<GLfloat>{1,0};
        vb_map[2] = Vec2<GLfloat>{1,1};
        vb_map[3] = Vec2<GLfloat>{0,1};
      }
      indexbuffer().resize(6);
      {
        auto data = indexbuffer().wmap();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 3;
        data[3]   = 1;
        data[4]   = 2;
        data[5]   = 3;
      }
    }
  };
  //==============================================================================
  struct shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "uniform mat4 modelview_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform vec2 extent;\n"
        "uniform vec2 pixel_width;\n"
        "out vec2 texcoord;\n"
        "void main() {\n"
        "  texcoord = (position * extent + pixel_width/2 ) /\n"
        "             (extent+pixel_width);\n"
        "  gl_Position = projection_matrix *\n"
        "                modelview_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform sampler2D data;\n"
        "uniform sampler1D color_scale;\n"
        "uniform float min;\n"
        "uniform float max;\n"
        "in vec2 texcoord;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  float scalar = texture(data, texcoord).r;\n"
        "  scalar = clamp((scalar - min) / (max - min), 0, 1);\n"
        "  vec3 col = texture(color_scale, scalar).rgb;\n"
        //"  vec3 col = vec3(scalar);\n"
        "  out_color = vec4(col, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
      set_uniform("data", 0);
      set_uniform("color_scale", 1);
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_modelview_matrix(Mat4<GLfloat>::eye());
      set_min(0);
      set_max(1);
    }
    //------------------------------------------------------------------------------
   public:
    //------------------------------------------------------------------------------
    auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_modelview_matrix(Mat4<GLfloat> const& MV) -> void {
      set_uniform_mat4("modelview_matrix", MV.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_extent(Vec2<GLfloat> const& extent) -> void {
      set_uniform_vec2("extent", extent.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_pixel_width(Vec2<GLfloat> const& pixel_width) -> void {
      set_uniform_vec2("pixel_width", pixel_width.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_min(GLfloat const min) -> void { set_uniform("min", min); }
    auto set_max(GLfloat const max) -> void { set_uniform("max", max); }
  };
  //==============================================================================
  struct render_data {
    int           line_width = 1;
    Vec4<GLfloat> color      = {0, 0, 0, 1};
    GLfloat       min = 0, max = 1;
    gl::tex2r32f  tex;
  };
  //==============================================================================
  static auto init(prop_type const& prop) {
    auto       d            = render_data{};
    auto       data         = std::vector<T>{};
    data.reserve(prop.grid().vertices().size());
    d.min = std::numeric_limits<GLfloat>::max();
    d.max = -std::numeric_limits<GLfloat>::max();
    prop.grid().vertices().iterate_indices([&](auto const... is) {
      auto const p = prop(is...);
      d.min = std::min<GLfloat>(d.min, p);
      d.max = std::max<GLfloat>(d.max, p);
      data.push_back(p);
    });
    d.tex.upload_data(data, prop.grid().template size<0>(),
                      prop.grid().template size<1>());
    return d;
  }
  //==============================================================================
  static auto properties(render_data& data) {
    ImGui::Text("Rectilinear Grid Vertex Property");
    ImGui::DragInt("Line width", &data.line_width, 1, 1, 20);
    ImGui::DragFloat("Min", &data.min, 0.01f, -FLT_MAX, data.max);
    ImGui::DragFloat("Max", &data.max, 0.01f, data.min, FLT_MAX);
  }
  //==============================================================================
  static auto render(camera auto const& cam, prop_type const& prop,
                     render_data& data) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_t;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    auto& shader = shader::get();
    shader.bind();
    shader.set_extent(Vec2<GLfloat>{prop.grid().extent()});
    shader.set_pixel_width(
        Vec2<GLfloat>{prop.grid().template dimension<0>()[1] -
                          prop.grid().template dimension<0>()[0],
                      prop.grid().template dimension<1>()[1] -
                          prop.grid().template dimension<1>()[0]});
    if constexpr (cam_is_float) {
      shader.set_projection_matrix(cam.projection_matrix());
    } else {
      shader.set_projection_matrix(Mat4<GLfloat>{cam.projection_matrix()});
    }

    if constexpr (cam_is_float) {
      shader.set_modelview_matrix(
          cam.view_matrix() *
          scale_matrix<GLfloat>(prop.grid().template extent<0>(),
                                prop.grid().template extent<1>(), 1) *
          translation_matrix<GLfloat>(
              prop.grid().template dimension<0>().front(),
              prop.grid().template dimension<1>().front(), 0));
    } else {
      shader.set_modelview_matrix(
          Mat4<GLfloat>{cam.view_matrix()} *
          scale_matrix<GLfloat>(prop.grid().template extent<0>(),
                                prop.grid().template extent<1>(), 1) *
          translation_matrix<GLfloat>(
              prop.grid().template dimension<0>().front(),
              prop.grid().template dimension<1>().front(), 0));
    }

    data.tex.bind(0);
    scale::viridis().tex.bind(1);
    shader::get().set_min(data.min);
    shader::get().set_max(data.max);
    gl::line_width(data.line_width);
    geometry::get().draw_triangles();
  }
};
//==============================================================================
}  // namespace tatooine::rendering::detail::interactive
//==============================================================================
#endif
