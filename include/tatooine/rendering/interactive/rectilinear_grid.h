#ifndef TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID_H
#define TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID_H
//==============================================================================
#include <tatooine/color_scales/viridis.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename Axis0, typename Axis1>
struct renderer<tatooine::rectilinear_grid<Axis0, Axis1>> {
  using renderable_type = tatooine::rectilinear_grid<Axis0, Axis1>;
  //==============================================================================
  struct color_scale : gl::indexeddata<Vec2<GLfloat>> {
    struct viridis_t {
      gl::tex1rgb32f tex;
      viridis_t() {
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
  //============================================================================
  struct geometry : gl::indexeddata<Vec2<GLfloat>> {
    static auto get() -> auto& {
      static auto instance = geometry{};
      return instance;
    }
    explicit geometry() {
      vertexbuffer().resize(4);
      {
        auto vb_map = vertexbuffer().wmap();
        vb_map[0]   = Vec2<GLfloat>{0, 0};
        vb_map[1]   = Vec2<GLfloat>{1, 0};
        vb_map[2]   = Vec2<GLfloat>{1, 1};
        vb_map[3]   = Vec2<GLfloat>{0, 1};
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
  struct property_shader : gl::shader {
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
        "  out_color = vec4(col, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = property_shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    property_shader() {
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
  struct line_shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "uniform mat4 modelview_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "void main() {\n"
        "  gl_Position = projection_matrix *\n"
        "                modelview_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  out_color = color;"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = line_shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    line_shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
      set_color(0, 0, 0);
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_modelview_matrix(Mat4<GLfloat>::eye());
    }
    //------------------------------------------------------------------------------
   public:
    //------------------------------------------------------------------------------
    auto set_color(GLfloat const r, GLfloat const g, GLfloat const b,
                   GLfloat const a = 1) -> void {
      set_uniform("color", r, g, b, a);
    }
    //------------------------------------------------------------------------------
    auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_modelview_matrix(Mat4<GLfloat> const& MV) -> void {
      set_uniform_mat4("modelview_matrix", MV.data().data());
    }
  };
  //==============================================================================
 private:
  int                line_width             = 1;
  bool               show_grid              = true;
  bool               show_property          = false;
  Vec4<GLfloat>      color                  = {0, 0, 0, 1};
  std::string const* selected_property_name = nullptr;
  typename renderable_type::vertex_property_t const* selected_property =
      nullptr;
  bool        vector_property    = false;
  char const* selected_component = nullptr;

  gl::indexeddata<Vec2<GLfloat>> geometry;
  GLfloat                        min_scalar = 0, max_scalar = 1;
  gl::tex2r32f                   tex;

 public:
  //==============================================================================
  renderer(renderable_type const& grid) {
    // grid geometry
    auto const num_vertices =
        grid.template size<0>() * 2 + grid.template size<1>() * 2;
    geometry.vertexbuffer().resize(num_vertices);
    geometry.indexbuffer().resize(num_vertices);
    {
      auto data = geometry.vertexbuffer().wmap();
      auto k    = std::size_t{};
      for (std::size_t i = 0; i < grid.template size<0>(); ++i) {
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>()[i],
                                  grid.template dimension<1>().front()};
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>()[i],
                                  grid.template dimension<1>().back()};
      }
      for (std::size_t i = 0; i < grid.template size<1>(); ++i) {
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>().front(),
                                  grid.template dimension<1>()[i]};
        data[k++] = Vec2<GLfloat>{grid.template dimension<0>().back(),
                                  grid.template dimension<1>()[i]};
      }
    }
    {
      auto data = geometry.indexbuffer().wmap();
      for (std::size_t i = 0; i < num_vertices; ++i) {
        data[i] = i;
      }
    }

    tex.resize(grid.template size<0>(), grid.template size<1>());
  }
  //==============================================================================
  template <typename T>
  static auto cast_prop(auto const* prop) {
    return static_cast<typename renderable_type::
                           typed_vertex_property_interface_t<T, false> const*>(
        prop);
  }
  //==============================================================================
  auto properties(renderable_type const& grid) {
    static const char* vector_items[] = {"magnitude", "x", "y", "z", "w"};
    auto               upload         = [&](auto&& prop, auto&& get_data) {
      auto texdata = std::vector<GLfloat>{};
      texdata.reserve(grid.vertices().size());
      min_scalar = std::numeric_limits<GLfloat>::max();
      max_scalar = -std::numeric_limits<GLfloat>::max();

      grid.vertices().iterate_indices([&](auto const... is) {
        auto const p = get_data(prop, is...);
        min_scalar   = std::min<GLfloat>(min_scalar, p);
        max_scalar   = std::max<GLfloat>(max_scalar, p);
        texdata.push_back(p);
                            });
      tex.upload_data(texdata, grid.template size<0>(),
                                            grid.template size<1>());
    };
    auto retrieve_typed_scalar_prop = [](auto&& prop, auto&& f) {
      if (prop->type() == typeid(float)) {
        return f(*cast_prop<float>(prop));
      } else if (prop->type() == typeid(double)) {
        return f(*cast_prop<double>(prop));
      }
    };
    auto retrieve_typed_vec_prop = [](auto&& prop, auto&& f) {
      if (prop->type() == typeid(vec2d)) {
        return f(*cast_prop<vec2d>(prop));
      } else if (prop->type() == typeid(vec2f)) {
        return f(*cast_prop<vec2f>(prop));
      } else if (prop->type() == typeid(vec3d)) {
        return f(*cast_prop<vec3d>(prop));
      } else if (prop->type() == typeid(vec3f)) {
        return f(*cast_prop<vec3f>(prop));
      } else if (prop->type() == typeid(vec4d)) {
        return f(*cast_prop<vec4d>(prop));
      } else if (prop->type() == typeid(vec4f)) {
        return f(*cast_prop<vec4f>(prop));
      }
      throw std::runtime_error{"property does not hold a vector type"};
    };
    auto upload_scalar = [&](auto&& prop) {
      upload(prop,
             [](auto const& prop, auto const... is) { return prop(is...); });
      vector_property    = false;
      selected_component = nullptr;
    };
    auto upload_magnitude = [&](auto&& prop) {
      upload(prop, [](auto const& prop, auto const... is) {
        using prop_type = std::decay_t<decltype(prop)>;
        using vec_type  = typename prop_type::value_type;
        auto mag        = typename vec_type::value_type{};
        for (std::size_t j = 0; j < vec_type::num_components(); ++j) {
          mag += prop(is...)(j) * prop(is...)(j);
        }
        return mag / vec_type::num_components();
      });
    };
    auto upload_x = [&](auto&& prop) {
      upload(prop, [](auto const& prop, auto const... is) {
        return prop(is...).x();
      });
    };
    auto upload_y = [&](auto&& prop) {
      if constexpr (std::decay_t<decltype(prop)>::value_type::num_components() >
                    1) {
        upload(prop, [](auto const& prop, auto const... is) {
          return prop(is...).y();
        });
      }
    };
    auto upload_z = [&](auto&& prop) {
      if constexpr (std::decay_t<decltype(prop)>::value_type::num_components() >
                    2) {
        upload(prop, [](auto const& prop, auto const... is) {
          return prop(is...).z();
        });
      }
    };
    auto upload_w = [&](auto&& prop) {
      if constexpr (std::decay_t<decltype(prop)>::value_type::num_components() >
                    3) {
        upload(prop, [](auto const& prop, auto const... is) {
          return prop(is...).w();
        });
      }
    };
    ImGui::Text("Rectilinear Grid");
    ImGui::Checkbox("Show Grid", &show_grid);

    if (show_grid) {
      ImGui::DragInt("Line width", &line_width, 1, 1, 20);
      ImGui::ColorEdit4("Color", color.data().data());
    }

    ImGui::Checkbox("Show Property", &show_property);
    if (show_property) {
      if (ImGui::BeginCombo("##combo", selected_property_name != nullptr
                                           ? selected_property_name->c_str()
                                           : nullptr)) {
        for (auto const& [name, prop] : grid.vertex_properties()) {
          if (prop->type() == typeid(double) || prop->type() == typeid(float) ||
              prop->type() == typeid(vec2f) || prop->type() == typeid(vec2d) ||
              prop->type() == typeid(vec3f) || prop->type() == typeid(vec3d) ||
              prop->type() == typeid(vec4f) || prop->type() == typeid(vec4d)) {
            auto is_selected = selected_property == prop.get();
            if (ImGui::Selectable(name.c_str(), is_selected)) {
              selected_property      = prop.get();
              selected_property_name = &name;
              if (prop->type() == typeid(float) ||
                  prop->type() == typeid(double)) {
                retrieve_typed_scalar_prop(prop.get(), upload_scalar);
              } else if (prop->type() == typeid(vec2f) ||
                         prop->type() == typeid(vec2d) ||
                         prop->type() == typeid(vec3f) ||
                         prop->type() == typeid(vec3d) ||
                         prop->type() == typeid(vec4f) ||
                         prop->type() == typeid(vec4d)) {
                retrieve_typed_vec_prop(prop.get(), upload_magnitude);
                vector_property    = true;
                selected_component = vector_items[0];
              }
            }
            if (is_selected) {
              ImGui::SetItemDefaultFocus();
            }
          }
        }
        ImGui::EndCombo();
      }
    }
    if (vector_property) {
      if (ImGui::BeginCombo("##combo2", selected_component)) {
        auto n = retrieve_typed_vec_prop(selected_property, [](auto&& prop) {
          return std::decay_t<decltype(prop)>::value_type::num_components();
        });
        for (std::size_t i = 0; i < n + 1; ++i) {
          auto const is_selected = selected_component == vector_items[i];
          if (ImGui::Selectable(vector_items[i], is_selected)) {
            selected_component = vector_items[i];
            if (i == 0) {
              retrieve_typed_vec_prop(selected_property, upload_magnitude);
            } else if (i == 1) {
              retrieve_typed_vec_prop(selected_property, upload_x);
            } else if (i == 2) {
              retrieve_typed_vec_prop(selected_property, upload_y);
            } else if (i == 3) {
              retrieve_typed_vec_prop(selected_property, upload_z);
            } else if (i == 4) {
              retrieve_typed_vec_prop(selected_property, upload_w);
            }
          }
        }
        ImGui::EndCombo();
      }
      ImGui::DragFloat("Min", &min_scalar, 0.01f, -FLT_MAX, max_scalar);
      ImGui::DragFloat("Max", &max_scalar, 0.01f, min_scalar, FLT_MAX);
    }
  }
  //==============================================================================
  auto render() {
    if (show_grid) {
      render_grid();
    }
    if (show_property) {
      render_property();
    }
  }
  //------------------------------------------------------------------------------
  auto update(renderable_type const& grid, camera auto const& cam,
              auto const dt) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_t;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    if (show_grid) {
      if constexpr (cam_is_float) {
        line_shader::get().set_projection_matrix(cam.projection_matrix());
      } else {
        line_shader::get().set_projection_matrix(
            Mat4<GLfloat>{cam.projection_matrix()});
      }

      if constexpr (cam_is_float) {
        line_shader::get().set_modelview_matrix(cam.view_matrix());
      } else {
        line_shader::get().set_modelview_matrix(
            Mat4<GLfloat>{cam.view_matrix()});
      }
    }
    if (show_property) {
      if constexpr (cam_is_float) {
        property_shader::get().set_projection_matrix(cam.projection_matrix());
      } else {
        property_shader::get().set_projection_matrix(
            Mat4<GLfloat>{cam.projection_matrix()});
      }

      if constexpr (cam_is_float) {
        property_shader::get().set_modelview_matrix(
            cam.view_matrix() *
            scale_matrix<GLfloat>(grid.template extent<0>(),
                                  grid.template extent<1>(), 1) *
            translation_matrix<GLfloat>(grid.template dimension<0>().front(),
                                        grid.template dimension<1>().front(),
                                        0));
      } else {
        property_shader::get().set_modelview_matrix(
            Mat4<GLfloat>{cam.view_matrix()} *
            scale_matrix<GLfloat>(grid.template extent<0>(),
                                  grid.template extent<1>(), 1) *
            translation_matrix<GLfloat>(grid.template dimension<0>().front(),
                                        grid.template dimension<1>().front(),
                                        0));
      }
      property_shader::get().set_extent(Vec2<GLfloat>{grid.extent()});
      property_shader::get().set_pixel_width(Vec2<GLfloat>{
          grid.template dimension<0>()[1] - grid.template dimension<0>()[0],
          grid.template dimension<1>()[1] - grid.template dimension<1>()[0]});
    }
  }
  //------------------------------------------------------------------------------
  auto render_grid() {
    auto& line_shader                  = line_shader::get();
    line_shader.bind();

    line_shader.set_color(color(0), color(1), color(2), color(3));
    gl::line_width(line_width);
    geometry.draw_lines();
  }
  //----------------------------------------------------------------------------
  auto render_property() {
    property_shader::get().bind();
    tex.bind(0);
    color_scale::viridis().tex.bind(1);
    property_shader::get().set_min(min_scalar);
    property_shader::get().set_max(max_scalar);
    gl::line_width(line_width);
    geometry::get().draw_triangles();
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
