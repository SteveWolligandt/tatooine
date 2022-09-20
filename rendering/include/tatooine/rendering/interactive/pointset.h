#ifndef TATOOINE_RENDERING_INTERACTIVE_POINTSET_H
#define TATOOINE_RENDERING_INTERACTIVE_POINTSET_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/linspace.h>
#include <tatooine/pointset.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
#include <tatooine/rendering/interactive/color_scale.h>
#include <tatooine/rendering/interactive/interactively_renderable.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <floating_point Real>
struct renderer<tatooine::pointset<Real, 2>> {
  static constexpr std::array<std::string_view, 5> vector_component_names = {
      "magnitude", "x", "y", "z", "w"};
  using renderable_type = tatooine::pointset<Real, 2>;
  template <typename T>
  using typed_vertex_property_type =
      typename renderable_type::template typed_vertex_property_type<T>;
  //==============================================================================
  struct shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "layout (location = 1) in float prop;\n"
        "out float prop_frag;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 view_matrix;\n"
        "void main() {\n"
        "  prop_frag = prop;\n"
        "  gl_Position = projection_matrix * view_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "in float prop_frag;\n"
        "out vec4 out_color;\n"
        "uniform int show_property;\n"
        "uniform int invert_scale;\n"
        "uniform float min;\n"
        "uniform float max;\n"
        "uniform sampler1D color_scale;\n"
        "void main() {\n"
        "  if (show_property == 0) {\n"
        "    out_color = color;\n"
        "  } else {\n"
        "    float scalar = prop_frag;\n"
        "    if (isnan(scalar)) { discard; }\n"
        "    scalar = clamp((scalar - min) / (max - min), 0, 1);\n"
        "    if (invert_scale == 1) {\n"
        "      scalar = 1 - scalar;\n"
        "    }\n"
        "    vec3 col = texture(color_scale, scalar).rgb;\n"
        "    out_color = vec4(col, 1);\n"
        "  }\n"
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
      set_color(0, 0, 0);
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_view_matrix(Mat4<GLfloat>::eye());
      set_uniform("color_scale", 0);
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
      set_uniform_mat4("projection_matrix", P.data());
    }
    //------------------------------------------------------------------------------
    auto set_view_matrix(Mat4<GLfloat> const& V) -> void {
      set_uniform_mat4("view_matrix", V.data());
    }
    //--------------------------------------------------------------------------
    auto invert_scale(bool const invert) -> void {
      set_uniform("invert_scale", invert ? 1 : 0);
    }
    //--------------------------------------------------------------------------
    auto show_property(bool const show) -> void {
      set_uniform("show_property", show ? 1 : 0);
    }
    //--------------------------------------------------------------------------
    auto set_min(GLfloat const min) -> void {
      set_uniform("min", min);
    }
    //--------------------------------------------------------------------------
    auto set_max(GLfloat const max) -> void {
      set_uniform("max", max);
    }
  };
  static auto set_projection_matrix(Mat4<GLfloat> const& P) {
    shader::get().set_projection_matrix(P);
  }
  static auto set_view_matrix(Mat4<GLfloat> const& V) {
    shader::get().set_view_matrix(V);
  }
  //==============================================================================
  struct property_settings {
    color_scale* c              = nullptr;
    GLfloat      min_scalar     = std::numeric_limits<GLfloat>::max();
    GLfloat      max_scalar     = -std::numeric_limits<GLfloat>::max();
    bool         scale_inverted = false;
  };
  //==============================================================================
  int                                                point_size = 1;
  Vec4<GLfloat>                                      color      = {0, 0, 0, 1};
  gl::indexeddata<Vec2<GLfloat>, GLfloat>            geometry;
  vec2d                                              cursor_pos;
  typename renderable_type::vertex_property_type const* selected_property =
      nullptr;
  std::string const* selected_property_name = nullptr;
  std::unordered_map<std::string, property_settings> settings;
  std::unordered_map<std::string, std::string_view>  selected_component;
  bool                                               vector_property = false;
  bool                                               show_property   = false;
  //==============================================================================
  renderer(renderable_type const& ps) {
    {
      geometry.vertexbuffer().resize(ps.vertices().size());
      auto m = geometry.vertexbuffer().wmap();
      for (auto v : ps.vertices()) {
        m[v.index()] = {Vec2<GLfloat>{ps[v]}, GLfloat{}};
      }
    }
    {
      geometry.indexbuffer().resize(ps.vertices().size());
      auto m = geometry.indexbuffer().wmap();
      for (auto v : ps.vertices()) {
        m[v.index()] = v.index();
      }
    }
    init_properties(ps);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  static auto cast_prop(auto&& prop) {
    return static_cast<typed_vertex_property_type<T> const*>(prop);
  }
  //----------------------------------------------------------------------------
  auto retrieve_typed_prop(auto&& prop, auto&& f) {
    if (prop->type() == typeid(float)) {
      f(*cast_prop<float>(prop));
    } else if (prop->type() == typeid(double)) {
      f(*cast_prop<double>(prop));
    } else if (prop->type() == typeid(vec2d)) {
      f(*cast_prop<vec2d>(prop));
    } else if (prop->type() == typeid(vec2f)) {
      f(*cast_prop<vec2f>(prop));
    } else if (prop->type() == typeid(vec3d)) {
      f(*cast_prop<vec3d>(prop));
    } else if (prop->type() == typeid(vec3f)) {
      f(*cast_prop<vec3f>(prop));
    } else if (prop->type() == typeid(vec4d)) {
      f(*cast_prop<vec4d>(prop));
    } else if (prop->type() == typeid(vec4f)) {
      f(*cast_prop<vec4f>(prop));
    }
  }
  //----------------------------------------------------------------------------
  auto prop_holds_scalar(auto const& prop) {
    return prop->template holds_type<float>() ||
           prop->template holds_type<double>();
  }
  //----------------------------------------------------------------------------
  auto prop_holds_vector(auto const& prop) {
    return prop->template holds_type<vec2f>() ||
           prop->template holds_type<vec2d>() ||
           prop->template holds_type<vec3f>() ||
           prop->template holds_type<vec3d>() ||
           prop->template holds_type<vec4f>() ||
           prop->template holds_type<vec4d>();
  }
  //----------------------------------------------------------------------------
  auto init_properties(renderable_type const& ps) {
    for (auto const& [name, prop] : ps.vertex_properties()) {
      if (prop_holds_scalar(prop)) {
        auto min_scalar = std::numeric_limits<GLfloat>::max();
        auto max_scalar = -std::numeric_limits<GLfloat>::max();
        retrieve_typed_prop(prop.get(), [&](auto const& prop) {
          using prop_type  = std::decay_t<decltype(prop)>;
          using value_type = typename prop_type::value_type;
          if constexpr (is_arithmetic<value_type>) {
            for (auto const v: ps.vertices()) {
              auto const p = prop[v];
              min_scalar   = std::min<GLfloat>(min_scalar, p);
              max_scalar   = std::max<GLfloat>(max_scalar, p);
            }
          }
        });
        settings[name] = {&color_scale::viridis(), min_scalar, max_scalar};
      } else if (prop_holds_vector(prop)) {
        retrieve_typed_prop(prop.get(), [&](auto const& prop) {
          using prop_type  = std::decay_t<decltype(prop)>;
          using value_type = typename prop_type::value_type;
          if constexpr (static_vec<value_type>) {
            auto constexpr num_comps = value_type::num_components();
            auto min_scalars         = std::vector<GLfloat>(
                num_comps + 1, std::numeric_limits<GLfloat>::max());
            auto max_scalars = std::vector<GLfloat>(
                num_comps + 1, -std::numeric_limits<GLfloat>::max());
            for (auto const v: ps.vertices()) {
              auto const p   = prop[v];
              auto       mag = typename value_type::value_type{};
              for (std::size_t j = 0; j < num_comps; ++j) {
                mag += p(j) * p(j);
                min_scalars[j + 1] =
                    std::min<GLfloat>(min_scalars[j + 1], p(j));
                max_scalars[j + 1] =
                    std::max<GLfloat>(max_scalars[j + 1], p(j));
              }
              mag            = std::sqrt(mag);
              min_scalars[0] = std::min<GLfloat>(min_scalars[0], mag);
              max_scalars[0] = std::max<GLfloat>(max_scalars[0], mag);
            }

            for (std::size_t j = 0; j < num_comps + 1; ++j) {
              settings[name + '_' + std::string{vector_component_names[j]}] = {
                  &color_scale::viridis(), min_scalars[j], max_scalars[j]};
            }
            selected_component[name] = vector_component_names[0];
          }
        });
      }
    }
  }
  //----------------------------------------------------------------------------
  auto selected_settings_name() const {
    auto name = std::string{};
    if (selected_property_name != nullptr) {
      name = *selected_property_name;
      if (auto c = selected_component.find(name);
          c != end(selected_component)) {
        name += "_";
        name += c->second;
      }
    }
    return name;
  }
  //----------------------------------------------------------------------------
  auto upload_data(auto&& prop, auto&& get_data,
                              renderable_type const& ps) {
    auto map = geometry.vertexbuffer().wmap();
    for (auto const v : ps.vertices()) {
      map[v.index()] = {Vec2<GLfloat>{ps[v]}, get_data(prop, v)};
    }
  };
  //----------------------------------------------------------------------------
  auto upload_scalar(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (is_arithmetic<value_type>) {
        upload_data(
            prop, [](auto const& prop, auto const v) { return prop[v]; }, grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_magnitude(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        upload_data(
            prop,
            [](auto const& prop, auto const v) {
              auto mag = typename value_type::value_type{};
              for (std::size_t j = 0; j < value_type::num_components(); ++j) {
                mag += prop[v](j) * prop[v](j);
              }
              return mag / value_type::num_components();
            },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_x(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        upload_data(
            prop, [](auto const& prop, auto const v) { return prop[v].x(); },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_y(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        upload_data(
            prop, [](auto const& prop, auto const v) { return prop[v].y(); },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_z(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        if constexpr (value_type::num_components() > 2) {
          upload_data(
              prop, [](auto const& prop, auto const v) { return prop[v].z(); },
              grid);
        }
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_w(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        if constexpr (value_type::num_components() > 3) {
          upload_data(
              prop, [](auto const& prop, auto const v) { return prop[v].w(); },
              grid);
        }
      }
    });
  }
  //----------------------------------------------------------------------------
  auto pointset_property_selection(renderable_type const& ps) {
    if (ImGui::BeginCombo("##combo", selected_property_name != nullptr
                                         ? selected_property_name->c_str()
                                         : nullptr)) {
      for (auto const& [name, prop] : ps.vertex_properties()) {
        if (prop->type() == typeid(float) || prop->type() == typeid(double) ||
            prop->type() == typeid(vec2f) || prop->type() == typeid(vec2d) ||
            prop->type() == typeid(vec3f) || prop->type() == typeid(vec3d) ||
            prop->type() == typeid(vec4f) || prop->type() == typeid(vec4d)) {
          auto is_selected = selected_property == prop.get();
          if (ImGui::Selectable(name.c_str(), is_selected)) {
            selected_property      = prop.get();
            selected_property_name = &name;
            if (prop_holds_scalar(prop)) {
              upload_scalar(selected_property, ps);
              vector_property = false;
            } else if (prop_holds_vector(prop)) {
              upload_magnitude(selected_property, ps);
              vector_property = true;
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
  //==============================================================================
  auto properties(renderable_type const& ps) {
    ImGui::Text("Pointset");
    ImGui::DragInt("Point Size", &point_size, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data());
    ImGui::Checkbox("Show Property", &show_property);
    pointset_property_selection(ps);
    if (selected_property != nullptr) {
      auto& setting = settings[selected_settings_name()];
      ImGui::DragFloat("Min", &setting.min_scalar, 0.01f, -FLT_MAX,
                       setting.max_scalar, "%.06f");
      ImGui::DragFloat("Max", &setting.max_scalar, 0.01f, setting.min_scalar,
                       FLT_MAX, "%.06f");
    }
  }
  //==============================================================================
  auto render() {
    shader::get().bind();
    shader::get().set_color(color(0), color(1), color(2), color(3));
    gl::point_size(point_size);
    auto show = selected_property != nullptr && show_property;
    shader::get().show_property(show);
    if (show) {
      auto& setting = settings[selected_settings_name()];
      setting.c->tex.bind(0);
      shader::get().set_min(setting.min_scalar);
      shader::get().set_max(setting.max_scalar);
      shader::get().invert_scale(setting.scale_inverted);
    }
    geometry.draw_points();
  }

  auto on_cursor_moved(double const x, double const y) { cursor_pos = {x, y}; }
  auto on_button_pressed(gl::button) {}
  auto on_button_released(gl::button) {}
};
static_assert(interactively_renderable<renderer<pointset2>>);
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
