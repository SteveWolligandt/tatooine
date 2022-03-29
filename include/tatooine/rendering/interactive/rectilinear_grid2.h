#ifndef TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID2_H
#define TATOOINE_RENDERING_INTERACTIVE_RECTILINEAR_GRID2_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/color_scale.h>
#include <tatooine/rendering/interactive/renderer.h>
#include <tatooine/rendering/interactive/shaders.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <typename Axis0, typename Axis1>
struct renderer<tatooine::rectilinear_grid<Axis0, Axis1>> {
  static constexpr std::array<std::string_view, 5> vector_component_names = {
      "magnitude", "x", "y", "z", "w"};
  using renderable_type = tatooine::rectilinear_grid<Axis0, Axis1>;
  template <typename T>
  using typed_vertex_property_interface_type =
      typename renderable_type::typed_vertex_property_interface_type<T>;
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
  //============================================================================
  struct property_shader : gl::shader {
    //--------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "uniform mat4 model_view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform vec2 extent;\n"
        "uniform vec2 pixel_width;\n"
        "out vec2 texcoord;\n"
        "void main() {\n"
        "  texcoord = (position * extent + pixel_width / 2) /\n"
        "             (extent+pixel_width);\n"
        "  gl_Position = projection_matrix *\n"
        "                model_view_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //--------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform sampler2D data;\n"
        "uniform sampler1D color_scale;\n"
        "uniform float min;\n"
        "uniform float max;\n"
        "uniform int invert_scale;\n"
        "in vec2 texcoord;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  float scalar = texture(data, texcoord).r;\n"
        "  if (isnan(scalar)) {\n"
        "    out_color = vec4(1,0,0,1);\n"
        "    return;\n"
        "  }\n"
        "  scalar = clamp((scalar - min) / (max - min), 0, 1);\n"
        "  if (invert_scale == 1) {\n"
        "    scalar = 1 - scalar;\n"
        "  }\n"
        "  vec3 col = texture(color_scale, scalar).rgb;\n"
        "  out_color = vec4(col, 1);\n"
        "}\n";
    //--------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = property_shader{};
      return s;
    }
    //--------------------------------------------------------------------------
   private:
    //--------------------------------------------------------------------------
    property_shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
      set_uniform("data", 0);
      set_uniform("color_scale", 1);
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_model_view_matrix(Mat4<GLfloat>::eye());
      set_min(0);
      set_max(1);
      invert_scale(false);
    }
    //--------------------------------------------------------------------------
   public:
    //--------------------------------------------------------------------------
    auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data());
    }
    //--------------------------------------------------------------------------
    auto set_model_view_matrix(Mat4<GLfloat> const& MV) -> void {
      set_uniform_mat4("model_view_matrix", MV.data());
    }
    //--------------------------------------------------------------------------
    auto set_extent(Vec2<GLfloat> const& extent) -> void {
      set_uniform_vec2("extent", extent.data());
    }
    //--------------------------------------------------------------------------
    auto set_pixel_width(Vec2<GLfloat> const& pixel_width) -> void {
      set_uniform_vec2("pixel_width", pixel_width.data());
    }
    //--------------------------------------------------------------------------
    auto set_min(GLfloat const min) -> void { set_uniform("min", min); }
    auto set_max(GLfloat const max) -> void { set_uniform("max", max); }
    //--------------------------------------------------------------------------
    auto invert_scale(bool const invert) -> void {
      set_uniform("invert_scale", invert ? 1 : 0);
    }
  };
  //============================================================================
  using line_shader = shaders::colored_pass_through_2d;
  //============================================================================
 private:
  struct property_settings {
    color_scale* c              = nullptr;
    GLfloat      min_scalar     = std::numeric_limits<GLfloat>::max();
    GLfloat      max_scalar     = -std::numeric_limits<GLfloat>::max();
    bool         scale_inverted = false;
  };
  bool                                               show_grid     = true;
  bool                                               show_property = false;
  int                                                line_width    = 1;
  Vec4<GLfloat>                                      grid_color = {0, 0, 0, 1};
  std::unordered_map<std::string, property_settings> settings;
  std::unordered_map<std::string, std::string_view>  selected_component;
  std::string const* selected_property_name = nullptr;
  typename renderable_type::vertex_property_type const* selected_property =
      nullptr;

  bool vector_property = false;

  gl::indexeddata<Vec2<GLfloat>> geometry;
  gl::tex2r32f                   tex;

 public:
  //============================================================================
  renderer(renderable_type const& grid) {
    init_grid_geometry(grid);
    init_properties(grid);
  }
  //----------------------------------------------------------------------------
  auto init_grid_geometry(renderable_type const& grid) {
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
  }
  //----------------------------------------------------------------------------
  auto init_properties(renderable_type const& grid) {
    tex.resize(grid.template size<0>(), grid.template size<1>());

    for (auto const& [name, prop] : grid.vertex_properties()) {
      if (prop_holds_scalar(prop)) {
        auto min_scalar = std::numeric_limits<GLfloat>::max();
        auto max_scalar = -std::numeric_limits<GLfloat>::max();
        retrieve_typed_prop(prop.get(), [&](auto const& prop) {
          using prop_type  = std::decay_t<decltype(prop)>;
          using value_type = typename prop_type::value_type;
          if constexpr (is_arithmetic<value_type>) {
            grid.vertices().iterate_indices([&](auto const... is) {
              auto const p = prop.at(is...);
              min_scalar   = std::min<GLfloat>(min_scalar, p);
              max_scalar   = std::max<GLfloat>(max_scalar, p);
            });
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
            grid.vertices().iterate_indices([&](auto const... is) {
              auto const p   = prop.at(is...);
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
            });

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
  //============================================================================
  auto retrieve_typed_prop(auto&& prop, auto&& f) {
    if (prop->type() == typeid(float)) {
      f(prop->template cast_to_typed<float>());
    } else if (prop->type() == typeid(double)) {
      f(prop->template cast_to_typed<double>());
    } else if (prop->type() == typeid(vec2d)) {
      f(prop->template cast_to_typed<vec2d>());
    } else if (prop->type() == typeid(vec2f)) {
      f(prop->template cast_to_typed<vec2f>());
    } else if (prop->type() == typeid(vec3d)) {
      f(prop->template cast_to_typed<vec3d>());
    } else if (prop->type() == typeid(vec3f)) {
      f(prop->template cast_to_typed<vec3f>());
    } else if (prop->type() == typeid(vec4d)) {
      f(prop->template cast_to_typed<vec4d>());
    } else if (prop->type() == typeid(vec4f)) {
      f(prop->template cast_to_typed<vec4f>());
    }
  }
  //----------------------------------------------------------------------------
  auto prop_holds_scalar(auto const& prop) {
    return prop->type() == typeid(float) || prop->type() == typeid(double);
  }
  //----------------------------------------------------------------------------
  auto prop_holds_vector(auto const& prop) {
    return prop->type() == typeid(vec2f) || prop->type() == typeid(vec2d) ||
           prop->type() == typeid(vec3f) || prop->type() == typeid(vec3d) ||
           prop->type() == typeid(vec4f) || prop->type() == typeid(vec4d);
  }
  //----------------------------------------------------------------------------
  auto upload_data_to_texture(auto&& prop, auto&& get_data,
                              renderable_type const& grid) {
    auto texdata = std::vector<GLfloat>{};
    texdata.reserve(grid.vertices().size());

    grid.vertices().iterate_indices(
        [&](auto const... is) { texdata.push_back(get_data(prop, is...)); });
    tex.upload_data(texdata, grid.template size<0>(), grid.template size<1>());
  };
  //----------------------------------------------------------------------------
  auto upload_scalar_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (is_arithmetic<value_type>) {
        upload_data_to_texture(
            prop,
            [](auto const& prop, auto const... is) { return prop(is...); },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_magnitude_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        upload_data_to_texture(
            prop,
            [](auto const& prop, auto const... is) {
              auto mag = typename value_type::value_type{};
              for (std::size_t j = 0; j < value_type::num_components(); ++j) {
                mag += prop(is...)(j) * prop(is...)(j);
              }
              return mag / value_type::num_components();
            },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_x_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        upload_data_to_texture(
            prop,
            [](auto const& prop, auto const... is) { return prop(is...).x(); },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_y_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        upload_data_to_texture(
            prop,
            [](auto const& prop, auto const... is) { return prop(is...).y(); },
            grid);
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_z_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        if constexpr (value_type::num_components() > 2) {
          upload_data_to_texture(
              prop,
              [](auto const& prop, auto const... is) {
                return prop(is...).z();
              },
              grid);
        }
      }
    });
  }
  //----------------------------------------------------------------------------
  auto upload_w_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (static_vec<value_type>) {
        if constexpr (value_type::num_components() > 3) {
          upload_data_to_texture(
              prop,
              [](auto const& prop, auto const... is) {
                return prop(is...).w();
              },
              grid);
        }
      }
    });
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
  auto grid_property_selection(renderable_type const& grid) {
    if (ImGui::BeginCombo("##combo", selected_property_name != nullptr
                                         ? selected_property_name->c_str()
                                         : nullptr)) {
      for (auto const& [name, prop] : grid.vertex_properties()) {
        if (prop->type() == typeid(float) || prop->type() == typeid(double) ||
            prop->type() == typeid(vec2f) || prop->type() == typeid(vec2d) ||
            prop->type() == typeid(vec3f) || prop->type() == typeid(vec3d) ||
            prop->type() == typeid(vec4f) || prop->type() == typeid(vec4d)) {
          auto is_selected = selected_property == prop.get();
          if (ImGui::Selectable(name.c_str(), is_selected)) {
            show_property = true;
            selected_property      = prop.get();
            selected_property_name = &name;
            if (prop_holds_scalar(prop)) {
              upload_scalar_to_texture(selected_property, grid);
              vector_property = false;
            } else if (prop_holds_vector(prop)) {
              upload_magnitude_to_texture(selected_property, grid);

              for (std::size_t i = 0; i < 5; ++i) {
                auto const is_selected =
                    selected_component.at(*selected_property_name) ==
                    vector_component_names[i];
                if (is_selected && i == 0) {
                  upload_magnitude_to_texture(selected_property, grid);
                } else if (is_selected && i == 1) {
                  upload_x_to_texture(selected_property, grid);
                } else if (is_selected && i == 2) {
                  upload_y_to_texture(selected_property, grid);
                } else if (is_selected && i == 3) {
                  upload_z_to_texture(selected_property, grid);
                } else if (is_selected && i == 4) {
                  upload_w_to_texture(selected_property, grid);
                }
              }

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
  //----------------------------------------------------------------------------
  auto vector_component_selection(renderable_type const& grid) {
    if (ImGui::BeginCombo(
            "##combo_vector_component",
            std::string{selected_component.at(*selected_property_name)}
                .c_str())) {
      auto n = std::size_t{};
      retrieve_typed_prop(selected_property, [&](auto&& prop) {
        using prop_type  = std::decay_t<decltype(prop)>;
        using value_type = typename prop_type::value_type;
        if constexpr (static_vec<value_type>) {
          n = value_type::num_components();
        }
      });
      for (std::size_t i = 0; i < n + 1; ++i) {
        auto const is_selected =
            selected_component.at(*selected_property_name) ==
            vector_component_names[i];
        if (ImGui::Selectable(std::string{vector_component_names[i]}.c_str(),
                              is_selected)) {
          selected_component.at(*selected_property_name) =
              vector_component_names[i];
          if (i == 0) {
            upload_magnitude_to_texture(selected_property, grid);
          } else if (i == 1) {
            upload_x_to_texture(selected_property, grid);
          } else if (i == 2) {
            upload_y_to_texture(selected_property, grid);
          } else if (i == 3) {
            upload_z_to_texture(selected_property, grid);
          } else if (i == 4) {
            upload_w_to_texture(selected_property, grid);
          }
        }
      }
      ImGui::EndCombo();
    }
    // ImGui::DragFloat("Min", &min_scalar, 0.01f, -FLT_MAX, max_scalar,
    // "%.06f"); ImGui::DragFloat("Max", &max_scalar, 0.01f, min_scalar,
    // FLT_MAX, "%.06f");
  }
  //----------------------------------------------------------------------------
  auto color_scale_selection(renderable_type const& grid) {
    if (selected_property != nullptr) {
      auto  combo_pos = ImGui::GetCursorScreenPos();
      auto& setting   = settings.at(selected_settings_name());
      if (ImGui::BeginCombo("##combocolor",
                            selected_property_name != nullptr
                                ? selected_property_name->c_str()
                                : nullptr)) {
        ImGui::PushID("##viridis");
        auto viridis_selected =
            ImGui::Selectable("", setting.c == &color_scale::viridis());
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::Image((void*)(std::intptr_t)color_scale::viridis().tex_2d.id(),
                     ImVec2(256, 20));
        if (viridis_selected) {
          setting.c = &color_scale::viridis();
        }

        ImGui::PushID("##GYPi");
        auto gypi_selected =
            ImGui::Selectable("", setting.c == &color_scale::GYPi());
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::Image((void*)(std::intptr_t)color_scale::GYPi().tex_2d.id(),
                     ImVec2(256, 20));
        if (gypi_selected) {
          setting.c = &color_scale::GYPi();
        }
        ImGui::EndCombo();
      }
      ImGui::SameLine();
      ImGui::Text("Invert Color");
      ImGui::SameLine();
      ImGui::ToggleButton("inverted", &setting.scale_inverted);

      auto const  backup_pos = ImGui::GetCursorScreenPos();
      ImGuiStyle& style      = ImGui::GetStyle();
      ImGui::SetCursorScreenPos(
          ImVec2(combo_pos.x + style.FramePadding.x, combo_pos.y));
      ImGui::Image((void*)(std::intptr_t)setting.c->tex_2d.id(),
                   ImVec2(256, 20));
      ImGui::SetCursorScreenPos(backup_pos);

      if (ImGui::Button("Rescale")) {
        rescale_current_property(grid);
      }
    }
  }
  //----------------------------------------------------------------------------
  auto rescale_current_property(renderable_type const& grid) {
    auto const name       = selected_settings_name();
    auto&      setting    = settings[name];
    auto       min_scalar = std::numeric_limits<GLfloat>::max();
    auto       max_scalar = -std::numeric_limits<GLfloat>::max();
    if (prop_holds_scalar(selected_property)) {
      retrieve_typed_prop(selected_property, [&](auto const& prop) {
        using prop_type  = std::decay_t<decltype(prop)>;
        using value_type = typename prop_type::value_type;
        if constexpr (is_arithmetic<value_type>) {
          grid.vertices().iterate_indices([&](auto const... is) {
            auto const p = prop.at(is...);
            min_scalar   = std::min<GLfloat>(min_scalar, p);
            max_scalar   = std::max<GLfloat>(max_scalar, p);
          });
        }
      });
    } else if (prop_holds_vector(selected_property)) {
      retrieve_typed_prop(selected_property, [&](auto const& prop) {
        using prop_type  = std::decay_t<decltype(prop)>;
        using value_type = typename prop_type::value_type;
        if constexpr (static_vec<value_type>) {
          auto constexpr num_comps = value_type::num_components();
          grid.vertices().iterate_indices([&](auto const... is) {
            auto const p = prop.at(is...);
            if (selected_component.at(*selected_property_name) ==
                vector_component_names[0]) {
              auto mag = typename value_type::value_type{};
              for (std::size_t i = 0; i < num_comps; ++i) {
                mag += p(i) * p(i);
              }
              mag        = std::sqrt(mag);
              min_scalar = std::min<GLfloat>(min_scalar, mag);
              max_scalar = std::max<GLfloat>(max_scalar, mag);
            } else {
              auto s = typename value_type::value_type{};
              if (selected_component.at(*selected_property_name) ==
                  vector_component_names[1]) {
                s = p.x();
              } else if (selected_component.at(*selected_property_name) ==
                         vector_component_names[2]) {
                s = p.y();
              } else if (selected_component.at(*selected_property_name) ==
                         vector_component_names[3]) {
                if constexpr (value_type::num_components() > 2) {
                  s = p.z();
                }
              } else if (selected_component.at(*selected_property_name) ==
                         vector_component_names[4]) {
                if constexpr (value_type::num_components() > 3) {
                  s = p.w();
                }
              }
              min_scalar = std::min<GLfloat>(min_scalar, s);
              max_scalar = std::max<GLfloat>(max_scalar, s);
            }
          });
        }
      });
    }
    setting.min_scalar = min_scalar;
    setting.max_scalar = max_scalar;
  }
  //----------------------------------------------------------------------------
  auto properties(renderable_type const& grid) {
    ImGui::Text("Rectilinear Grid");
    ImGui::Checkbox("Show Grid", &show_grid);
    ImGui::Checkbox("Show Property", &show_property);
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Grid Color", grid_color.data());
    grid_property_selection(grid);
    if (selected_property != nullptr && vector_property) {
      vector_component_selection(grid);
    }
    if (selected_property != nullptr) {
      auto& setting = settings[selected_settings_name()];
      ImGui::DragFloat("Min", &setting.min_scalar, 0.01f, -FLT_MAX,
                       setting.max_scalar, "%.06f");
      ImGui::DragFloat("Max", &setting.max_scalar, 0.01f, setting.min_scalar,
                       FLT_MAX, "%.06f");
    }

    color_scale_selection(grid);
  }
  //============================================================================
  auto render() {
    if (show_grid) {
      render_grid();
    }
    if (show_property && selected_property != nullptr) {
      render_property();
    }
  }
  //----------------------------------------------------------------------------
  auto update(auto const dt, renderable_type const& grid,
              camera auto const& cam) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_type;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    if (show_grid) {
      if constexpr (cam_is_float) {
        line_shader::get().set_projection_matrix(cam.projection_matrix());
      } else {
        line_shader::get().set_projection_matrix(
            Mat4<GLfloat>{cam.projection_matrix()});
      }

      if constexpr (cam_is_float) {
        line_shader::get().set_model_view_matrix(cam.view_matrix());
      } else {
        line_shader::get().set_model_view_matrix(
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
        property_shader::get().set_model_view_matrix(
            cam.view_matrix() *
            translation_matrix<GLfloat>(grid.template dimension<0>().front(),
                                        grid.template dimension<1>().front(),
                                        0) *
            scale_matrix<GLfloat>(grid.template extent<0>(),
                                  grid.template extent<1>(), 1));
      } else {
        property_shader::get().set_model_view_matrix(
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
  //----------------------------------------------------------------------------
  auto render_grid() {
    auto& line_shader = line_shader::get();
    line_shader.bind();

    line_shader.set_color(grid_color(0), grid_color(1), grid_color(2),
                          grid_color(3));
    gl::line_width(line_width);
    geometry.draw_lines();
  }
  //----------------------------------------------------------------------------
  auto render_property() {
    property_shader::get().bind();
    tex.bind(0);
    if (selected_property_name != nullptr) {
      auto const name    = selected_settings_name();
      auto&      setting = settings.at(name);
      property_shader::get().set_min(setting.min_scalar);
      property_shader::get().set_max(setting.max_scalar);
      property_shader::get().invert_scale(setting.scale_inverted);
      setting.c->tex.bind(1);
      geometry::get().draw_triangles();
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
