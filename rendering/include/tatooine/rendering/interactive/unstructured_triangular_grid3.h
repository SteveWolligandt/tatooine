#ifndef TATOOINE_RENDERING_INTERACTIVE_UNSTRUCTURED_TRIANGULAR_GRID3_H
#define TATOOINE_RENDERING_INTERACTIVE_UNSTRUCTURED_TRIANGULAR_GRID3_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/unstructured_triangular_grid.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/cook_torrance_brdf_shader.h>
#include <tatooine/rendering/interactive/color_scale.h>
#include <tatooine/rendering/interactive/renderer.h>
#include <tatooine/rendering/interactive/shaders.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <floating_point Real>
struct renderer<tatooine::unstructured_triangular_grid<Real, 3>> {
  static constexpr std::array<std::string_view, 5> vector_component_names = {
      "magnitude", "x", "y", "z", "w"};
  using renderable_type =
      tatooine::unstructured_simplicial_grid<Real, 3, 2>;
  template <typename T>
  using typed_vertex_property_interface_type =
      typename renderable_type::typed_vertex_property_type<T>;
  //============================================================================
  using property_shader = cook_torrance_brdf_shader;
  //============================================================================
 private:
  struct property_settings {
    color_scale* c              = nullptr;
    GLfloat      min_scalar     = std::numeric_limits<GLfloat>::max();
    GLfloat      max_scalar     = -std::numeric_limits<GLfloat>::max();
    bool         scale_inverted = false;
  };
  bool                                               show_property = false;
  int                                                line_width    = 1;
  Vec4<GLfloat>                                      wireframe_color = {0, 0, 0, 1};
  std::unordered_map<std::string, property_settings> settings;
  std::unordered_map<std::string, std::string_view>  selected_component;
  std::string const* selected_property_name = nullptr;
  typename renderable_type::vertex_property_type const* selected_property =
      nullptr;

  bool vector_property = false;

  gl::vertexbuffer<Vec3<GLfloat>, Vec3<GLfloat>, GLfloat> m_geometry;
  gl::indexbuffer                                         m_triangles;

 public:
  //============================================================================
  renderer(renderable_type const& grid) {
    init_grid_geometry(grid);
    init_properties(grid);
  }
  //----------------------------------------------------------------------------
  auto init_grid_geometry(renderable_type const& grid) {
    auto normals = std::vector<Vec3<GLfloat>>(grid.vertices().size(),
                                              Vec3<GLfloat>::zeros());
    for (auto const t : grid.simplices()) {
      auto const [v0, v1, v2] = grid[t];
      auto const n            = cross(grid[v1] - grid[v0], grid[v2] - grid[v0]);
      normals[v0.index()] += n;
      normals[v1.index()] += n;
      normals[v2.index()] += n;
    }

    m_geometry.resize(grid.vertices().size());
    m_triangles.resize(grid.simplices().size() * 3);
    {
      auto data = m_geometry.wmap();
      auto k    = std::size_t{};
      for (auto const v : grid.vertices()) {
        get<0>(data[k]) = Vec3<GLfloat>{grid[v]};
        get<1>(data[k]) = Vec3<GLfloat>{normals[v.index()]};
        ++k;
      }
    }
    {
      auto data = m_triangles.wmap();
      auto k    = std::size_t{};
      for (auto const s : grid.simplices()) {
        auto const [v0, v1, v2] = grid[s];
        data[k++] = v0.index();
        data[k++] = v1.index();
        data[k++] = v2.index();
      }
    }
  }
  //----------------------------------------------------------------------------
  auto init_properties(renderable_type const& grid) {
    for (auto const& [name, prop] : grid.vertex_properties()) {
      if (prop_holds_scalar(prop)) {
        auto min_scalar = std::numeric_limits<GLfloat>::max();
        auto max_scalar = -std::numeric_limits<GLfloat>::max();
        retrieve_typed_prop(prop.get(), [&](auto const& prop) {
          using prop_type  = std::decay_t<decltype(prop)>;
          using value_type = typename prop_type::value_type;
          if constexpr (is_arithmetic<value_type>) {
            for (auto const v : grid.vertices()) {
              auto const p = prop[v.index()];
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
            for (auto const v : grid.vertices()) {
              auto const p   = prop[v.index()];
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
  auto upload_data(auto&& prop, auto&& get_data, renderable_type const& grid) {
    auto data    = m_geometry.rwmap();
    for (std::size_t i = 0; i < grid.vertices().size(); ++i) {
      get<2>(data[i]) = get_data(prop, i);
    }
  };
  //----------------------------------------------------------------------------
  auto upload_scalar_to_texture(auto&& prop, renderable_type const& grid) {
    retrieve_typed_prop(prop, [&](auto&& prop) {
      using prop_type  = std::decay_t<decltype(prop)>;
      using value_type = typename prop_type::value_type;
      if constexpr (is_arithmetic<value_type>) {
        upload_data(
            prop,
            [](auto const& prop, auto const i) { return prop[i]; },
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
        upload_data(
            prop,
            [](auto const& prop, auto const i) {
              auto mag = typename value_type::value_type{};
              for (std::size_t j = 0; j < value_type::num_components(); ++j) {
                mag += prop[i](j) * prop[i](j);
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
        upload_data(
            prop,
            [](auto const& prop, auto const i) { return prop[i].x(); },
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
        upload_data(
            prop,
            [](auto const& prop, auto const i) { return prop[i].y(); },
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
          upload_data(
              prop,
              [](auto const& prop, auto const i) {
                return prop[i].z();
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
          upload_data(
              prop,
              [](auto const& prop, auto const i) {
                return prop[i].w();
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
          for (auto const v :grid.vertices()) {
            auto const p = prop[v.index()];
            min_scalar   = std::min<GLfloat>(min_scalar, p);
            max_scalar   = std::max<GLfloat>(max_scalar, p);
          }
        }
      });
    } else if (prop_holds_vector(selected_property)) {
      retrieve_typed_prop(selected_property, [&](auto const& prop) {
        using prop_type  = std::decay_t<decltype(prop)>;
        using value_type = typename prop_type::value_type;
        if constexpr (static_vec<value_type>) {
          auto constexpr num_comps = value_type::num_components();
          for (auto const v : grid.vertices()) {
            auto const p = prop[v.index()];
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
          }
        }
      });
    }
    setting.min_scalar = min_scalar;
    setting.max_scalar = max_scalar;
  }
  //----------------------------------------------------------------------------
  auto properties(renderable_type const& grid) {
    //ImGui::Text("Triangular Grid");
    ImGui::Checkbox("Show Property", &show_property);
    ImGui::DragInt("Line width", &line_width, 1, 1, 20);
    ImGui::ColorEdit4("Wireframe Color", wireframe_color.data());
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
    if (show_property && selected_property != nullptr) {
      render_property();
    }
  }
  //----------------------------------------------------------------------------
  auto update(auto const dt, renderable_type const& grid,
              camera auto const& cam) {
    using CamReal = typename std::decay_t<decltype(cam)>::real_type;
    static auto constexpr cam_is_float = is_same<GLfloat, CamReal>;
    if (show_property) {
      if constexpr (cam_is_float) {
        property_shader::get().set_projection_matrix(cam.projection_matrix());
      } else {
        property_shader::get().set_projection_matrix(
            Mat4<GLfloat>{cam.projection_matrix()});
      }

      if constexpr (cam_is_float) {
        property_shader::get().set_view_matrix(cam.view_matrix());
      } else {
        property_shader::get().set_view_matrix(
            Mat4<GLfloat>{cam.view_matrix()});
      }
      property_shader::get().set_camera_position(cam.eye());
    }
  }
  //----------------------------------------------------------------------------
  auto render_property() {
    if (selected_property_name != nullptr) {
      auto const name    = selected_settings_name();
      auto&      setting = settings.at(name);
      setting.c->tex.bind(0);
      //property_shader::get().set_min(setting.min_scalar);
      //property_shader::get().set_max(setting.max_scalar);
      //property_shader::get().invert_scale(setting.scale_inverted);
      property_shader::get().bind();
      auto vao = gl::vertexarray{};
      vao.bind();
      m_geometry.bind();
      m_geometry.activate_attributes();
      m_triangles.bind();
      vao.draw_triangles(m_triangles.size());
    }
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
