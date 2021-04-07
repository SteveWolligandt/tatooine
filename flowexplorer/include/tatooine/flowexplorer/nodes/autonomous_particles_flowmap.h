#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_H
//==============================================================================
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/triangular_mesh.h>
#include <yavin/indexeddata.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct autonomous_particles_flowmap : renderable<autonomous_particles_flowmap> {
  std::string                                 m_currently_read_path;
  std::unique_ptr<triangular_mesh<double, 2>> m_mesh;
  yavin::indexeddata<vec3f>                   m_edges;
  int                                         m_line_width = 1;
  std::array<GLfloat, 4> m_line_color{0.0f, 0.0f, 0.0f, 1.0f};
  //============================================================================
  autonomous_particles_flowmap(flowexplorer::scene& s)
      : renderable<autonomous_particles_flowmap>{
            "Autonomous Particles Flowmap", s,
            typeid(autonomous_particles_flowmap)} {}
  virtual ~autonomous_particles_flowmap() = default;
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    bool changed = false;
    if (m_currently_read_path.empty()) {
      ImGui::Text("no dataset read");
    } else {
      ImGui::TextUnformatted(m_currently_read_path.c_str());
    }
    if (ImGui::Button("load double gyre autonomous adaptive forward")) {
      load(
          "/home/steve/libs/tatooine2/build/autonomous_particles/"
          "doublegyre_autonomous_forward_flowmap.vtk");
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("load double gyre autonomous adaptive backward")) {
      load(
          "/home/steve/libs/tatooine2/build/autonomous_particles/"
          "doublegyre_autonomous_backward_flowmap.vtk");
      changed = true;
    }
    if (ImGui::Button("load double gyre regular forward")) {
      load(
          "/home/steve/libs/tatooine2/build/autonomous_particles/"
          "doublegyre_regular_forward_flowmap.vtk");
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("load double gyre regular backward")) {
      load(
          "/home/steve/libs/tatooine2/build/autonomous_particles/"
          "doublegyre_regular_backward_flowmap.vtk");
      changed = true;
    }
    changed |= ImGui::SliderInt("line width", &m_line_width, 1, 50);
    changed |= ImGui::ColorEdit4("line color", m_line_color.data());
    return changed;
    return changed;
  }
  auto load(filesystem::path const& path) -> void {
    std::cerr << "loading ... ";
    m_currently_read_path = path;
    m_mesh                = std::make_unique<triangular_mesh<double, 2>>(path);
    {
      m_edges.vertexbuffer().resize(m_mesh->num_vertices());
      auto map = m_edges.vertexbuffer().wmap();
      for (auto v : m_mesh->vertices()) {
        map[v.i] = vec3f{m_mesh->at(v)(0), m_mesh->at(v)(1), 0.0f};
      }
    }
    m_edges.indexbuffer().clear();
    m_edges.indexbuffer().reserve(m_mesh->num_faces() * 6);
    for (auto f : m_mesh->faces()) {
      auto [v0, v1, v2] = m_mesh->at(f);
      m_edges.indexbuffer().push_back(v0.i);
      m_edges.indexbuffer().push_back(v1.i);
      m_edges.indexbuffer().push_back(v1.i);
      m_edges.indexbuffer().push_back(v2.i);
      m_edges.indexbuffer().push_back(v2.i);
      m_edges.indexbuffer().push_back(v0.i);
    }
    std::cerr << "done\n";
  }
  //------------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V) -> void override {
    if (m_edges.indexbuffer().size() > 0) {
      auto& shader = line_shader::get();
      shader.bind();
      shader.set_projection_matrix(P);
      shader.set_modelview_matrix(V);
      yavin::gl::line_width(m_line_width);
      shader.set_color(m_line_color[0], m_line_color[1], m_line_color[2],
                       m_line_color[3]);
      m_edges.draw_lines();
    }
  }
  //----------------------------------------------------------------------------
  bool is_transparent() const override { return m_line_color[3] < 1; }
  //----------------------------------------------------------------------------
  auto data_available() const -> bool { return m_mesh != nullptr; }
  auto mesh() const -> auto const& { return *m_mesh; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::autonomous_particles_flowmap,
    TATOOINE_REFLECTION_INSERT_METHOD(line_width, m_line_width),
    TATOOINE_REFLECTION_INSERT_METHOD(line_color, m_line_color))
#endif
