#ifndef TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_H
#define TATOOINE_FLOWEXPLORER_NODES_AUTONOMOUS_PARTICLES_FLOWMAP_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/triangular_mesh.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes{
//==============================================================================
struct autonomous_particles_flowmap : ui::node<autonomous_particles_flowmap> {
  std::unique_ptr<triangular_mesh<double, 2>> m_mesh;
  //============================================================================
  autonomous_particles_flowmap(flowexplorer::scene& s)
      : ui::node<autonomous_particles_flowmap>{"Autonomous Particles Flowmap",
                                               s} {
    this->template insert_output_pin<autonomous_particles_flowmap>("Out");
  }
  virtual ~autonomous_particles_flowmap() = default;
  //----------------------------------------------------------------------------
  auto draw_properties() -> bool override {
    bool changed = false;
    if (ImGui::Button("load double gyre")) {
      std::cerr << "loading ... ";
      m_mesh = std::make_unique<triangular_mesh<double, 2>>(
          "/home/steve/libs/tatooine2/build/autonomous_particles/"
          "doublegyre_flowmap.vtk");
      std::cerr << "done\n";
      changed = true;
    }
    return changed;
  }
  //------------------------------------------------------------------------------
  auto mesh_available() const -> bool { return m_mesh != nullptr; }
  auto mesh() const -> auto const& { return *m_mesh; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::autonomous_particles_flowmap)
#endif
