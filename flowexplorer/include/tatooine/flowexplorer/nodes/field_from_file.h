#ifndef TATOOINE_FLOWEXPLORER_NODES_FIELD_FROM_FILE_H
#define TATOOINE_FLOWEXPLORER_NODES_FIELD_FROM_FILE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/sampled_grid_property_field.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <arithmetic Real, size_t N, bool is_time_dependent,
          size_t... TensorDims>
struct field_from_file
    : sampled_grid_property_field_creator_t<Real, N, is_time_dependent,
                                            TensorDims...>,
      ui::node<field_from_file<Real, N, is_time_dependent, TensorDims...>> {
  //============================================================================
  using this_t = field_from_file<Real, N, is_time_dependent, TensorDims...>;
  using node_parent_t = ui::node<this_t>;
  //============================================================================
 private:
  std::string m_path;
  bool        m_picking_file = false;

  //============================================================================
 public:
  auto path() const -> auto const& { return m_path; }
  auto path() -> auto & { return m_path; }

  //============================================================================
 public:
  field_from_file(flowexplorer::scene& s)
      : ui::node<field_from_file>{"Field", s} {
    this->template insert_output_pin<polymorphic::field<double, N, TensorDims...>>(
        "Field Out", *this);
  }
  //----------------------------------------------------------------------------
  virtual ~field_from_file() = default;
  //============================================================================
  auto draw_properties() -> bool override {
    auto& win = this->scene().window();
    if (!win.file_explorer_is_opened() && ImGui::Button("read")) {
      m_picking_file = true;
      win.open_file_explorer("Load File", {".am", ".vtk"}, *this);
    }
    bool changed = false;
    return changed;
  }
  //----------------------------------------------------------------------------
  auto on_path_selected(std::string const& path) -> void override {
    auto& win = this->scene().window();
    m_path    = path;
    std::cerr << m_path << '\n';
    this->read(m_path);
    win.close_file_explorer();
  }
  //----------------------------------------------------------------------------
  auto deserialize(toml::table const& serialized_node) -> void override {
    node_parent_t::deserialize(serialized_node);
    if (!m_path.empty()) {
      std::cerr << "read: " << m_path << '\n';
      this->read(m_path);
    }
  }
};
//==============================================================================
template <arithmetic Real, size_t N, size_t... TensorDims>
using unsteady_field_from_file = field_from_file<Real, N, true, TensorDims...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic Real, size_t N, size_t... TensorDims>
using steady_field_from_file = field_from_file<Real, N, false, TensorDims...>;
//==============================================================================
using steady_scalarfield_2d   = steady_field_from_file<double, 2>;
using steady_scalarfield_3d   = steady_field_from_file<double, 3>;
using unsteady_scalarfield_2d = unsteady_field_from_file<double, 2>;
using unsteady_scalarfield_3d = unsteady_field_from_file<double, 3>;
//------------------------------------------------------------------------------
using steady_vectorfield_2d   = steady_field_from_file<double, 2, 2>;
using steady_vectorfield_3d   = steady_field_from_file<double, 3, 3>;
using unsteady_vectorfield_2d = unsteady_field_from_file<double, 2, 2>;
using unsteady_vectorfield_3d = unsteady_field_from_file<double, 3, 3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::steady_scalarfield_2d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::steady_scalarfield_3d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::unsteady_scalarfield_2d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::unsteady_scalarfield_3d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::steady_vectorfield_2d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::steady_vectorfield_3d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::unsteady_vectorfield_2d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::unsteady_vectorfield_3d,
    TATOOINE_REFLECTION_INSERT_GETTER(path));
#endif
