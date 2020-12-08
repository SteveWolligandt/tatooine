#ifndef TATOOINE_FLOWEXPLORER_NODES_FIELD_FROM_FILE_H
#define TATOOINE_FLOWEXPLORER_NODES_FIELD_FROM_FILE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/sampled_grid_property_field.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <real_number Real, size_t N, bool is_time_dependent,
          size_t... TensorDims>
struct field_from_file
    : sampled_grid_property_field_creator_t<Real, N, is_time_dependent,
                                            TensorDims...>,
      ui::node<field_from_file<Real, N, is_time_dependent, TensorDims...>> {
  //============================================================================
 private:
  std::string m_path =
      "/home/steve/flows/2DCavity/Cavity2DTimeFilter3x3x7_100_bin.am";

  //============================================================================
 public:
  auto path() const -> auto const& { return m_path; }
  auto path() -> auto & { return m_path; }
  //============================================================================
 public:
  field_from_file(flowexplorer::scene& s)
      : ui::node<field_from_file>{"Field", s} {
    this->template insert_output_pin<parent::field<double, N, TensorDims...>>(
        "Field Out");
  }
  //----------------------------------------------------------------------------
  virtual ~field_from_file() = default;
  //============================================================================
  auto draw_properties() -> bool override {
    auto changed = ImGui::InputText("path", &m_path);
    if (ImGui::Button("read")) {
      this->read(m_path);
      changed = true;
    }
    return changed;
  }
};
//==============================================================================
template <real_number Real, size_t N, size_t... TensorDims>
using unsteady_field_from_file = field_from_file<Real, N, true, TensorDims...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number Real, size_t N, size_t... TensorDims>
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
