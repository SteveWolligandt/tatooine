#ifndef TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
#define TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct doublegyre : tatooine::analytical::fields::numerical::doublegyre<Real>,
                    ui::node {
  doublegyre(scene const& s) : ui::node{"Double Gyre", s} {
    this->set_infinite_domain(true);
    this->template insert_output_pin<parent::vectorfield<Real, 2>>("Field Out");
  }
  virtual ~doublegyre() = default;
  auto serialize() const -> toml::table override {
    return toml::table{};
  }
  void deserialize(toml::table const& serialized_data) override {
    
  }
  constexpr auto node_type_name() const -> std::string_view override {
    return "doublegyre";
  }
};
//REGISTER_NODE(doublegyre<double>);
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
