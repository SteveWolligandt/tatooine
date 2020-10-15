#ifndef TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
#define TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct abcflow : tatooine::analytical::fields::numerical::abcflow<Real>,
                 ui::node<abcflow<Real>> {
  constexpr auto node_type_name() const -> std::string_view override {
    return "abcflow";
  }
  abcflow() : ui::node<abcflow<Real>>{"ABC Flow"} {
    this->template insert_output_pin<parent::field<Real, 3, 3>>("Field Out");
  }
  virtual ~abcflow() = default;
  void serialize() override {}
  void deserialize() override {}
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
