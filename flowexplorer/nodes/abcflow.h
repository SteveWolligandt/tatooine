#ifndef TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
#define TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct abcflow : tatooine::analytical::fields::numerical::abcflow<Real>, ui::node {
  abcflow() : ui::node{"ABC Flow"} {
    this->template insert_output_pin<parent::field<Real, 3, 3>>("Field Out");
  }
  void draw_ui() override {
    ui::node::draw_ui([this] {});
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
