#ifndef TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
#define TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include "../renderable.h"
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct abcflow : tatooine::analytical::fields::numerical::abcflow<Real>, renderable {
  abcflow(struct window& w) : renderable{w, "ABC Flow"} {
    this->template insert_output_pin<parent::field<Real, 3, 3>>("Field Out");
  }
  void render(const yavin::mat4&, const yavin::mat4&) override {}
  void draw_ui() override {
    ui::node::draw_ui([this] {});
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif