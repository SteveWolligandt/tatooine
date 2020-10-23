#ifndef TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
#define TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct abcflow : tatooine::analytical::fields::numerical::abcflow<double>,
                 ui::node<abcflow> {
  abcflow(flowexplorer::scene& s) : ui::node<abcflow>{"ABC Flow", s} {
    setup_pins();
  }
  virtual ~abcflow() = default;

 private:
  void setup_pins() {
    this->template insert_output_pin<parent::field<double, 3, 3>>("Field Out");
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::abcflow);
#endif
