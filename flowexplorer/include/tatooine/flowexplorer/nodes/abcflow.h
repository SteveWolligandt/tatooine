#ifndef TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
#define TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct abcflow : tatooine::analytical::fields::numerical::abcflow<double>,
                 ui::node {
  abcflow(scene const& s) : ui::node{"ABC Flow", s} {
    setup_pins();
  }
  abcflow(scene const& s, toml::table const& serialized_data) : ui::node{s} {
    setup_pins();
    deserialize(serialized_data);
  }
  virtual ~abcflow() = default;

 private:
  void setup_pins() {
    this->template insert_output_pin<parent::field<double, 3, 3>>("Field Out");
  }

 public:
  auto serialize() const -> toml::table override {
    return toml::table{};
  }
  void           deserialize(toml::table const&) override {}
  constexpr auto node_type_name() const -> std::string_view override {
    return "abcflow";
  }
};
REGISTER_NODE(abcflow);
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
