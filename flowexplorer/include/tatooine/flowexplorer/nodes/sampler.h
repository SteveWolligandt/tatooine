#ifndef TATOOINE_FLOWEXPLORER_SAMPLER_H
#define TATOOINE_FLOWEXPLORER_SAMPLER_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes{
//==============================================================================
template <size_t N>
struct sampler : ui::node<sampler<N>> {
 private:
  nonuniform_rectilinear_grid<real_t, N>* m_discretized_domain = nullptr;
  polymorphic::vectorfield<real_t, 2>*    m_field              = nullptr;
  real_t                                  m_time               = 0.0;
  ui::input_pin*                          m_discretized_domain_pin;
  ui::input_pin*                          m_field_pin;

 public:
  sampler(flowexplorer::scene& s)
      : ui::node<sampler<N>>{"Field Sampler", s},
        m_discretized_domain_pin{&this->template insert_input_pin<
            nonuniform_rectilinear_grid<real_t, N>>("Discretized Domain")},
        m_field_pin{&this->template insert_input_pin<
            polymorphic::vectorfield<real_t, 2>>("2D Vector Field")} {}
  //----------------------------------------------------------------------------
  virtual ~sampler() = default;
  //----------------------------------------------------------------------------
  auto time() const -> auto const& { return m_time; }
  auto time() -> auto& { return m_time; }
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& /*this_pin*/, ui::output_pin& other_pin)
      -> void {
    if (other_pin.type() == typeid(polymorphic::vectorfield<real_t, 2>)) {
      m_field =
          dynamic_cast<polymorphic::vectorfield<real_t, 2>*>(&other_pin.node());
    } else if ((other_pin.type() ==
                typeid(nonuniform_rectilinear_grid<real_t, N>))) {
      m_discretized_domain =
          dynamic_cast<nonuniform_rectilinear_grid<real_t, N>*>(
              &other_pin.node());
    }
    if (m_discretized_domain != nullptr && m_field != nullptr) {
      sample();
    }
  }
  //----------------------------------------------------------------------------
  auto sample() -> void {
    discretize(*m_field, *m_discretized_domain, this->title(), time());
  }
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void {
    if (m_field != nullptr && m_discretized_domain != nullptr) {
      sample();
    }
  }
  //----------------------------------------------------------------------------
  auto on_title_changed(std::string const& old_title) -> void override {
    if (m_field != nullptr && m_discretized_domain != nullptr) {
      m_discretized_domain->rename_vertex_property(old_title, this->title());
    }
  }
  //----------------------------------------------------------------------------
  auto on_pin_disconnected(ui::input_pin& pin) -> void {
    if (m_discretized_domain != nullptr && m_field != nullptr) {
      m_discretized_domain->remove_vertex_property(this->title());
    }
    if (&pin == m_field_pin) {
      m_field = nullptr;
    } else if (&pin == m_discretized_domain_pin) {
      m_discretized_domain = nullptr;
    }
  }
};
using sampler_2 = sampler<2>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::sampler_2,
    TATOOINE_REFLECTION_INSERT_METHOD(time, time()));
#endif
