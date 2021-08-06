#ifndef TATOOINE_FLOWEXPLORER_SAMPLE_TO_GRID_H
#define TATOOINE_FLOWEXPLORER_SAMPLE_TO_GRID_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/visit.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct sample_to_grid : ui::node<sample_to_grid> {
 private:
  std::variant<std::monostate,
               nonuniform_rectilinear_grid<real_t, 2>*,
               nonuniform_rectilinear_grid<real_t, 3>*>
      m_discretized_domain;
  std::variant<std::monostate,
               polymorphic::scalarfield<real_t, 2>*,
               polymorphic::scalarfield<real_t, 3>*,
               polymorphic::vectorfield<real_t, 2>*,
               polymorphic::vectorfield<real_t, 3>*>
                 m_field;
  real_t         m_time = 0.0;
  ui::input_pin* m_discretized_domain_pin;
  ui::input_pin* m_field_pin;

 public:
  sample_to_grid(flowexplorer::scene& s)
      : ui::node<sample_to_grid>{"Field Sampler", s},
        m_discretized_domain_pin{&this->template insert_input_pin<
            nonuniform_rectilinear_grid<real_t, 2>,
            nonuniform_rectilinear_grid<real_t, 3>>("Discretized Domain")},
        m_field_pin{&this->template insert_input_pin<
            polymorphic::scalarfield<real_t, 2>,
            polymorphic::scalarfield<real_t, 3>,
            polymorphic::vectorfield<real_t, 2>,
            polymorphic::vectorfield<real_t, 3>>("Field")} {}
  //----------------------------------------------------------------------------
  virtual ~sample_to_grid() = default;
  //----------------------------------------------------------------------------
  auto time() const -> auto const& { return m_time; }
  auto time() -> auto& { return m_time; }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto set_field(ui::output_pin& other_pin) {
    (
        [&] {
          if (other_pin.type() == typeid(Ts)) {
            m_field = dynamic_cast<Ts*>(&other_pin.get_as<Ts>());
          }
        }(),
        ...);
  }
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& this_pin, ui::output_pin& other_pin)
      -> void {
    if (this_pin == *m_field_pin) {
      set_field<polymorphic::scalarfield<real_t, 2>,
                polymorphic::scalarfield<real_t, 3>,
                polymorphic::vectorfield<real_t, 2>,
                polymorphic::vectorfield<real_t, 3>>(other_pin);
    } else if (this_pin == *m_discretized_domain_pin) {
      if (other_pin.type() == typeid(nonuniform_rectilinear_grid<real_t, 2>)) {
        m_discretized_domain =
            dynamic_cast<nonuniform_rectilinear_grid<real_t, 2>*>(
                &other_pin.node());
      } else if (other_pin.type() ==
                 typeid(nonuniform_rectilinear_grid<real_t, 3>)) {
        m_discretized_domain =
            dynamic_cast<nonuniform_rectilinear_grid<real_t, 3>*>(
                &other_pin.node());
      }
    }
    if (m_discretized_domain.index() > 0 && m_field.index() > 0) {
      sample();
    }
}
  //----------------------------------------------------------------------------
  auto sample() -> void {
    field_domain_do([&](auto& field, auto& domain) {
      using field_t  = std::decay_t<decltype(field)>;
      using domain_t = std::decay_t<decltype(domain)>;
      if constexpr (field_t::num_dimensions() == domain_t::num_dimensions()) {
        if (domain.vertices().size() > 0) {
          discretize(field, domain, this->title(), time());
        }
      }
    });
  }
  //----------------------------------------------------------------------------
  auto on_property_changed() -> void {
    if (m_field.index() > 0 && m_discretized_domain.index() > 0) {
      sample();
    }
  }
  //----------------------------------------------------------------------------
  template <typename F>
  auto domain_do(F&& f) {
    visit(
        [&](auto domain) {
          if constexpr (!is_same<std::monostate,
                                 std::decay_t<decltype(domain)>>) {
            f(*domain);
          }
        },
        m_discretized_domain);
  }
  //----------------------------------------------------------------------------
  template <typename F>
  auto field_do(F&& f) {
    visit(
        [&](auto field) {
          if constexpr (!is_same<std::monostate,
                                 std::decay_t<decltype(field)>>) {
            f(*field);
          }
        },
        m_field);
  }
  //----------------------------------------------------------------------------
  template <typename F>
  auto field_domain_do(F&& f) -> void {
    visit(
        [&](auto field, auto domain) {
          if constexpr (!is_same<std::monostate, decltype(field)> &&
                        !is_same<std::monostate, decltype(domain)>) {
            f(*field, *domain);
          }
        },
        m_field, m_discretized_domain);
  }
  //----------------------------------------------------------------------------
  auto on_title_changed(std::string const& old_title) -> void override {
    if (m_field.index() > 0 && m_discretized_domain.index() > 0) {
      domain_do([&](auto& domain) {
        domain.rename_vertex_property(old_title, this->title());
      });
    }
  }
  //----------------------------------------------------------------------------
  auto on_pin_disconnected(ui::input_pin& pin) -> void {
    domain_do([&](auto& domain) {
      domain.remove_vertex_property(this->title());
    });
    if (&pin == m_field_pin) {
      m_field = std::monostate{};
    } else if (&pin == m_discretized_domain_pin) {
      m_discretized_domain = std::monostate{};
    }
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::sample_to_grid,
    TATOOINE_REFLECTION_INSERT_METHOD(time, time()));
#endif
