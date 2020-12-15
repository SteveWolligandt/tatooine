#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/window.h>
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#include <tatooine/flowexplorer/nodes/lic.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
lic::lic(flowexplorer::scene& s)
    : renderable<lic>{"LIC", s},
      m_shader{std::make_unique<gpu::texture_shader>()},
      m_lic_res{100, 100},
      m_num_samples{100},
      m_stepsize{0.001},
      m_alpha{1.0f} {
  init();
}
//==============================================================================
auto lic::init() -> void {
  setup_pins();
  setup_quad();
}
//----------------------------------------------------------------------------
auto lic::write_png() -> void {
  if (m_lic_tex) {
    std::stringstream str;
    std::string       type_name{
        dynamic_cast<ui::base::node const*>(m_v)->type_name()};
    auto const last_colon_pos = type_name.find_last_of(':');
    type_name                 = type_name.substr(last_colon_pos + 1,
                                 size(type_name) - last_colon_pos - 1);

    str << "lic_" << type_name << "_x_" << m_bb->min(0) << "_" << m_bb->max(0)
        << "_y_" << m_bb->min(1) << "_" << m_bb->max(1) << "_t_" << m_v->time()
        << ".png";
    m_lic_tex->write_png(str.str());
  }
}
//----------------------------------------------------------------------------
auto lic::setup_pins() -> void {
  insert_input_pin<vectorfield_t>("2D Vector Field");
  insert_input_pin<bb_t>("2D Bounding Box");
}
//----------------------------------------------------------------------------
auto lic::setup_quad() -> void {
  m_quad.vertexbuffer().resize(4);
  m_quad.vertexbuffer()[0] = {vec{0.0f, 0.0f}, vec{0.0f, 0.0f}};
  m_quad.vertexbuffer()[1] = {vec{1.0f, 0.0f}, vec{1.0f, 0.0f}};
  m_quad.vertexbuffer()[2] = {vec{0.0f, 1.0f}, vec{0.0f, 1.0f}};
  m_quad.vertexbuffer()[3] = {vec{1.0f, 1.0f}, vec{1.0f, 1.0f}};

  m_quad.indexbuffer().resize(6);
  m_quad.indexbuffer()[0] = 0;
  m_quad.indexbuffer()[1] = 1;
  m_quad.indexbuffer()[2] = 2;

  m_quad.indexbuffer()[3] = 1;
  m_quad.indexbuffer()[4] = 3;
  m_quad.indexbuffer()[5] = 2;
}
//----------------------------------------------------------------------------
auto lic::render(mat<float, 4, 4> const& projection_matrix,
                 mat<float, 4, 4> const& view_matrix) -> void {
  if (m_lic_tex && m_v && m_bb) {
    update_shader(projection_matrix, view_matrix);
    m_shader->bind();
    m_shader->set_alpha(m_alpha);
    {
      std::lock_guard lock{m_mutex};
      m_lic_tex->bind(0);
    }
    m_quad.draw_triangles();
  }
}
//----------------------------------------------------------------------------
void lic::calculate_lic() {
  if (m_calculating) {
    return;
  }
  m_calculating = true;
  this->scene().window().do_async([this] {
    auto tex = gpu::lic(
        *m_v,
        uniform_grid<real_t, 2>{
            linspace{m_bb->min(0), m_bb->max(0),
                     static_cast<size_t>(m_v->resolution()(0))},
            linspace{m_bb->min(1), m_bb->max(1),
                     static_cast<size_t>(m_v->resolution()(1))}},
        vec<size_t, 2>{m_lic_res(0), m_lic_res(1)}, m_num_samples, m_stepsize);

    std::lock_guard lock{m_mutex};
    m_lic_tex     = std::make_unique<yavin::tex2rgba<float>>(std::move(tex));
    m_calculating = false;
  });
}
//----------------------------------------------------------------------------
auto lic::update_shader(mat<float, 4, 4> const& projection_matrix,
                        mat<float, 4, 4> const& view_matrix) -> void {
  m_shader->set_modelview_matrix(
      view_matrix *
      rendering::translation_matrix<float>(m_bb->min(0), m_bb->min(1), 0) *
      rendering::scale_matrix<float>(m_bb->max(0) - m_bb->min(0),
                                     m_bb->max(1) - m_bb->min(1), 1));
  m_shader->set_projection_matrix(projection_matrix);
}
//----------------------------------------------------------------------------
auto lic::on_pin_connected(ui::input_pin& /*this_pin*/,
                           ui::output_pin& other_pin) -> void {
  if (other_pin.type() == typeid(bb_t)) {
    m_bb = dynamic_cast<bb_t*>(&other_pin.node());
  } else if ((other_pin.type() == typeid(vectorfield_t))) {
    m_v = dynamic_cast<vectorfield_t*>(&other_pin.node());
  }
  if (m_bb != nullptr && m_v != nullptr) {
    calculate_lic();
  }
}
//----------------------------------------------------------------------------
auto lic::on_property_changed() -> void {
  if (m_v != nullptr && m_bb != nullptr) {
    calculate_lic();
  }
}
//----------------------------------------------------------------------------
auto lic::on_pin_disconnected(ui::input_pin & /*this_pin*/) -> void {
  m_lic_tex.reset();
}
//----------------------------------------------------------------------------
auto lic::is_transparent() const -> bool { return m_alpha < 1; }
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
