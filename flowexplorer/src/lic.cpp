#include <tatooine/flowexplorer/nodes/lic.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
lic::lic(flowexplorer::scene& s)
    : renderable<lic>{"LIC", s},
      m_shader{std::make_unique<gpu::texture_shader>()},
      m_lic_res{100, 100},
      m_vectorfield_sample_res{100, 100},
      m_t{0.0},
      m_num_samples{100},
      m_stepsize{0.001},
      m_alpha{1.0f} {
  init();
}
//----------------------------------------------------------------------------
lic::lic(lic const& other)
    : renderable<lic>{other},
      m_shader{std::make_unique<gpu::texture_shader>()},
      m_lic_res{other.m_lic_res},
      m_vectorfield_sample_res{other.m_vectorfield_sample_res},
      m_t{other.m_t},
      m_num_samples{other.m_num_samples},
      m_stepsize{other.m_stepsize},
      m_alpha{other.m_alpha} {
  init();
}
//==============================================================================
auto lic::init() -> void {
  setup_pins();
  setup_quad();
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
  if (m_lic_tex && m_v && m_boundingbox) {
    update_shader(projection_matrix, view_matrix);
    m_shader->bind();
    m_shader->set_alpha(m_alpha);
    m_lic_tex->bind(0);
    m_quad.draw_triangles();
  }
}
//----------------------------------------------------------------------------
void lic::calculate_lic() {
  if (m_calculating) {
    return;
  }
  m_calculating = true;
  this->scene().window().do_async([node = this] {
    node->m_lic_tex = std::make_unique<yavin::tex2rgba<float>>(gpu::lic(
        *node->m_v,
        linspace{node->m_boundingbox->min(0), node->m_boundingbox->max(0),
                 static_cast<size_t>(node->m_vectorfield_sample_res(0))},
        linspace{node->m_boundingbox->min(1), node->m_boundingbox->max(1),
                 static_cast<size_t>(node->m_vectorfield_sample_res(1))},
        node->m_t, vec<size_t, 2>{node->m_lic_res(0), node->m_lic_res(1)},
        node->m_num_samples, node->m_stepsize));

    node->m_calculating = false;
  });
}
//----------------------------------------------------------------------------
auto lic::draw_ui() -> void {
  ImGui::DragInt2("LIC Resolution", m_lic_res.data_ptr(), 5, 5, 10000);
  ImGui::DragInt2("Sample Resolution", m_vectorfield_sample_res.data_ptr(), 5,
                  5, 10000);
  ImGui::DragDouble("t", &m_t, 0.1);
  //ImGui::DragInt("number of samples", &m_num_samples, 5, 5, 1000000);
  ImGui::DragDouble("stepsize", &m_stepsize, 0.001);
  ImGui::DragFloat("alpha", &m_alpha, 0.1, 0.0f, 1.0f);

   namespace hana = boost::hana;
   hana::for_each(*this, [](auto const& pair) {
    auto const& var_name = hana::to<char const*>(hana::first(pair));
    auto&       value    = hana::second(pair);
    using T              = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<int, T>) {
      ImGui::DragInt(var_name, &const_cast<std::decay_t<T>&>(value), 0.1,
      0.0f,
                     1.0f);
    //} else if constexpr (std::is_same_v<float, T>) {
    //  ImGui::DragFloat(var_name, &const_cast<std::decay_t<T>&>(value), 0.1,
    //                   0.0f, 1.0f);
    //} else if constexpr (std::is_same_v<double, T>) {
    //  ImGui::DragDouble(var_name, &const_cast<std::decay_t<T>&>(value), 0.1,
    //                   0.0f, 1.0f);
    }
  });
}
//----------------------------------------------------------------------------
auto lic::update_shader(mat<float, 4, 4> const& projection_matrix,
                        mat<float, 4, 4> const& view_matrix) -> void {
  m_shader->set_modelview_matrix(
      view_matrix *
      rendering::translation_matrix<float>(m_boundingbox->min(0),
                                           m_boundingbox->min(1), 0) *
      rendering::scale_matrix<float>(
          m_boundingbox->max(0) - m_boundingbox->min(0),
          m_boundingbox->max(1) - m_boundingbox->min(1), 1));
  m_shader->set_projection_matrix(projection_matrix);
}
//----------------------------------------------------------------------------
auto lic::on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) -> void {
  if (other_pin.type() == typeid(bb_t)) {
    m_boundingbox = dynamic_cast<bb_t*>(&other_pin.node());
  } else if ((other_pin.type() == typeid(vectorfield_t))) {
    m_v = dynamic_cast<vectorfield_t*>(&other_pin.node());
  }
  if (m_boundingbox != nullptr && m_v != nullptr) {
    calculate_lic();
  }
}
//----------------------------------------------------------------------------
auto lic::on_pin_disconnected(ui::pin& this_pin) -> void {
  m_lic_tex.reset();
}
//----------------------------------------------------------------------------
auto lic::is_transparent() const -> bool{
  return m_alpha < 1;
}
//----------------------------------------------------------------------------
auto lic::serialize() const -> toml::table {
  toml::table serialized_node;
  namespace hana = boost::hana;
  hana::for_each(*this, [&serialized_node](auto const& pair) {
    auto const& var_name = hana::to<char const*>(hana::first(pair));
    auto const& value    = hana::second(pair);
    using T              = std::decay_t<decltype(value)>;
    if constexpr (std::is_arithmetic_v<T>) {
      serialized_node.insert(var_name, value);
    } else if constexpr (std::is_arithmetic_v<T>) {
    }
  });
  return serialized_node;
}
//----------------------------------------------------------------------------
void lic::deserialize(toml::table const& serialization) {
  auto const& serialized_lic_res    = *serialization["lic_res"].as_array();
  m_lic_res(0)                      = serialized_lic_res[0].as_integer()->get();
  m_lic_res(1)                      = serialized_lic_res[1].as_integer()->get();
  auto const& serialized_sample_res = *serialization["sample_res"].as_array();
  m_vectorfield_sample_res(0) = serialized_sample_res[0].as_integer()->get();
  m_vectorfield_sample_res(1) = serialized_sample_res[1].as_integer()->get();

  m_t           = serialization["t"].as_floating_point()->get();
  m_num_samples = serialization["num_samples"].as_integer()->get();
  m_stepsize    = serialization["stepsize"].as_floating_point()->get();
  m_alpha       = serialization["alpha"].as_floating_point()->get();
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
