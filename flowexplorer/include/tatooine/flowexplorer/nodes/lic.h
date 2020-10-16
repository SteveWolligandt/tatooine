#ifndef TATOOINE_FLOWEXPLORER_NODES_LIC_H
#define TATOOINE_FLOWEXPLORER_NODES_LIC_H
//==============================================================================
#include <tatooine/flowexplorer/nodes/boundingbox.h>
#include <tatooine/gpu/texture_shader.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/rendering/matrices.h>
#include <yavin>
#include <tatooine/rendering/yavin_interop.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct lic : renderable {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::vectorfield<double, 2>;
  using bb_t          = flowexplorer::nodes::boundingbox<2>;
  //----------------------------------------------------------------------------
  vectorfield_t const*                    m_v           = nullptr;
  bb_t*                                   m_boundingbox = nullptr;
  std::unique_ptr<gpu::texture_shader>    m_shader;
  std::unique_ptr<yavin::tex2rgba<float>> m_lic_tex;
  yavin::indexeddata<vec<float, 2>, vec<float, 2>, float> m_quad;
  vec<int, 2>                                             m_lic_res;
  vec<int, 2>                                             m_vectorfield_sample_res;
  double                                                  m_t;
  int                                                     m_num_samples;
  double                                                  m_stepsize;
  float                                                   m_alpha;
  bool                                                    m_calculating = false;
  //----------------------------------------------------------------------------
  lic(flowexplorer::scene& s)
      : renderable{"LIC", s},
        m_lic_res{100, 100},
        m_vectorfield_sample_res{100, 100},
        m_t{0.0},
        m_num_samples{100},
        m_stepsize{0.001},
        m_alpha{1.0f} {
    this->template insert_input_pin<vectorfield_t>("2D Vector Field");
    this->template insert_input_pin<bb_t>("2D Bounding Box");
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
    m_shader = std::make_unique<gpu::texture_shader>();
  }
  //----------------------------------------------------------------------------
  void render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) override {
    if (m_lic_tex && m_v && m_boundingbox) {
      update_shader(projection_matrix, view_matrix);
      m_shader->bind();
      m_shader->set_alpha(m_alpha);
      m_lic_tex->bind(0);
      m_quad.draw_triangles();
    }
  }
  //----------------------------------------------------------------------------
  void update(const std::chrono::duration<double>&) override {}
  //----------------------------------------------------------------------------
  void draw_ui() override {
    ImGui::DragInt2("LIC Resolution", m_lic_res.data_ptr(), 5, 5, 10000);
    ImGui::DragInt2("Sample Resolution", m_vectorfield_sample_res.data_ptr(), 5,
                    5, 10000);
    ImGui::DragDouble("t", &m_t, 0.1);
    ImGui::DragInt("number of samples", &m_num_samples, 5, 5, 1000000);
    ImGui::DragDouble("stepsize", &m_stepsize, 0.001);
    ImGui::DragFloat("alpha", &m_alpha, 0.1, 0.0f, 1.0f);
  }
  //----------------------------------------------------------------------------
  void calculate_lic();
  //----------------------------------------------------------------------------
  void update_shader(mat<float, 4, 4> const& projection_matrix,
                     mat<float, 4, 4> const& view_matrix) {
    m_shader->set_modelview_matrix(
        view_matrix *
        rendering::translation_matrix<float>(m_boundingbox->min(0),
                                             m_boundingbox->min(1), 0) 
        *
        rendering::scale_matrix<float>(
            m_boundingbox->max(0) - m_boundingbox->min(0),
            m_boundingbox->max(1) - m_boundingbox->min(1), 1)
        );
    m_shader->set_projection_matrix(projection_matrix);
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override {
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
  void on_pin_disconnected(ui::pin& this_pin) override {
    m_lic_tex.reset();
  }
  //----------------------------------------------------------------------------
  bool is_transparent() const override {
    return m_alpha < 1;
  }
  auto serialize() const -> toml::table override {
    toml::table serialization;
    serialization.insert("lic_res", toml::array{m_lic_res(0), m_lic_res(1)});
    serialization.insert("sample_res", toml::array{m_lic_res(0), m_lic_res(1)});

    serialization.insert("t", m_t);
    serialization.insert("num_samples", m_num_samples);
    serialization.insert("stepsize", m_stepsize);
    serialization.insert("alpha", m_alpha);

    return serialization;
  }
  void           deserialize(toml::table const& serialization) override {
    auto const& serialized_lic_res =
        *serialization["lic_res"].as_array();
    m_lic_res(0) = serialized_lic_res[0].as_integer()->get();
    m_lic_res(1) = serialized_lic_res[1].as_integer()->get();
    auto const& serialized_sample_res =
        *serialization["sample_res"].as_array();
    m_vectorfield_sample_res(0) = serialized_sample_res[0].as_integer()->get();
    m_vectorfield_sample_res(1) = serialized_sample_res[1].as_integer()->get();

    m_t           = serialization["t"].as_floating_point()->get();
    m_num_samples = serialization["num_samples"].as_integer()->get();
    m_stepsize    = serialization["stepsize"].as_floating_point()->get();
    m_alpha       = serialization["alpha"].as_floating_point()->get();
  }
  constexpr auto node_type_name() const -> std::string_view override {
    return "lic";
  }
};
REGISTER_NODE(lic);
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
