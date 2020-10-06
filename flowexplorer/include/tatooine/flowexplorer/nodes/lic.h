#ifndef TATOOINE_FLOWEXPLORER_NODES_LIC_H
#define TATOOINE_FLOWEXPLORER_NODES_LIC_H
//==============================================================================
#include <tatooine/boundingbox.h>
#include <tatooine/gpu/texture_shader.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/rendering/matrices.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct lic : renderable {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::vectorfield<Real, 2>;
  using bb_t          = boundingbox<Real, 2>;
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
  lic(flowexplorer::window& w)
      : renderable{w, "LIC"},
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
  void calculate_lic() {
    if (m_calculating) {
      return;
    }
    m_calculating = true;
    this->window().do_async([node = this] {
      node->m_lic_tex = std::make_unique<yavin::tex2rgba<float>>(
        gpu::lic(
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
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
