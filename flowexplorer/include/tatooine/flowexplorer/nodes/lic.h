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
//namespace tatooine::flowexplorer::nodes {struct lic ;}
//namespace boost::hana {template <>struct accessors_impl<tatooine::flowexplorer::nodes::lic>;}
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct lic : renderable<lic> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using vectorfield_t = parent::vectorfield<double, 2>;
  using bb_t          = flowexplorer::nodes::boundingbox<2>;

 private:
  //----------------------------------------------------------------------------
  vectorfield_t const*                    m_v           = nullptr;
  bb_t*                                   m_boundingbox = nullptr;
  bool                                    m_calculating = false;
  std::unique_ptr<gpu::texture_shader>    m_shader;
  std::unique_ptr<yavin::tex2rgba<float>> m_lic_tex;
  yavin::indexeddata<vec<float, 2>, vec<float, 2>, float> m_quad;
  vec<int, 2>                                             m_lic_res;
  vec<int, 2> m_vectorfield_sample_res;
  double      m_t;
  int         m_num_samples;
  double      m_stepsize;
  float       m_alpha;

 public:
  //----------------------------------------------------------------------------
  lic(flowexplorer::scene& s);
  //----------------------------------------------------------------------------
  lic(lic const& other);
  //============================================================================
  auto init() -> void;
  //----------------------------------------------------------------------------
  auto setup_pins() -> void;
  //----------------------------------------------------------------------------
  auto setup_quad() -> void;
  //----------------------------------------------------------------------------
  auto render(mat<float, 4, 4> const& projection_matrix,
              mat<float, 4, 4> const& view_matrix) -> void override;
  //----------------------------------------------------------------------------
  auto update(const std::chrono::duration<double>&) -> void override {}
  //----------------------------------------------------------------------------
  auto draw_ui() -> void override;
  //----------------------------------------------------------------------------
  auto calculate_lic() -> void;
  //----------------------------------------------------------------------------
  auto update_shader(mat<float, 4, 4> const& projection_matrix,
                     mat<float, 4, 4> const& view_matrix) -> void;
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override ;
  //----------------------------------------------------------------------------
  void on_pin_disconnected(ui::pin& this_pin) override ;
  //----------------------------------------------------------------------------
  auto           is_transparent() const -> bool override;
  auto           serialize() const -> toml::table override;
  auto           deserialize(toml::table const& serialization) -> void override;
  constexpr auto node_type_name() const -> std::string_view override {
    return "lic";
  }
  //============================================================================
  auto lic_res() -> auto& {
    return m_lic_res;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto lic_res() const -> auto const& {
    return m_lic_res;
  }
  //----------------------------------------------------------------------------
  auto vectorfield_sample_res() -> auto& {
    return m_vectorfield_sample_res;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vectorfield_sample_res() const -> auto const& {
    return m_vectorfield_sample_res;
  }
  //----------------------------------------------------------------------------
  auto t() -> auto& {
    return m_t;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto t() const {
    return m_t;
  }
  //----------------------------------------------------------------------------
  auto num_samples() -> auto& {
    return m_num_samples;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto num_samples() const {
    return m_num_samples;
  }
  //----------------------------------------------------------------------------
  auto stepsize() -> auto& {
    return m_stepsize;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto stepsize() const {
    return m_stepsize;
  }
  //----------------------------------------------------------------------------
  auto alpha() -> auto& {
    return m_alpha;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto alpha() const {
    return m_alpha;
  }
};
REGISTER_NODE(lic);
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
BEGIN_META_NODE(tatooine::flowexplorer::nodes::lic)
  //META_NODE_ACCESSOR(lic_res, lic_res()),
  //META_NODE_ACCESSOR(vectorfield_sample_res, vectorfield_sample_res()),
  META_NODE_ACCESSOR(t, t())
  //META_NODE_ACCESSOR(num_samples, num_samples()),
  //META_NODE_ACCESSOR(stepsize, stepsize()),
  //META_NODE_ACCESSOR(alpha, alpha())
END_META_NODE()
#endif
