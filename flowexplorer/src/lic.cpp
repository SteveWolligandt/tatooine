#include <tatooine/flowexplorer/nodes/lic.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes{
//==============================================================================
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
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
