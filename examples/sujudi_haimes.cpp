#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/analytical/numerical/tornado.h>
#include <tatooine/vortex_core_line_extraction.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto calc_abc_flow() {
  auto  v        = analytical::numerical::abcflow{};
  auto  g        = rectilinear_grid{linspace{-3.0, 3.0 + 1e-6, 100},
                            linspace{-3.0, 3.0 + 1e-5, 100},
                            linspace{-3.0, 3.0 + 1e-4, 100}};
  auto& v_disc   = g.sample_to_vertex_property(v, "velocity");
  auto& J_disc   = g.sample_to_vertex_property(diff(v_disc), "velocity_diff");
   auto const vortex_core_lines =
       extract_vortex_core_lines(v_disc, J_disc, algorithm::sujudi_haimes);
   write(vortex_core_lines, "sujudi_haimes_abc.vtp");
   g.write("sujudi_haimes_abc.vtr");
}
//==============================================================================
auto calc_tornado_flow() {
  auto  v        = analytical::numerical::tornado{};
  auto  g        = rectilinear_grid{linspace{-1.0, 1.0 + 1e-6, 300},
                            linspace{-1.0, 1.0 + 1e-5, 300},
                            linspace{0.0, 1.0 + 1e-4, 150}};
  auto& v_disc   = g.sample_to_vertex_property(v, "velocity");
  auto& J_disc   = g.sample_to_vertex_property(diff(v_disc), "velocity_diff");
   auto const vortex_core_lines =
       extract_vortex_core_lines(v_disc, J_disc, algorithm::sujudi_haimes);
   write(vortex_core_lines, "sujudi_haimes_tornado.vtp");
   g.write("sujudi_haimes_tornado.vtr");
}
//==============================================================================
auto main() -> int {
  calc_abc_flow();
  calc_tornado_flow();
}
