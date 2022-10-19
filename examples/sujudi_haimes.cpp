#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/extract_vortex_core_lines.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
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
