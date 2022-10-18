#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/extract_vortex_core_lines.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v      = analytical::numerical::abcflow{};
  auto g      = rectilinear_grid{linspace{-3.0, 3.0 + 1e-6, 100},
                            linspace{-3.0, 3.0 + 1e-5, 100},
                            linspace{-3.0, 3.0 + 1e-4, 100}};
  auto& v_disc = g.sample_to_vertex_property(v, "velocity");
  write(extract_vortex_core_lines(v_disc, algorithm::sujudi_haimes),
        "sujudi_haimes_abc.vtp");
}
