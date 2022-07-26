#include <tatooine/analytical/numerical/cylinder_flow.h>
#include <tatooine/gpu/lic.h>

namespace tatooine::analytical {
TEST_CASE() {
  numerical::cylinder_flow.h v;

  gpu::lic(v, linspace{-3.0, -2.0, 2001}, linspace{7.0, 2.0, 1001}, 0.0,
           vec<size_t, 2>{1000, 500}, 30, 0.001)
      .write_png("cylinder_flow_lic.png");
}
}
