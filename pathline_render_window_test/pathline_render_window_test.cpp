#include <tatooine/abcflow.h>
#include <tatooine/doublegyre.h>
#include <tatooine/duffing_oscillator.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/gpu/pathline_render_window.h>

template <typename V, typename Real, size_t N>
void work(int argc, char** argv, const tatooine::field<V, Real, N, N>& v,
          const tatooine::boundingbox<Real, N>& seedarea) {
  using namespace tatooine;
  const size_t num_pathlines = argc > 2 ? atoi(argv[2]) : 100;
  const double btau          = argc > 3 ? atof(argv[3]) : -10;
  const double ftau          = argc > 4 ? atof(argv[4]) : 10;
  gpu::pathline_render_window w{v, seedarea, num_pathlines, btau, ftau};
}
int main(int argc, char** argv) {
  using namespace tatooine;
  const std::string fieldname = argv[1];
  if (fieldname == "dg") {
    numerical::doublegyre v2;
    spacetime_field       v{v2};
    work(argc, argv, v,
         boundingbox<double, 3>{vec{0.0, 0.0, 0.0}, vec{2.0, 1.0, 0.0}});
  } else if (fieldname == "duffing") {
    numerical::duffing_oscillator v2{0.5, 0.5, 0.5};
    spacetime_field               v{v2};
    work(
        argc, argv, v,
        boundingbox<double, 3> {vec{0.0, 0.0, 0.0}, vec{2.0, 1.0, 0.0}});
  } else if (fieldname == "abc") {
    numerical::abcflow v;
    work(argc, argv, v,
         boundingbox<double, 3>{vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}});
  }
}
