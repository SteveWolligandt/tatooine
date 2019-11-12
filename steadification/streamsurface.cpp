#include <Tatooine/streamsurface.h>
#include <Tatooine/spacetime_vectorfield.h>
#include "datasets.h"

void bou() {
  Boussinesq bou;
  tatooine::SpacetimeVectorfield bou_st{bou};

  double t0 = 10;
  double btau = 0, ftau = 5;
  double x0 = -0.1, y0 = 0, x1 = 0.1, y1 = 0;
  size_t seed_res = 10;
  double stepwidth = 0.01;

  // tatooine::Streamsurface ssf {bou, t0,
  //                              {{{x0, y0}, 0},
  //                               {{x1, y1}, 1}}};
  tatooine::Streamsurface ssf_st {bou_st, 0,
                               {{{x0,y0, t0}, 0},
                                {{x1,y1, t0}, 1}}};

  // ssf.integrator().cache().set_max_memory_usage(1024*1024*2);
  ssf_st.integrator().cache().set_max_memory_usage(1024*1024*2);

  // ssf.discretize(seed_res, stepwidth, btau, ftau).write_vtk("Boussinesq_ssf.vtk");
  ssf_st.discretize(seed_res, stepwidth, btau, ftau).write_vtk("Boussinesq_ssf_st.vtk");
}

void dg() {
  tatooine::analytical::DoubleGyre dg;
  tatooine::FixedTimeVectorfield fdg{dg,0};

  double btau = -10, ftau = 10;
  double x0 = 0.99, y0 = 0.1, x1 = 1.01, y1 = 0.1;
  size_t seed_res = 100;
  double stepwidth = 0.1;

  tatooine::Streamsurface ssf {fdg, 0,
                               {{{x0, y0}, 0},
                                {{x1, y1}, 1}}};

  ssf.discretize(seed_res, stepwidth, btau, ftau).write_vtk("fdg_ssf.vtk");
}

int main () {
  // dg();
  bou();
}
