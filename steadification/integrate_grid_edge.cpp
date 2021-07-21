#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/streamsurface.h>
#include <tatooine/rectilinear_grid.h>
#include "settings.h"
#include "datasets.h"

namespace tatooine::steadification {
template <typename V, typename Real>
void integrate(const field<V, Real, 2, 2>& v, const vec<Real, 2>& x0, Real t0,
               const vec<Real, 2>& x1, Real t1, Real btau, Real ftau,
               size_t seed_res, Real stepsize, const std::string& outname) {
  std::cerr << "x0: " << x0 << ' ';
  std::cerr << "x1: " << x1 << ' ';
  std::cerr << "t0: " << t0 << ' ';
  std::cerr << "t1: " << t1 << ' ';
  using integrator_t =
      integration::vclibs::rungekutta43<Real, 2, interpolation::hermite>;
  using seedcurve_t = parameterized_line<Real, 2, interpolation::linear>;
  const seedcurve_t seedcurve{{x0, 0}, {x1, 1}};
  streamsurface     ssf{v, t0, t1, seedcurve, integrator_t{}};
  ssf.template discretize<hultquist_discretization>(seed_res, stepsize, btau, ftau)
      .write_vtk(outname);
}

template <typename V, typename Real>
void calc3(const field<V, Real, 2, 2>& v, int argc, char** argv) {
  const Real        u0t0       = argc > 3 ? atof(argv[3]) : 0;
  const Real        u1t0       = argc > 4 ? atof(argv[4]) : 0;
  const Real        btau       = argc > 5 ? atof(argv[5]) : -5;
  const Real        ftau       = argc > 6 ? atof(argv[6]) : 5;
  const size_t      seed_res   = argc > 7 ? atoi(argv[7]) : 2;
  const Real        stepsize   = argc > 8 ? atof(argv[8]) : 0.1;
  const size_t      grid_res_x = argc > 9 ? atoi(argv[9]) : 20;
  const size_t      grid_res_y = argc > 10 ? atoi(argv[10]) : 20;
  const size_t      grid_res_z = argc > 11 ? atoi(argv[11]) : 20;
  const size_t      edge_idx   = argc > 12 ? atoi(argv[12]) : 20;
  const std::string outname   = argv[13];

  using s = settings<V>;
  rectilinear_grid<Real, 3> g{s::domain.add_dimension(u0t0, u1t0),
                  std::array{grid_res_x, grid_res_y, grid_res_z}};
  auto          e  = g.edge_at(edge_idx);
  auto          x30 = e.first.position();
  auto          x31 = e.second.position();
  vec x0{x30(0), x30(1)};
  vec x1{x31(0), x31(1)};
  Real t0 = x30(2);
  Real t1 = x31(2);
  integrate(v, x0, t0, x1, t1, btau, ftau, seed_res, stepsize, outname);
}
template <typename V, typename Real>
void calc2(const field<V, Real, 2, 2>& v, int argc, char** argv) {
  const Real        t0         = atof(argv[3]);
  const Real        btau       = atof(argv[4]);
  const Real        ftau       = atof(argv[5]);
  const size_t      seed_res   = atoi(argv[6]);
  const Real        stepsize   = atof(argv[7]);
  const size_t      grid_res_x = atoi(argv[8]);
  const size_t      grid_res_y = atoi(argv[9]);
  const size_t      edge_idx   = atoi(argv[10]);
  const std::string outname    = argv[11];

  using s = settings<V>;
  rectilinear_grid<Real, 2> g{s::domain, std::array{grid_res_x, grid_res_y}};
  auto          e  = g.edge_at(edge_idx);
  auto          x0 = e.first.position();
  auto          x1 = e.second.position();
  integrate(v, x0, t0, x1, t0, btau, ftau, seed_res, stepsize,outname);
}
}
//------------------------------------------------------------------------------
int main(int argc, char**argv) {
  using namespace tatooine;
  using namespace steadification;
  const std::string v       = argv[1];
  const size_t      griddim = atoi(argv[2]);
  if (griddim < 2 || griddim > 3) {
    throw std::runtime_error{"grid dimension must be 2 or 3"};
  }
  if (v == "dg" || v == "doublegyre") {
    numerical::doublegyre v;
    if (griddim == 2) {
      calc2(v, argc, argv);
    } else if (griddim == 3) {
      calc3(v, argc, argv);
    }
  } else if (v == "la" || v == "laminar") {
    if (griddim == 2) {
      calc2(laminar<double>{}, argc, argv);
    } else if (griddim == 3) {
      calc3(laminar<double>{}, argc, argv);
    }
    //} else if (v == "fdg") {
    //  calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
  } else if (v == "sc" || v == "sinuscosinus") {
    if (griddim == 2) {
      calc2(numerical::sinuscosinus<double>{}, argc, argv);
    } else if (griddim == 3) {
      calc3(numerical::sinuscosinus<double>{}, argc, argv);
    }
    //} else if (v == "cy")  { calc            (cylinder{}, argc, argv);
    // } else if (v == "fw")  { calc        (FlappingWing{}, argc, argv);
    //} else if (v == "mg") {
    //  calc(movinggyre<double>{}, argc, argv);
    // else if (v == "rbc") {
    //  calc(rbc{}, argc, argv);
  } else if (v == "bou") {
    std::cerr << "reading boussinesq... ";
    boussinesq v{dataset_dir + "/boussinesq.am"};
    std::cerr << "done!\n";
    if (griddim == 2) {
      calc2(v, argc, argv);
    } else if (griddim == 3) {
      calc3(v, argc, argv);
    }
  } else if (v == "cav") {
    std::cerr << "reading cavity... ";
    cavity v{};
    std::cerr << "done!\n";
    if (griddim == 2) {
      calc2(v, argc, argv);
    } else if (griddim == 3) {
      calc3(v, argc, argv);
    }
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
