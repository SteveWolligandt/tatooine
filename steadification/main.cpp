#include <fstream>
#include "datasets.h"
#include "settings.h"
#include "random_seed.h"
#include "steadification.h"

//==============================================================================
using namespace std::chrono;
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename VReal, typename T0Real, typename BTauReal,
          typename FTauReal, typename StepsizeReal, typename CovReal>
void calc(const field<V, VReal, 2, 2>& v, T0Real /*t0*/, BTauReal /*btau*/,
          FTauReal /*ftau*/, size_t /*num_its*/, size_t /*seed_res*/,
          StepsizeReal stepsize, CovReal /*desired_coverage*/,
          const std::string& seed_str) {

  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  steadification s(settings<V>::domain, settings<V>::render_resolution, randeng);
  //s.random_domain_filling_streamsurfaces(v, stepsize);
  auto [mesh, surf] = s.pathsurface(v, s.random_seedcurve(0.1, 0.2), stepsize);
  mesh.write_vtk("pathsurface.vtk");
  std::cerr << s.curvature(mesh, surf) << '\n';
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
void calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double t0               = argc > 2 ? atof(argv[2]) : 0;
  const double btau             = argc > 3 ? atof(argv[3]) : -5;
  const double ftau             = argc > 4 ? atof(argv[4]) : 5;
  const size_t num_its          = argc > 5 ? atoi(argv[5]) : 5000;
  const size_t seed_res         = argc > 6 ? atoi(argv[6]) : 2;
  const double stepsize         = argc > 7 ? atof(argv[7]) : 0.1;
  const double desired_coverage = argc > 8 ? atof(argv[8]) : 0.99;
  const auto   seed_str         = argc > 9 ? argv[9] : random_string(10);
  if (argc <= 9) { std::cerr << "seed: " << seed_str << '\n'; }

  calc(v, t0, btau, ftau, num_its, seed_res, stepsize, desired_coverage,
       seed_str);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
auto main(int argc, char** argv) -> int {
  using namespace tatooine;
  const std::string v = argc > 1 ? argv[1] : "dg";
  if (v == "dg") {
    calc(numerical::doublegyre<double>{}, argc, argv);
  //} else if (v == "fdg") {
  //  calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
  //} else if (v == "sc") {
  //  calc(numerical::sinuscosinus<double>{}, argc, argv);
  //} else if (v == "la") {
  //  calc(laminar<double>{}, argc, argv);
  //}
  //// else if (v == "cy")  { calc            (cylinder{}, argc, argv); }
  //// else if (v == "fw")  { calc        (FlappingWing{}, argc, argv); }
  //// else if (v == "mg")  { calc  (movinggyre<double>{}, argc, argv); }
  //else if (v == "rbc") {
  //  calc(rbc{}, argc, argv);
  //} else if (v == "bou") {
  //  calc(boussinesq{dataset_dir + "/boussinesq.am"}, argc, argv);
  //} else if (v == "cav") {
  //  calc(cavity{}, argc, argv);
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
