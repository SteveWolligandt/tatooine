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
template <typename V, typename VReal, typename BTauReal, typename FTauReal,
          typename StepsizeReal, typename CovReal>
void calc(const field<V, VReal, 2, 2>& v, BTauReal btau, FTauReal ftau,
          size_t /*num_its*/, size_t seed_res, StepsizeReal   stepsize,
          CovReal /*desired_coverage*/, const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  steadification  s(v, settings<V>::domain, settings<V>::render_resolution,
                   randeng);
  auto cov_tex = s.make_domain_coverage_tex();


  auto [seedcurve0, t0u00, t0u10] = s.random_seedcurve(0.1, 0.2);
  auto [seedcurve1, t0u01, t0u11] = s.random_seedcurve(0.1, 0.2);
  auto [mesh0, surf0] = s.pathsurface(seedcurve0, t0u00, t0u10, btau, ftau, seed_res, stepsize);
  auto [mesh1, surf1] = s.pathsurface(seedcurve1, t0u01, t0u11, btau, ftau, seed_res, stepsize);
  auto rast0 = s.rasterize(mesh0, cov_tex);
  auto rast1 = s.rasterize(mesh1, cov_tex);
  std::cerr << "single weight 0: " << s.weight(rast0) << '\n';
  std::cerr << "single weight 1: " << s.weight(rast1) << '\n';
  std::cerr << "dual weight: " << s.weight(rast0, rast1) << '\n';
  s.to_curvature_tex(rast0).write_png("curv0.png");
  s.to_curvature_tex(rast1).write_png("curv1.png");

  //grid g{linspace{0.0, 2.0, 3}, linspace{0.0, 1.0, 3}};
  //size_t cnt = 0;
  //for (const auto& mesh : s.integrate_grid_edges(v, g, stepsize)) {
  //  s.to_pos_tex(s.rasterize(mesh, cov_tex))
  //      .write_png("mesh_" + std::to_string(cnt) + ".png");
  //  mesh.write_vtk("mesh_" + std::to_string(cnt++) + ".vtk");
  //}
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
void calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double btau             = argc > 2 ? atof(argv[2]) : -5;
  const double ftau             = argc > 3 ? atof(argv[3]) : 5;
  const size_t num_its          = argc > 4 ? atoi(argv[4]) : 5000;
  const size_t seed_res         = argc > 5 ? atoi(argv[5]) : 2;
  const double stepsize         = argc > 6 ? atof(argv[6]) : 0.1;
  const double desired_coverage = argc > 7 ? atof(argv[7]) : 0.99;
  const auto   seed_str         = argc > 8 ? argv[8] : random_string(10);
  if (argc <= 8) { std::cerr << "seed: " << seed_str << '\n'; }

  calc(v, btau, ftau, num_its, seed_res, stepsize, desired_coverage,
       seed_str);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
auto main(int argc, char** argv) -> int {
  using namespace tatooine;
  using namespace steadification;
  using namespace numerical;
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
