#include <filesystem>
#include <tatooine/chrono.h>
#include <fstream>
#include "datasets.h"
#include "settings.h"
#include "random_seed.h"
#include "steadification.h"

//==============================================================================
using namespace std::filesystem;
using namespace std::chrono;
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, typename VReal, typename T0Real, typename BTauReal,
          typename FTauReal, typename StepsizeReal, typename CovReal>
void calc(const field<V, VReal, 2, 2>& v, T0Real /*t0*/, BTauReal /*btau*/,
          FTauReal /*ftau*/, size_t /*num_its*/, size_t /*seed_res*/,
          StepsizeReal stepsize, CovReal /*desired_coverage*/,
          std::string seed_str) {

  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  auto p = std::string{settings<V>::name} + "/";
  size_t i = 1;
  while (exists(p)) {
    p = std::string(settings<V>::name) + "_" + std::to_string(i) + "/";
    ++i;
  }
  create_directory(p);
  steadification s(settings<V>::domain, settings<V>::render_resolution, randeng);

  const parameterized_line<VReal, 3> seedcurve{
      {settings<V>::domain.random_point(randeng), 0},
      {settings<V>::domain.random_point(randeng), 1}};
  std::cerr << seedcurve[0].first << '\n';
  std::cerr << seedcurve[1].first << '\n';
  auto rasterized = s.rasterize(v, seedcurve, stepsize);
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
void calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double t0               = argc > 2 ? atof(argv[2]) : 0;
  const double btau             = argc > 3 ? atof(argv[3]) : -5;
  const double ftau             = argc > 4 ? atof(argv[4]) : 5;
  const size_t num_its          = argc > 5 ? atoi(argv[5]) : 100;
  const size_t seed_res         = argc > 6 ? atoi(argv[6]) : 3;
  const double stepsize         = argc > 7 ? atof(argv[7]) : 0.2;
  const double desired_coverage = argc > 8 ? atof(argv[8]) : 0.999;
  const auto   seed_str         = argc > 9 ? argv[9] : random_string(10);
  if (argc <= 9) { std::cerr << "seed: " << seed_str << '\n'; }

  calc(v, t0, btau, ftau, num_its, seed_res, stepsize, desired_coverage,
       seed_str);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
auto main(int argc, char** argv) -> int {
  using namespace tatooine;
  std::string v = argv[1];
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
