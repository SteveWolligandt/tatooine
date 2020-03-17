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
          size_t seed_res, StepsizeReal stepsize,
          const vec<size_t, 2>& grid_res, CovReal desired_coverage,
          const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  constexpr auto  dom = settings<V>::domain;
  steadification  s(v, dom, settings<V>::render_resolution,
                   randeng);

  grid  g{linspace{dom.min(0), dom.max(0), grid_res(0)},
          linspace{dom.min(1), dom.max(1), grid_res(1)}};
  s.greedy_set_cover(g, 0, btau, ftau, seed_res, stepsize, desired_coverage);
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
void calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double btau             = argc > 2 ? atof(argv[2]) : -5;
  const double ftau             = argc > 3 ? atof(argv[3]) : 5;
  const size_t seed_res         = argc > 4 ? atoi(argv[4]) : 2;
  const double stepsize         = argc > 5 ? atof(argv[5]) : 0.1;
  const size_t grid_res_x       = argc > 6 ? atoi(argv[6]) : 20;
  const size_t grid_res_y       = argc > 7 ? atoi(argv[7]) : 20;
  const double desired_coverage = argc > 8 ? atof(argv[8]) : 0.99;
  const auto   seed_str         = argc > 9 ? argv[9] : random_string(10);
  if (argc <= 7) { std::cerr << "seed: " << seed_str << '\n'; }

  calc(v, btau, ftau, seed_res, stepsize, vec{grid_res_x, grid_res_y},
       desired_coverage, seed_str);
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
  } else if (v == "fdg") {
    calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
    //} else if (v == "sc") {
    //  calc(numerical::sinuscosinus<double>{}, argc, argv);
  } else if (v == "la") {
    calc(laminar<double>{}, argc, argv);
    // } else if (v == "cy")  { calc            (cylinder{}, argc, argv);
    // } else if (v == "fw")  { calc        (FlappingWing{}, argc, argv);
  //} else if (v == "mg") {
  //  calc(movinggyre<double>{}, argc, argv);
    // else if (v == "rbc") {
    //  calc(rbc{}, argc, argv);
  } else if (v == "bou") {
    calc(boussinesq{dataset_dir + "/boussinesq.am"}, argc, argv);
    //} else if (v == "cav") {
    //  calc(cavity{}, argc, argv);
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
