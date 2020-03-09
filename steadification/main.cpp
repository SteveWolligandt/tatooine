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
          CovReal desired_coverage, const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  steadification  s(v, settings<V>::domain, settings<V>::render_resolution,
                   randeng);

  VReal eps = 1e-5;
  grid  g{linspace{eps, 2.0 - eps, 21}, linspace{eps, 1.0 - eps, 11}};
  s.greedy_set_cover(g, 0, btau, ftau, seed_res, stepsize, desired_coverage);
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
    // else if (v == "rbc") {
    //  calc(rbc{}, argc, argv);
    //} else if (v == "bou") {
    //  calc(boussinesq{dataset_dir + "/boussinesq.am"}, argc, argv);
    //} else if (v == "cav") {
    //  calc(cavity{}, argc, argv);
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
