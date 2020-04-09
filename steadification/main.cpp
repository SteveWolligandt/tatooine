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
void calc(const field<V, VReal, 2, 2>& v, T0Real t0, BTauReal btau,
          FTauReal ftau, size_t seed_res, StepsizeReal stepsize,
          const vec<size_t, 2>& grid_res, CovReal desired_coverage,
          const double neighbor_weight,
          const float penalty,
          const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  constexpr auto  dom = settings<V>::domain;
  steadification  s(v, dom, settings<V>::render_resolution, randeng);
  grid domain{linspace{dom.min(0), dom.max(0), grid_res(0)},
              linspace{dom.min(1), dom.max(1), grid_res(1)}};

   s.greedy_set_cover(domain, t0, btau, ftau, seed_res, stepsize,
   desired_coverage, neighbor_weight, penalty);
}
template <typename V, typename VReal, typename U0T0Real, typename U1T0Real,
          typename BTauReal, typename FTauReal, typename StepsizeReal,
          typename CovReal>
void calc(const field<V, VReal, 2, 2>& v, U0T0Real u0t0, U1T0Real u1t0,
          BTauReal btau, FTauReal ftau, size_t seed_res, StepsizeReal stepsize,
          const vec<size_t, 3>& grid_res, CovReal desired_coverage,
          const double neighbor_weight, const float penalty,
          const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  constexpr auto  dom = settings<V>::domain;
  steadification  s(v, dom, settings<V>::render_resolution, randeng);
  grid domain{linspace{dom.min(0), dom.max(0), grid_res(0)},
              linspace{dom.min(1), dom.max(1), grid_res(1)},
              linspace{u0t0, u1t0, grid_res(2)}};

   s.greedy_set_cover(domain, 0, btau, ftau, seed_res, stepsize,
   desired_coverage, neighbor_weight, penalty);
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
void calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double t0               = argc > 2 ? atof(argv[2]) : 0;
  const double btau             = argc > 3 ? atof(argv[3]) : -5;
  const double ftau             = argc > 4 ? atof(argv[4]) : 5;
  const size_t seed_res         = argc > 5 ? atoi(argv[5]) : 2;
  const double stepsize         = argc > 6 ? atof(argv[6]) : 0.1;
  const size_t grid_res_x       = argc > 7 ? atoi(argv[7]) : 20;
  const size_t grid_res_y       = argc > 8 ? atoi(argv[8]) : 20;
  const double desired_coverage = argc > 9 ? atof(argv[9]) : 0.99;
  const double neighbor_weight  = argc > 10 ? atof(argv[10]) : 1.2;
  const float  penalty          = argc > 11 ? atof(argv[11]) : -2;
  const auto   seed_str         = argc > 12 ? argv[12] : random_string(10);
  if (argc <= 11) { std::cerr << "seed: " << seed_str << '\n'; }

  //calc(v, 0, 10, btau, ftau, seed_res, stepsize, vec{grid_res_x, grid_res_y, 5},
  //     desired_coverage, neighbor_weight, penalty, seed_str);
  calc(v, t0, btau, ftau, seed_res, stepsize, vec{grid_res_x, grid_res_y},
       desired_coverage, neighbor_weight, penalty, seed_str);
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
  } else if (v == "la") {
    calc(laminar<double>{}, argc, argv);
  //} else if (v == "fdg") {
  //  calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
  //} else if (v == "sc") {
  //  calc(numerical::sinuscosinus<double>{}, argc, argv);
    // } else if (v == "cy")  { calc            (cylinder{}, argc, argv);
    // } else if (v == "fw")  { calc        (FlappingWing{}, argc, argv);
  //} else if (v == "mg") {
  //  calc(movinggyre<double>{}, argc, argv);
    // else if (v == "rbc") {
    //  calc(rbc{}, argc, argv);
  } else if (v == "bou") {
    std::cerr << "reading boussinesq... ";
    boussinesq v{dataset_dir + "/boussinesq.am"};
    std::cerr << "done!\n";
    calc(v, argc, argv);
  //} else if (v == "cav") {
  //  std::cerr << "reading cavity... ";
  //  cavity v{};
  //  std::cerr << "done!\n";
  //  calc(v, argc, argv);
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
