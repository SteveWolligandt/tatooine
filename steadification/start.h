#ifndef TATOOINE_STEADIFICATION_START_H
#define TATOOINE_STEADIFICATION_START_H
//==============================================================================
#include <boost/lexical_cast.hpp>
#include "random_seed.h"
#include "steadification.h"
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename VReal, typename MinTReal, typename MaxTReal,
          typename BTauReal, typename FTauReal, typename StepsizeReal,
          typename CovReal>
auto calc(const field<V, VReal, 2, 2>& v, const MinTReal min_t,
          const MaxTReal max_t, BTauReal btau, FTauReal ftau, size_t seed_res,
          StepsizeReal stepsize, const vec<size_t, 2>& grid_res,
          CovReal desired_coverage, const double neighbor_weight,
          const float penalty, const float max_curvature, const bool use_tau,
          const bool normalize_weight, const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  constexpr auto  dom = settings<V>::domain;
  steadification  s(v, dom, settings<V>::render_resolution, randeng);
  grid            domain{linspace{dom.min(0), dom.max(0), grid_res(0)},
                         linspace{dom.min(1), dom.max(1), grid_res(1)}};
  return s.greedy_set_cover(domain, min_t, max_t, btau, ftau, seed_res,
                            stepsize, desired_coverage, neighbor_weight,
                            penalty, max_curvature, use_tau, normalize_weight);
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
auto calc(const field<V, VReal, 2, 2>& v, const double min_t,
          const double max_t, const double btau, const double ftau,
          const size_t seed_res, const double stepsize, const size_t grid_res_x,
          const size_t grid_res_y, const double desired_coverage,
          const double neighbor_weight, const float penalty,
          const float max_curvature, const bool use_tau,
          const bool normalize_weight, const std::string& seed_str) {
  return calc(v, min_t, max_t, btau, ftau, seed_res, stepsize,
              vec{grid_res_x, grid_res_y}, desired_coverage, neighbor_weight,
              penalty, max_curvature, use_tau, normalize_weight, seed_str);
}
//------------------------------------------------------------------------------
template <typename V, typename VReal>
auto calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double min_t            = argc > 2 ? atof(argv[2]) : 0;
  const double max_t            = argc > 3 ? atof(argv[3]) : 0;
  const double btau             = argc > 4 ? atof(argv[4]) : -5;
  const double ftau             = argc > 5 ? atof(argv[5]) : 5;
  const size_t seed_res         = argc > 6 ? atoi(argv[6]) : 2;
  const double stepsize         = argc > 7 ? atof(argv[7]) : 0.1;
  const size_t grid_res_x       = argc > 8 ? atoi(argv[8]) : 20;
  const size_t grid_res_y       = argc > 9 ? atoi(argv[9]) : 20;
  const double desired_coverage = argc > 10 ? atof(argv[10]) : 0.99;
  const double neighbor_weight  = argc > 11 ? atof(argv[11]) : 1.2;
  const float  penalty          = argc > 12 ? atof(argv[12]) : -2;
  const float  max_curvature    = argc > 13 ? atof(argv[13]) : 10;
  const bool   use_tau = argc > 14 ? boost::lexical_cast<bool>(argv[14]) : true;
  const bool   normalize_weight =
      argc > 15 ? boost::lexical_cast<bool>(argv[15]) : true;
  const auto seed_str = argc > 16 ? argv[16] : random_string(10);
  if (argc < 15) { std::cerr << "seed: " << seed_str << '\n'; }

  return calc(v, min_t, max_t, btau, ftau, seed_res, stepsize,
              vec{grid_res_x, grid_res_y}, desired_coverage, neighbor_weight,
              penalty, max_curvature, use_tau, normalize_weight, seed_str);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
