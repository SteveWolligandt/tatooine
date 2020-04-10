#ifndef TATOOINE_STEADIFICATION_START_H
#define TATOOINE_STEADIFICATION_START_H
//==============================================================================
#include "steadification.h"
#include "random_seed.h"
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename VReal, typename T0Real, typename BTauReal,
          typename FTauReal, typename StepsizeReal, typename CovReal>
auto calc(const field<V, VReal, 2, 2>& v, T0Real t0, BTauReal btau,
          FTauReal ftau, size_t seed_res, StepsizeReal stepsize,
          const vec<size_t, 2>& grid_res, CovReal desired_coverage,
          const double neighbor_weight, const float penalty,
          const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  constexpr auto  dom = settings<V>::domain;
  steadification  s(v, dom, settings<V>::render_resolution, randeng);
  grid            domain{linspace{dom.min(0), dom.max(0), grid_res(0)},
              linspace{dom.min(1), dom.max(1), grid_res(1)}};

  return s.greedy_set_cover(domain, t0, btau, ftau, seed_res, stepsize,
                            desired_coverage, neighbor_weight, penalty);
}
template <typename V, typename VReal, typename U0T0Real, typename U1T0Real,
          typename BTauReal, typename FTauReal, typename StepsizeReal,
          typename CovReal>
auto calc(const field<V, VReal, 2, 2>& v, U0T0Real u0t0, U1T0Real u1t0,
          BTauReal btau, FTauReal ftau, size_t seed_res, StepsizeReal stepsize,
          const vec<size_t, 3>& grid_res, CovReal desired_coverage,
          const double neighbor_weight, const float penalty,
          const std::string& seed_str) {
  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 randeng{seed};
  constexpr auto  dom = settings<V>::domain;
  steadification  s(v, dom, settings<V>::render_resolution, randeng);
  grid            domain{linspace{dom.min(0), dom.max(0), grid_res(0)},
              linspace{dom.min(1), dom.max(1), grid_res(1)},
              linspace{u0t0, u1t0, grid_res(2)}};

  return s.greedy_set_cover(domain, 0, btau, ftau, seed_res, stepsize,
                            desired_coverage, neighbor_weight, penalty);
}

//------------------------------------------------------------------------------
template <typename V, typename VReal>
auto calc2(const field<V, VReal, 2, 2>& v, const double t0, const double btau,
           const double ftau, const size_t seed_res, const double stepsize,
           const size_t grid_res_x, const size_t grid_res_y,
           const double desired_coverage, const double neighbor_weight,
           const float penalty, const std::string& seed_str) {
  return calc(v, t0, btau, ftau, seed_res, stepsize,
              vec{grid_res_x, grid_res_y}, desired_coverage,
              neighbor_weight, penalty, seed_str);
}
//------------------------------------------------------------------------------
template <typename V, typename VReal>
auto calc2(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double t0               = argc > 3 ? atof(argv[3]) : 0;
  const double btau             = argc > 4 ? atof(argv[4]) : -5;
  const double ftau             = argc > 5 ? atof(argv[5]) : 5;
  const size_t seed_res         = argc > 6 ? atoi(argv[6]) : 2;
  const double stepsize         = argc > 7 ? atof(argv[7]) : 0.1;
  const size_t grid_res_x       = argc > 8 ? atoi(argv[8]) : 20;
  const size_t grid_res_y       = argc > 9 ? atoi(argv[9]) : 20;
  const double desired_coverage = argc > 10 ? atof(argv[10]) : 0.99;
  const double neighbor_weight  = argc > 11 ? atof(argv[11]) : 1.2;
  const float  penalty          = argc > 12 ? atof(argv[12]) : -2;
  const auto   seed_str         = argc > 13 ? argv[13] : random_string(10);
  if (argc < 13) { std::cerr << "seed: " << seed_str << '\n'; }

  return calc(v, t0, btau, ftau, seed_res, stepsize,
              vec{grid_res_x, grid_res_y}, desired_coverage, neighbor_weight,
              penalty, seed_str);
}
//------------------------------------------------------------------------------
template <typename V, typename VReal>
auto calc3(const field<V, VReal, 2, 2>& v, const double u0t0, const double u1t0,
           const double btau, const double ftau, const size_t seed_res,
           const double stepsize, const size_t grid_res_x,
           const size_t grid_res_y, const size_t grid_res_z,
           const double desired_coverage, const double neighbor_weight,
           const float penalty, const std::string& seed_str) {
  return calc(v, u0t0, u1t0, btau, ftau, seed_res, stepsize,
              vec{grid_res_x, grid_res_y, grid_res_z}, desired_coverage,
              neighbor_weight, penalty, seed_str);
}
//------------------------------------------------------------------------------
template <typename V, typename VReal>
auto calc3(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  const double u0t0             = argc > 3 ? atof(argv[3]) : 0;
  const double u1t0             = argc > 4 ? atof(argv[4]) : 0;
  const double btau             = argc > 5 ? atof(argv[5]) : -5;
  const double ftau             = argc > 6 ? atof(argv[6]) : 5;
  const size_t seed_res         = argc > 7 ? atoi(argv[7]) : 2;
  const double stepsize         = argc > 8 ? atof(argv[8]) : 0.1;
  const size_t grid_res_x       = argc > 9 ? atoi(argv[9]) : 20;
  const size_t grid_res_y       = argc > 10 ? atoi(argv[10]) : 20;
  const size_t grid_res_z       = argc > 11 ? atoi(argv[11]) : 20;
  const double desired_coverage = argc > 12 ? atof(argv[12]) : 0.99;
  const double neighbor_weight  = argc > 13 ? atof(argv[13]) : 1.2;
  const float  penalty          = argc > 14 ? atof(argv[14]) : -2;
  const auto   seed_str         = argc > 15 ? argv[15] : random_string(10);
  if (argc < 15) { std::cerr << "seed: " << seed_str << '\n'; }

  return calc(v, u0t0, u1t0, btau, ftau, seed_res, stepsize,
              vec{grid_res_x, grid_res_y, grid_res_z}, desired_coverage,
              neighbor_weight, penalty, seed_str);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
