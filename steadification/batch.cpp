#include "datasets.h"
#include "random_seed.h"
#include "settings.h"
#include "start.h"
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename VReal>
void run(std::vector<std::string>&                     working_dirs,
         const vectorfield<V, VReal, 2>&               v,
         const std::vector<std::pair<double, double>>& time_ranges,
         const std::vector<std::vector<double>>&       t0ss,
         const std::vector<double>& btaus, const std::vector<double>& ftaus,
         const std::vector<double>&                seed_ress,
         const std::vector<double>&                stepsizes,
         const std::vector<std::array<size_t, 2>>& grid_ress,
         const std::vector<double>&                desired_coverages,
         const std::vector<double>&                neighbor_weights,
         const std::vector<float>&                 penalties,
         const std::vector<float>&                 max_curvatures,
         const std::vector<bool>&                  use_taus,
         const std::vector<bool>& norm_weights, const std::string& seed_str) {
  for (auto [min_t, max_t] : time_ranges)
  for (auto t0s : t0ss)
  for (auto btau : btaus)
  for (auto ftau : ftaus)
  for (auto seed_res : seed_ress)
  for (auto stepsize : stepsizes)
  for (auto [grid_res_x, grid_res_y] : grid_ress)
  for (auto desired_coverage : desired_coverages)
  for (auto neighbor_weight : neighbor_weights)
  for (auto penalty : penalties) 
  for (auto max_curvature : max_curvatures) 
  for (auto use_tau : use_taus) 
  for (auto normalize_weight : norm_weights) {
    working_dirs.push_back(
        calc(v, min_t, max_t, t0s, btau, ftau, seed_res, stepsize, grid_res_x,
             grid_res_y, desired_coverage, neighbor_weight, penalty,
             max_curvature, use_tau, normalize_weight, seed_str));
  }
}
//------------------------------------------------------------------------------
void batch_doublegyre(std::vector<std::string> & working_dirs,
                      const std::string& random_seed) {
  numerical::doublegyre v;
  run(working_dirs, v, {{0, 10}},  // time_range
      {{5}, {0, 5, 10}},           // t0s
      {-10},                       // btau
      {10},                        // ftau
      {2},                         // seedres
      {0.1},                       // stepsize
      {{21, 11}, {41, 21}},        // grid_res
      {0.995},                     // coverage
      {1, 1.5, 2},                 // neighbor weight
      {0},                         // penalty
      {5, 10},                     // max_curvature
      {true},                      // use_tau
      {true, false},               // normalize_weight
      random_seed);
}
//------------------------------------------------------------------------------
void batch_sinuscosinus(std::vector<std::string>& working_dirs,
                  const std::string&        random_seed) {
  numerical::sinuscosinus v;
  run(working_dirs, v, {{0, M_PI * 2}},  // time_range
      {{M_PI}, /*{0, M_PI, 2 * M_PI}*/}, // t0s
      {-M_PI * 2},                       // btau
      {M_PI * 2},                        // ftau
      {2},                               // seedres
      {0.1},                             // stepsize
      {{21, 21}},                        // grid_res
      {0.999},                           // coverage
      {1, 1.5, 2},                       // neighbor weight
      {0},                               // penalty
      {5, 10},                           // max_curvature
      {true},                            // use_tau
      {true, false},                     // normalize_weight
      random_seed);
}
//------------------------------------------------------------------------------
void batch_boussinesq(std::vector<std::string>& working_dirs,
                      const std::string&        random_seed) {
  std::cerr << "reading boussinesq... ";
  boussinesq v{dataset_dir + "/boussinesq.am"};
  std::cerr << "done!\n";
  run(working_dirs, v, {{10, 20}},  // time_range
      {/*{15}, */{10, 15, 20}},         // t0s
      {-10},                        // btau
      {10},                         // ftau
      {2},                          // seedres
      {0.1},                        // stepsize
      {{20, 60}},                   // grid_res
      {0.99},                       // coverage
      {1, 1.5, 2},                  // neighbor weight
      {0},                          // penalty
      {5, 10},                      // max_curvature
      {true},                       // use_tau
      {true, false},                // normalize_weight
      random_seed);
}
//------------------------------------------------------------------------------
void batch_cavity(std::vector<std::string>& working_dirs,
                  const std::string&        random_seed) {
  std::cerr << "reading cavity... ";
  cavity v;
  std::cerr << "done!\n";
  run(working_dirs, v, {{0, 10}},  // time_range
      {{5}, {0, 5, 10}},           // t0s
      {-10},                       // btau
      {10},                        // ftau
      {2},                         // seedres
      {0.1},                       // stepsize
      {{18 * 2, 5 * 2}},           // grid_res
      {0.99},                      // coverage
      {1, 1.5, 2},                 // neighbor weight
      {0},                         // penalty
      {5, 10},                     // max_curvature
      {true, false},               // use_tau
      {true, false},               // normalize_weight
      random_seed);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
auto main() -> int {
  yavin::context context{4, 5};
  using namespace tatooine::steadification;
  std::vector<std::string> working_dirs;
  std::string              random_seed = "abcd";
  batch_cavity(working_dirs, random_seed);
  batch_boussinesq(working_dirs, random_seed);
  //batch_doublegyre(working_dirs, random_seed);
  //batch_sinuscosinus(working_dirs, random_seed);
}
