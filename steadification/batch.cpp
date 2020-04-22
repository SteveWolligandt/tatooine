#include "datasets.h"
#include "random_seed.h"
#include "settings.h"
#include "start.h"
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
void upload_index(const std::vector<std::string>& working_dirs) {
  std::ofstream index{"index.html"};
  index << "<html><body>\n";
  for (const auto& working_dir : working_dirs) {
    index << "<a href=\"" << working_dir << "\">" << working_dir << "</a>\n";
  }
  index << "</body></html>\n";
  index.close();
  const std::string rsync_cmd =
      "rsync -e ssh -avz index.html "
      "pandafax@diphda.uberspace.de:/home/pandafax/html/reports";
  //system(rsync_cmd.c_str());
}
//------------------------------------------------------------------------------
template <typename V, typename VReal>
void run(std::vector<std::string>&                     working_dirs,
         const vectorfield<V, VReal, 2>&               v,
         const std::vector<std::pair<double, double>>& time_range,
         const std::vector<double>& btaus, const std::vector<double>& ftaus,
         const std::vector<double>&                seed_ress,
         const std::vector<double>&                stepsizes,
         const std::vector<std::array<size_t, 3>>& grid_ress,
         const std::vector<double>&                desired_coverages,
         const std::vector<double>&                neighbor_weights,
         const std::vector<float>& penalties, const std::string& seed_str) {
  const std::string rsync_cmd =
      "rsync -e ssh -avz --exclude '*vtk' " + std::string{settings<V>::name} +
      "* "
      "pandafax@diphda.uberspace.de:/home/pandafax/html/reports";
  for (auto [min_t0, max_t0] : time_ranges)
  for (auto btau : btaus)
  for (auto ftau : ftaus)
  for (auto seed_res : seed_ress)
  for (auto stepsize : stepsizes)
  for (auto [grid_res_x, grid_res_y, grid_res_t] : grid_ress)
  for (auto desired_coverage : desired_coverages)
  for (auto neighbor_weight : neighbor_weights)
  for (auto penalty : penalties) {
    working_dirs.push_back(calc(v, min_t0, max_t0, btau, ftau, seed_res,
                                stepsize, grid_res_x, grid_res_y, grid_res_t,
                                desired_coverage, neighbor_weight, penalty,
                                seed_str));
    // system(rsync_cmd.c_str());
    // upload_index(working_dirs);
  }
}
//------------------------------------------------------------------------------
void batch_doublegyre(std::vector<std::string>& working_dirs,
                const std::string&        random_seed) {
  numerical::doublegyre v;
  run(working_dirs, v, {{0, 10}},  // time_range
      {-5},                        // btau
      {5},                         // ftau
      {2},                         // seedres
      {0.1},                       // stepsize
      {{20, 10, 5}},               // grid_res
      {0.995},                     // coverage
      {1, 1.5, 2, 2.5, 3},         // neighbor weight
      {0},                         // penalty
      random_seed);
}
//------------------------------------------------------------------------------
void batch_sinuscosinus(std::vector<std::string>& working_dirs,
                  const std::string&        random_seed) {
  numerical::sinuscosinus v;
  run(working_dirs, v, {{0, M_PI}},                // time_range
      {-M_PI},               // btau
      {M_PI},                // ftau
      {2},                   // seedres
      {0.1},                 // stepsize
      {{20, 20, 3}},                   // grid_res
      {0.999},               // coverage
      {1, 1.5, 2, 2.5, 3},   // neighbor weight
      {0},                   // penalty
      random_seed);
}
//------------------------------------------------------------------------------
void batch_boussinesq(std::vector<std::string>& working_dirs,
                  const std::string&        random_seed) {
  std::cerr << "reading boussinesq... ";
  boussinesq v{dataset_dir + "/boussinesq.am"};
  std::cerr << "done!\n";
  run(working_dirs, v, {{0, 20}},  // time_range
      {-5},                        // btau
      {5},                         // ftau
      {2},                         // seedres
      {0.1},                       // stepsize
      {{20, 60, 3}},               // grid_res
      {0.99},                      // coverage
      {1, 1.5, 2, 2.5, 3},         // neighbor weight
      {0},                         // penalty
      random_seed);
}
//------------------------------------------------------------------------------
void batch_cavity(std::vector<std::string>& working_dirs,
                  const std::string&        random_seed) {
  std::cerr << "reading cavity... ";
  cavity v;
  std::cerr << "done!\n";
  run(working_dirs, v, {{0, 10}},  // time_range
      {-10},                       // btau
      {10},                        // ftau
      {2},                         // seedres
      {0.1},                       // stepsize
      {{18 * 2, 5 * 2, 3}},        // grid_res
      {0.99},                      // coverage
      {1, 1.5, 2, 2.5, 3},         // neighbor weight
      {0},                         // penalty
      random_seed);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
auto main() -> int {
  yavin::context context{4, 5};
  using namespace tatooine;
  using namespace steadification;
  std::vector<std::string> working_dirs;
  std::string              random_seed = "abcd";
  batch_doublegyre(working_dirs, random_seed);
  batch_sinuscosinus(working_dirs, random_seed);
  batch_boussinesq(working_dirs, random_seed);
  batch_cavity(working_dirs, random_seed);
}
