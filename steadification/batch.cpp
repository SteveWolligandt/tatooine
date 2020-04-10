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
  system(rsync_cmd.c_str());
}
//------------------------------------------------------------------------------
template <typename V, typename VReal>
void run(std::vector<std::string>&       working_dirs,
         const vectorfield<V, VReal, 2>& v, const std::vector<double>& t0s,
         const std::vector<double>& btaus, const std::vector<double>& ftaus,
         const std::vector<double>& seed_ress,
         const std::vector<double>& stepsizes,
         const std::vector<size_t>& grid_res_xs,
         const std::vector<size_t>& grid_res_ys,
         const std::vector<double>& desired_coverages,
         const std::vector<double>& neighbor_weights,
         const std::vector<float>& penalties, const std::string& seed_str) {
  const std::string rsync_cmd =
      "rsync -e ssh -avz --exclude '*vtk' " + std::string{settings<V>::name} +
      "* "
      "pandafax@diphda.uberspace.de:/home/pandafax/html/reports";
  for (auto t0 : t0s) {
    for (auto btau : btaus) {
      for (auto ftau : ftaus) {
        for (auto seed_res : seed_ress) {
          for (auto stepsize : stepsizes) {
            for (auto grid_res_x : grid_res_xs) {
              for (auto grid_res_y : grid_res_ys) {
                for (auto desired_coverage : desired_coverages) {
                  for (auto neighbor_weight : neighbor_weights) {
                    for (auto penalty : penalties) {
                      working_dirs.push_back(
                          calc2(v, t0, btau, ftau, seed_res, stepsize,
                                grid_res_x, grid_res_y, desired_coverage,
                                neighbor_weight, penalty, seed_str));
                      system(rsync_cmd.c_str());
                      upload_index(working_dirs);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
void doublegyre(std::vector<std::string>& working_dirs,
                const std::string&        random_seed) {
  numerical::doublegyre v;
  run(working_dirs, v, {0},               // t0
      {-5},                               // btau
      {5},                                // ftau
      {2},                                // seedres
      {0.1},                              // stepsize
      {20},                               // grid_res_x
      {10},                               // grid_res_y
      {0.999},                            // coverage
      {1, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0},  // neighbor weight
      {-1, -2, -3},                       // penalty
      random_seed);
}
//------------------------------------------------------------------------------
void sinuscosinus(std::vector<std::string>& working_dirs,
                  const std::string&        random_seed) {
  numerical::sinuscosinus v;
  run(working_dirs, v, {0},               // t0
      {-M_PI},                            // btau
      {M_PI},                             // ftau
      {2},                                // seedres
      {0.1},                              // stepsize
      {20},                               // grid_res_x
      {20},                               // grid_res_y
      {0.999},                             // coverage
      {1, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0},  // neighbor weight
      {-1, -2, -3},                       // penalty
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
  doublegyre(working_dirs, random_seed);
  sinuscosinus(working_dirs, random_seed);
}
