#include <Tatooine/doublegyre.h>
#include "steadification.h"
#include "datasets.h"
#include "real_t.h"

//==============================================================================
using solution_t = Steadification::solution_t;

//==============================================================================
template <typename vf_t, typename settings = settings_t<vf_t>>
void lic_test(real_t t0, real_t tau, unsigned int num_segments,
              real_t stepwidth, const std::string& seed_str) {
  vf_t            vf;
  std::seed_seq   seed(seed_str.begin(), seed_str.end());
  std::mt19937_64 random_engine(seed);
  auto vertex_to_tuple = [tau](auto v) { return std::tuple{v, -tau, tau}; };
  Steadification steadification(settings::domain, settings::render_resolution,
                                t0, tau, 5, stepwidth);
  solution_t     sol;
  boost::transform(
      steadification.grid.random_vertex_sequence(num_segments, random_engine),
      std::back_inserter(sol), vertex_to_tuple);

  steadification.to_vectorfield_tex(vf, sol).save_png("vectorfield.png");
  steadification.to_tau_color_scale(vf, sol).save_png("tau_color_scale.png");
  steadification.to_lic(vf, sol, 100, true).save_png("lic_test.png");
  auto ribbons = steadification.ribbons(vf, sol);
  for (size_t i = 0; i < ribbons.size(); ++i)
    ribbons[i].write_vtk("ribbon" + std::to_string(i) + ".vtk");
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string vf_name;
  real_t       t0           = 0;
  real_t       tau          = 10;
  unsigned int num_segments = 10;
  real_t stepwidth  = 0.05;
  std::string seed_str = "abc";

  if (argc > 1) vf_name = argv[1];
  if (argc > 2) t0 = atof(argv[2]);
  if (argc > 3) tau = atof(argv[3]);
  if (argc > 4) num_segments = atoi(argv[4]);
  if (argc > 5) stepwidth = atof(argv[5]);
  if (argc > 6) seed_str = argv[6];

  if (vf_name == "dg")
    lic_test<tatooine::analytical::DoubleGyre<real_t>>(t0, tau, num_segments,
                                                       stepwidth,  seed_str);
  else if (vf_name == "sc")
    lic_test<tatooine::analytical::SinusCosinus<real_t>>(t0, tau, num_segments,
                                                         stepwidth,  seed_str);
  else if (vf_name == "mg")
    lic_test<MovingGyre>(t0, tau, num_segments, stepwidth, seed_str);
}

