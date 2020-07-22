#include "ensemble_member.h"
#include "ensemble_file_paths.h"
#include "integrate_pathline.h"

#include <filesystem>
#include <mutex>
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
std::mutex mutex;
//==============================================================================
auto create_grid() {
  V    v{ensemble_file_paths.front()};
  auto     dim0 = v.xc_axis;
  auto     dim1 = v.yc_axis;
  linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};

  g.dimension<0>().pop_front();
  g.dimension<0>().pop_back();
  g.dimension<0>().pop_back();

  g.dimension<1>().pop_front();
  g.dimension<1>().pop_back();
  g.dimension<1>().pop_back();

  g.dimension<2>().pop_front();
  g.dimension<2>().pop_back();
  g.dimension<2>().pop_back();
  return g;
}
//------------------------------------------------------------------------------
template <typename UncertainEddyProp>
void uncertain_eddy_detection(V const& v, typename V::pos_t const& x,
                              real_number auto const t,
                              real_number auto const threshold,
                              UncertainEddyProp&     uncertain_eddy,
                              integral auto const... is) {
  auto const Jf = diff(v, 1e-8);

  auto pathline      = integrate_pathline(v, x, t);
  auto pathline_prop = pathline.template add_vertex_property<double>("prop");

  // for each vertex of the pathline calculate properties
  for (size_t i = 0; i < pathline.num_vertices(); ++i) {
    typename std::decay_t<decltype(pathline)>::vertex_idx vi{i};
    auto const& px = pathline.vertex_at(i);
    auto const& pt = pathline.parameterization_at(i);
    if (v.in_domain(px, pt)) {
      auto const J     = Jf(px, pt);
      auto const S     = (J + transposed(J)) / 2;
      auto const Omega = (J - transposed(J)) / 2;
      //auto const SS    = S * S;
      //auto const OO    = Omega * Omega;
      //auto const SSOO  = SS + OO;

      //vec const vort{J2, 1) - J(1, 2),
      //               J(0, 2) - J(2, 0),
      //               J(1, 0) - J(0, 1)};
      // vorticity
      //pathline_prop[vi] = length(vort);
      // Q
      pathline_prop[vi] = (sqr_norm(Omega) - sqr_norm(S)) / 2;
      // lambda2
      //pathline_prop[vi] = eigenvalues_sym(SSOO)(1);
    } else {
      pathline_prop[vi] = 0.0 / 0.0;
    }
  }

  auto const lagrangian_eddy = pathline.integrate_property(pathline_prop);
  if (lagrangian_eddy > threshold) {
    ++uncertain_eddy.data_at(is...);
  }
}
//------------------------------------------------------------------------------
template <typename Grid, typename UncertainEddyProp>
void uncertain_eddy_detection(V const& v, real_number auto const t,
                              real_number auto const threshold, Grid const& g,
                              UncertainEddyProp& uncertain_eddy) {
  std::atomic_size_t cnt;
  std::mutex prog_mutex;
  auto iteration = [&](auto const... is) {
    auto const x = g.vertex_at(is...);
    uncertain_eddy_detection(v, x, t, threshold, uncertain_eddy, is...);
    ++cnt;
    std::lock_guard lock{prog_mutex};
    std::cerr << cnt / static_cast<double>(g.num_vertices()) << " %    \r";
  };
  parallel_for_loop(iteration, g.template size<0>(), g.template size<1>(),
                    g.template size<2>());
  std::cerr << '\n';
}
//------------------------------------------------------------------------------
void uncertain_eddy_detection(real_number auto const threshold) {
  auto  g = create_grid();
  auto& uncertain_eddy_prop =
      g.add_contiguous_vertex_property<double>("uncertain_eddy");

  V          v0{ensemble_file_paths.front()};
  auto const t = (v0.t_axis.back() - v0.t_axis.front()) / 2;
  for (auto const& ensemble_file_path : ensemble_file_paths) {
    V v{ensemble_file_path};
    std::cerr << "processing file " << ensemble_file_path << '\n';
    uncertain_eddy_detection(v, t, threshold, g, uncertain_eddy_prop);
  }
  g.dimension<2>().front() *= -0.0025;
  g.dimension<2>().back() *= -0.0025;
  g.write_vtk("red_sea_uncertain_eddy.vtk");
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
int main() {
  tatooine::scivis_contest_2020::uncertain_eddy_detection(0.0);
}
