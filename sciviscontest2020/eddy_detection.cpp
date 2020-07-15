#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <filesystem>
//==============================================================================
void print_usage(char**argv);
int main(int argc, char** argv) {
  using namespace tatooine;
  if (argc < 2) {
    print_usage(argv);
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  if (argc < 3) {
    print_usage(argv);
    throw std::runtime_error{"specify time!"};
  }
  fields::scivis_contest_2020_ensemble_member v{argv[1]};

  double t = std::stod(argv[2]);
  grid   g{v.xc_axis, v.yc_axis,
           linspace{v.z_axis.front(), v.z_axis.back(), 100}};
  auto& imag_prop = g.add_contiguous_vertex_property<double, x_fastest>("imag");
  auto& lambda2_prop = g.add_contiguous_vertex_property<double, x_fastest>("lambda2");
  auto& Q_prop = g.add_contiguous_vertex_property<double, x_fastest>("Q");
  auto& vorticity_magnitude_prop = g.add_contiguous_vertex_property<double, x_fastest>("vorticity_magnitude");

  g.dimension<0>().pop_front(); 
  g.dimension<1>().pop_front(); 
  g.dimension<2>().pop_front(); 
  g.dimension<0>().pop_back(); 
  g.dimension<1>().pop_back(); 
  g.dimension<0>().pop_back(); 
  g.dimension<1>().pop_back(); 
  g.dimension<2>().pop_back(); 
  auto Jf = diff(v, 1e-4);
  auto              apply_properties = [&](auto const... is) {
    vec const x = g.vertex_at(is...);
    if (v.in_domain(x, t)) {
      auto const J        = Jf(x, t);
      auto const lambda_J = eigenvalues(J);
      auto const S        = (J + transposed(J)) / 2;
      auto const Omega    = (J - transposed(J)) / 2;
      auto const SS       = S * S;
      auto const OO       = Omega * Omega;
      auto const SSOO     = SS + OO;
      vec const  vorticity{J(2, 1) - J(1, 2), J(0, 2) - J(2, 0),
                          J(1, 0) - J(0, 1)};

      bool       is_imag = std::abs(lambda_J(0).imag()) > 1e-10 ||
                     std::abs(lambda_J(1).imag()) > 1e-10 ||
                     std::abs(lambda_J(2).imag()) > 1e-10;
      imag_prop.data_at(is...)    = is_imag ? 1 : 0;
      lambda2_prop.data_at(is...) = eigenvalues_sym(SSOO)(1);
      Q_prop.data_at(is...)       = (sqr_norm(Omega) - sqr_norm(S)) / 2;
      vorticity_magnitude_prop.data_at(is...) = length(vorticity);
    } else {
      imag_prop.data_at(is...)                = 0.0 / 0.0;
      lambda2_prop.data_at(is...)             = 0.0 / 0.0;
      Q_prop.data_at(is...)                   = 0.0 / 0.0;
      vorticity_magnitude_prop.data_at(is...) = 0.0 / 0.0;
    }
  };
  parallel_for_loop(apply_properties, g.size<0>(), g.size<1>(), g.size<2>());

  namespace fs = std::filesystem;
  fs::path    p        = argv[1];
  std::string outpath =
      fs::path{p.filename()}.replace_extension("eddy_detection_" + std::to_string(t) + ".vtk");
  g.write_vtk(outpath);
}
void print_usage(char** argv) {
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " <path/to/ensemble> <time>\n";
}
