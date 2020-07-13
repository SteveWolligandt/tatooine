#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <tatooine/okubo_weiss_field.h>
//==============================================================================
void print_usage();
int main(int argc, char** argv) {
  using namespace tatooine;
  if (argc < 2) {
    print_usage();
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  if (argc < 3) {
    print_usage();
    throw std::runtime_error{"specify time!"};
  }
  fields::scivis_contest_2020_ensemble_member v{argv[1]};

  double t = std::stod(argv[2]);
  grid   g{v.xc_axis, v.yc_axis,
           linspace{v.z_axis.front(), v.z_axis.back(), 100}};
  auto& imag_prop = g.add_contiguous_vertex_property<double, x_fastest>("imag");
  auto& lambda2_prop = g.add_contiguous_vertex_property<double, x_fastest>("lambda2");

  g.dimension<0>().pop_front(); 
  g.dimension<1>().pop_front(); 
  g.dimension<2>().pop_front(); 
  g.dimension<0>().pop_back(); 
  g.dimension<1>().pop_back(); 
  g.dimension<0>().pop_back(); 
  g.dimension<1>().pop_back(); 
  g.dimension<2>().pop_back(); 
  auto J = diff(v, 1e-4);
  okubo_weiss_field lambda2{v};
  auto apply_properties = [&](auto const... is) {
    vec const  x       = g.vertex_at(is...);
    auto const j       = J(x, t);
    auto       lambda  = eigenvalues(j);
    bool       is_imag = std::abs(lambda(0).imag()) > 1e-10 ||
                         std::abs(lambda(1).imag()) > 1e-10 ||
                         std::abs(lambda(2).imag()) > 1e-10;
    imag_prop.data_at(is...) = is_imag ? 1 : 0;
    lambda2_prop.data_at(is...) = lambda2(x, t);
  };
  for_loop(apply_properties, g.size<0>(), g.size<1>(), g.size<2>());

  g.write_vtk("red_sea_imaginary.vtk");
}
void print_usage() {
  std::cerr << "usage:\n";
  std::cerr << "./eigenvalues <path/to/ensemble> <time>\n";
}
