#include <Tatooine/integration/vclibs/rungekutta43.h>
#include <boost/filesystem.hpp>
#include <random>
#include <yavin>
#include "datasets.h"
#include "random_seed.h"
#include "settings.h"
#include "shaders.h"

//==============================================================================
using namespace yavin;
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
using dg_t     = tatooine::analytical::doublegyre<double>;
using sincos_t = tatooine::analytical::sinuscosinus<double>;

static constexpr std::string_view pathline_dir = "pathlines";

//==============================================================================
template <typename T>
struct local_settings;

//------------------------------------------------------------------------------
template <>
struct local_settings<FlappingWing> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 149.989;
  static constexpr std::string_view name = "flapping-wing";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<FixedDoubleGyre> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 10;
  static constexpr std::string_view name = "fdg";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<dg_t> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 10;
  static constexpr std::string_view name = "doublegyre";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<sincos_t> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 2 * M_PI;
  static constexpr std::string_view name = "sincos";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<Boussinesq> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 20;
  static constexpr std::string_view name = "boussinesq";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<Cavity> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 10;
  static constexpr std::string_view name = "cavity";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<RBC> {
  static constexpr double           min  = 2000;
  static constexpr double           max  = 2020;
  static constexpr std::string_view name = "rbc";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<movinggyre<double>> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 10;
  static constexpr std::string_view name = "movinggyre";
};

//------------------------------------------------------------------------------
template <typename vf_t>
std::string filename(const std::string seed, size_t num_lines) {
  std::stringstream ss;
  ss << pathline_dir << "/" << local_settings<vf_t>::name << "/" << seed << "_"
     << num_lines << ".png";
  return ss.str();
}

//==============================================================================
context w;
LineShader  line_shader;

//==============================================================================
void draw_line(float x0, float y0, float z0, float x1, float y1, float z1,
               float r = 0, float g = 0, float b = 0, float a = 0.8) {
  indexeddata<vec3> line({{x0, y0, z0}, {x1, y1, z1}}, {0, 1});
  gl::line_width(3);
  line_shader.bind();
  line_shader.set_color(r, g, b, a);
  line.draw_lines();
}

//------------------------------------------------------------------------------
template <typename real_t>
void draw(const tatooine::line<real_t, 2>& l) {
  for (auto it = l.begin(); it != --l.end(); ++it)
    draw_line(std::get<0>(*it)(0), std::get<0>(*it)(1), 0,
              std::get<0>(*next(it))(0), std::get<0>(*next(it))(1), 0);
}

//------------------------------------------------------------------------------
template <typename vf_t>
void calc(size_t num_lines, const std::string& seed_str) {
  std::cout << "seed: " << seed_str << '\n';
  using s  = settings_t<vf_t>;
  using ls = local_settings<vf_t>;
  std::seed_seq      seed(begin(seed_str), end(seed_str));
  constexpr auto     res = s::render_resolution;
  constexpr auto     x   = s::domain.dimension(0);
  constexpr auto     y   = s::domain.dimension(1);
  orthographiccamera cam(x.min, x.max, y.min, y.max, ls::min, ls::max, 0, 0,
                         res(0), res(1));
  line_shader.set_projection(cam.projection_matrix());
  tex2rgba<float> col(res(0), res(1));
  framebuffer     fbo{col};
  fbo.bind();

  gl::viewport(cam.viewport());
  gl::clear_color(1, 1, 1, 0);
  enable_blending();
  blend_func_alpha();
  clear_color_buffer();
  vf_t                                                   vf;
  tatooine::integration::vclibs::rungekutta43<2, double> integrator;
  std::mt19937_64                                        random_engine{seed};
  std::uniform_real_distribution                         x_dist{x.min, x.max};
  std::uniform_real_distribution                         y_dist{y.min, y.max};
  std::uniform_real_distribution t0_dist{ls::min, ls::max};
  double                         acct = 0;
  for (size_t i = 0; i < num_lines; ++i) {
    real_t                   t0   = t0_dist(random_engine);
    real_t                   btau = ls::min - t0;
    real_t                   ftau = ls::max - t0;
    tatooine::Vec<real_t, 2> y0;
    do {
      y0 = {x_dist(random_engine), y_dist(random_engine)};
    } while (!vf.in_domain(y0, t0));
    // std::cout << y0 << ", " << t0 << ", " << btau << ", " << ftau << '\n';
    auto [t, line] = tatooine::measure(
        [&] { return integrator.integrate(vf, y0, t0, btau, ftau); });
    acct += std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() /
         1000000.0;
    draw(line);
  }
  std::cerr << "integration took " << acct << "ms\n";

  std::stringstream path;
  path << pathline_dir << "/" << ls::name;
  if (!boost::filesystem::exists(path.str())) {
    boost::filesystem::create_directory(path.str());
  }
  col.write_png(std::string{filename<vf_t>(seed_str, num_lines)});
}

//------------------------------------------------------------------------------
template <typename vf_t>
void calc(int argc, const char** argv) {
  size_t      num_lines = argc > 2 ? atoi(argv[2]) : 100;
  std::string seed_str  = argc > 3 ? argv[3] : random_string(6);
  calc<vf_t>(num_lines, seed_str);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================

int main(int argc, const char** argv) {
  using namespace tatooine::steadification;
  if (!boost::filesystem::exists(std::string{pathline_dir})) {
    boost::filesystem::create_directory(std::string{pathline_dir});
  }
  std::string vf = argv[1];
  if (vf == "fdg") calc<FixedDoubleGyre>(argc, argv);
  if (vf == "dg") calc<dg_t>(argc, argv);
  if (vf == "sc") calc<sincos_t>(argc, argv);
  if (vf == "mg") calc<movinggyre<real_t>>(argc, argv);
  if (vf == "fw" ) calc<FlappingWing>   (argc, argv);
  if (vf == "rbc") calc<RBC>(argc, argv);
  if (vf == "bou") calc<Boussinesq>(argc, argv);
  if (vf == "cav") calc<Cavity>(argc, argv);
}
