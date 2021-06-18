#include <tatooine/chrono.h>
#include <tatooine/filesystem.h>
#include <tatooine/integration/vclibs/rungekutta43.h>

#include <random>

#include "datasets.h"
#include "random_seed.h"
#include "settings.h"
#include "shaders.h"

//==============================================================================
using namespace tatooine::rendering::gl;
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
using dg_t     = tatooine::numerical::doublegyre<double>;
using sincos_t = tatooine::numerical::sinuscosinus<double>;

static constexpr std::string_view pathline_dir = "pathlines";

//==============================================================================
template <typename T>
struct local_settings;


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
struct local_settings<boussinesq> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 20;
  static constexpr std::string_view name = "boussinesq";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<cavity> {
  static constexpr double           min  = 0;
  static constexpr double           max  = 10;
  static constexpr std::string_view name = "cavity";
};

//------------------------------------------------------------------------------
template <>
struct local_settings<rbc> {
  static constexpr double           min  = 2000;
  static constexpr double           max  = 2020;
  static constexpr std::string_view name = "rbc";
};

//------------------------------------------------------------------------------
template <typename V>
std::string filename(const std::string seed, size_t num_lines) {
  std::stringstream ss;
  ss << pathline_dir << "/" << local_settings<V>::name << "/" << seed << "_"
     << num_lines << ".png";
  return ss.str();
}

//==============================================================================
context w;
seedcurve_shader  line_shader;

//==============================================================================
void draw_line(float x0, float y0, float z0, float x1, float y1, float z1,
               float r = 0, float g = 0, float b = 0, float a = 0.8) {
  using indexeddata_t = indexeddata<vec3f>;
  indexeddata_t line(typename indexeddata_t::vbo_data_vec{{vec3f{x0, y0, z0}}, {vec3f{x1, y1, z1}}}, {0, 1});
  gl::line_width(3);
  line_shader.bind();
  line_shader.set_color(r, g, b, a);
  line.draw_lines();
}

//------------------------------------------------------------------------------
template <typename real_t>
void draw(const tatooine::line<real_t, 2>& l) {
  for (auto it = l.vertices().begin(); it != --l.vertices().end(); ++it)
    draw_line((*it)(0), (*it)(1), 0, (*next(it))(0), (*next(it))(1), 0);
}

//------------------------------------------------------------------------------
template <typename V>
void calc(V&& vf, size_t num_lines, const std::string& seed_str) {
  std::cout << "seed: " << seed_str << '\n';
  using s  = settings<V>;
  using ls = local_settings<V>;
  std::seed_seq      seed(begin(seed_str), end(seed_str));
  constexpr auto     res = s::render_resolution;
  constexpr auto     d   = s::domain;
  orthographiccamera cam(d.min(0), d.max(0), d.min(1), d.max(1), ls::min, ls::max, 0, 0,
                         res(0), res(1));
  line_shader.set_projection(cam.projection_matrix());
  tex2rgba<float> col(res(0), res(1));
  framebuffer     fbo{col};
  fbo.bind();

  gl::viewport(cam);
  gl::clear_color(1, 1, 1, 0);
  enable_blending();
  blend_func_alpha();
  clear_color_buffer();
  tatooine::integration::vclibs::rungekutta43<double, 2, interpolation::hermite>
                                                         integrator;
  std::mt19937_64                                        random_engine{seed};
  std::uniform_real_distribution                         x_dist{d.min(0), d.max(0)};
  std::uniform_real_distribution                         y_dist{d.min(1), d.max(1)};
  std::uniform_real_distribution t0_dist{ls::min, ls::max};
  double                         acct = 0;
  for (size_t i = 0; i < num_lines; ++i) {
    double                   t0   = t0_dist(random_engine);
    double                   btau = ls::min - t0;
    double                   ftau = ls::max - t0;
    tatooine::vec<double, 2> y0;
    do {
      y0 = tatooine::vec{x_dist(random_engine), y_dist(random_engine)};
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
  if (!filesystem::exists(path.str())) {
    filesystem::create_directory(path.str());
  }
  col.write_png(std::string{filename<V>(seed_str, num_lines)});
}

//------------------------------------------------------------------------------
template <typename V>
void calc(V&& v, int argc, const char** argv) {
  size_t      num_lines = argc > 2 ? atoi(argv[2]) : 100;
  std::string seed_str  = argc > 3 ? argv[3] : random_string(6);
  calc<V>(std::forward<V>(v), num_lines, seed_str);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================

int main(int argc, const char** argv) {
  using namespace tatooine::steadification;
  if (!filesystem::exists(std::string{pathline_dir})) {
    filesystem::create_directory(std::string{pathline_dir});
  }
  std::string vf = argv[1];
  //if (vf == "dg") calc<dg_t>(argc, argv);
  //if (vf == "sc") calc<sincos_t>(argc, argv);
  //if (vf == "rbc") calc<rbc>(argc, argv);
  if (vf == "bou") calc(tatooine::boussinesq{dataset_dir + "/boussinesq.am"}, argc, argv);
  //if (vf == "cav") calc<cavity>(argc, argv);
}
