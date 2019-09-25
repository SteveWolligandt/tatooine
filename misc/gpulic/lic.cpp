#include <tatooine/newdoublegyre.h>
#include <tatooine/doublegyre.h>
#include <tatooine/boussinesq.h>
#include <tatooine/counterexample_sadlo.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/interpolation.h>
#include <tatooine/linspace.h>
#include <tatooine/sinuscosinus.h>
#include <iomanip>
#include <filesystem>
#include <random>
#include <string>
#include <yavin>

//==============================================================================
namespace tatooine::misc::gpulic {
//==============================================================================

struct shader : yavin::shader {
  shader(GLuint num_steps, GLfloat min_x, GLfloat min_y, GLfloat max_x,
         GLfloat max_y) {
    add_stage<yavin::computeshader>("lic.comp");
    create();
    set_uniform("vector_tex", 0);
    set_uniform("noise_tex", 1);
    set_num_steps(num_steps);
    set_bounding_min(min_x, min_y);
    set_bounding_max(max_x, max_y);
  }

  void dispatch(GLuint w, GLuint h) {
    bind();
    yavin::gl::dispatch_compute(w, h, 1);
  }

  void set_num_steps(GLuint num_steps) { set_uniform("num_steps", num_steps); }

  void set_bounding_min(GLfloat x, GLfloat y) {
    set_uniform("bounding_min", x, y);
  }

  void set_bounding_max(GLfloat x, GLfloat y) {
    set_uniform("bounding_max", x, y);
  }
};

//==============================================================================
template <typename V>
void lic(const std::string& name, const grid<float, 2>& grid,
         linspace<double> ts) {
  using namespace yavin;
  using interpolation::linear;
  yavin::context w{4, 3};
  auto           bb = grid.boundingbox();
  V              vf;

  // create noise
  grid_sampler<float, 2, float, linear, linear> noise{grid};
  noise.randu(std::mt19937{1234});
  tex2r<float> noise_tex{yavin::LINEAR, yavin::REPEAT, noise.data().unchunk(),
                         grid.dimension(0).size(), grid.dimension(1).size()};
  noise_tex.bind(1);

  shader lic_shader{100, (float)bb.min(0), (float)bb.min(1), (float)bb.max(0),
                    (float)bb.max(1)};

  size_t cnt = 0;
  for (auto t : ts) {
    yavin::tex2rgba<float> lic_tex{grid.dimension(0).size(),
                                   grid.dimension(1).size()};
    lic_tex.clear(0, 0, 0, 0);
    lic_tex.bind_image_texture(0);

    auto               resampled_vf = resample<linear, linear>(vf, grid, t);
    std::vector<float> data;
    data.reserve(grid.dimension(0).size() * grid.dimension(1).size() * 2);
    for (const auto& v : resampled_vf.sampler().data().unchunk()) {
      data.push_back(static_cast<float>(v(0)));
      data.push_back(static_cast<float>(v(1)));
    }
    tex2rg<float> vf_tex{yavin::LINEAR, yavin::REPEAT, data,
                         grid.dimension(0).size(), grid.dimension(1).size()};
    vf_tex.bind(0);
    lic_shader.dispatch(grid.dimension(0).size() / 16.0 + 1,
                        grid.dimension(1).size() / 16.0 + 1);
    std::stringstream out;
    if (!std::filesystem::exists(name)) {
      std::filesystem::create_directory(name);
    }
    out << name << "/" << std::setfill('0') << std::setw(4) << cnt++
        << "_" << name << ".png";
    lic_tex.write_png(out.str());
  }
}

//==============================================================================
}  // namespace tatooine::misc::gpulic
//==============================================================================

int main(int /*argc*/, const char** argv) {
  using tatooine::grid;
  using tatooine::linspace;

  std::string vf(argv[1]);
  linspace    ts{atof(argv[2]), atof(argv[3]), size_t(atoi(argv[4]))};
  if (vf == "dg") {
    tatooine::misc::gpulic::lic<tatooine::numerical::doublegyre<double>>(
        "doublegyre",
        grid{linspace{0.0f, 2.0f, 1000}, linspace{0.0f, 1.0f, 500}}, ts);

  } else if (vf == "ndg") {
    tatooine::misc::gpulic::lic<tatooine::numerical::newdoublegyre<double>>(
        "newdoublegyre",
        grid{linspace{0.0f, 2.0f, 1000}, linspace{0.0f, 1.0f, 500}}, ts);

  } else if (vf == "sc") {
    tatooine::misc::gpulic::lic<tatooine::numerical::sinuscosinus<double>>(
        "sinuscosinus",
        grid{linspace{-2.0f, 2.0f, 2000}, linspace{-2.0f, 2.0f, 2000}}, ts);
  } else if (vf == "counterexamplesadlo") {
    tatooine::misc::gpulic::lic<tatooine::numerical::counterexample_sadlo<double>>(
        "counterexample_sadlo",
        grid{linspace{-3.1f, 3.1f, 2000}, linspace{-3.1f, 3.1f, 2000}}, ts);
  } else if (vf == "boussinesq") {
    tatooine::misc::gpulic::lic<tatooine::boussinesq>(
        "boussinesq",
        grid{tatooine::boussinesq::domain.dimension(0),
             tatooine::boussinesq::domain.dimension(1)},
        ts);
  }
}
