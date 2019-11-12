#include <filesystem>
#include <tatooine/chrono.h>
#include <fstream>
#include "datasets.h"
#include "settings.h"
#include "random_seed.h"
#include "steadification.h"

//==============================================================================
using namespace std::filesystem;
using namespace std::chrono;
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, typename VReal, typename T0Real, typename BTauReal, typename FTauReal, typename StepsizeReal, typename CovReal>
void calc(const field<V, VReal, 2, 2>& v, T0Real t0, BTauReal btau, FTauReal ftau,
          size_t num_its, size_t seed_res, StepsizeReal stepsize,
          CovReal desired_coverage, std::string seed_str) {
  using settings = settings_t<V>;

  std::seed_seq   seed(begin(seed_str), end(seed_str));
  std::mt19937_64 random_engine{seed};
  auto p = std::string{settings::name} + "/";
  size_t i = 1;
  while (exists(p)) {
    p = std::string(settings::name) + "_" + std::to_string(i) + "/";
    ++i;
  }
  create_directory(p);
  Steadification steadification(settings::domain, settings::render_resolution,
                                  t0, btau, ftau, seed_res, stepsize);

  using listener_t = typename decltype(steadification)::listener_t;
  struct : listener_t {
    using real_t   = typename listener_t::energy_t;
    using status_t = typename listener_t::status_t;

    size_t num_its;
    void   on_end_of_iteration(size_t i, const real_t&, const status_t&,
                               const real_t&, const status_t&) const override {
       std::cerr << "[ " << i + 1 << " / " << num_its
                 << " ]                               \r" << std::flush;
    }

    // void on_new_best_status(size_t i, const real_t& best_e, const status_t&,
    //                         const real_t&, const status_t&) const override {
    //   std::cerr << "new best e = " << best_e << '\n';
    // }
    // void on_using_worse(size_t        i, const real_t&, const status_t&,
    //                     const real_t& cur_e, const status_t&) const override {
    //   std::cerr << "using worse e = " << cur_e << '\n';
    // }
    // void on_going_back(size_t        i, const real_t&, const status_t&,
    //                    const real_t& cur_e, const status_t&) const override {
    //   std::cerr << "going back e = " << cur_e << '\n';
    // }

  } listener;

  listener.num_its = num_its;

  auto [elapsed, sol] = measure([&]() {
    return steadification.calc(v, num_its, settings::num_edges, p,
                               desired_coverage, random_engine, {&listener});
  });

  if (std::ofstream f(p + "settings.md"); f) {
    f << settings::name << "\n===\n\n";

    auto [h, min, s, ms, mus] =
        break_down_durations<hours, minutes, seconds, milliseconds,
                                       microseconds>(elapsed);
    f << "h|min|s|ms|mus\n"
      << ":---:|:---:|:---:|:---:|:---:\n"
      << h.count() << "|" << min.count() << "|" << s.count() << "|"
      << ms.count() << "|" << mus.count() << "\n\n";

    f << "t0|backward tau|forward tau|#iterations|x-dimension|y-dimension|resolution|seed\n"
      << ":---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:\n"
      << t0 << "|" << btau << "|" << ftau << "|" << num_its << "|"
      << settings::domain.dimension(0) << "|" << settings::domain.dimension(1)
      << "|" << settings::render_resolution << "|" << seed_str << "\n\n";

    f << "![](/steadification/" << p << "result.png)\n\n";

    for (size_t i = 0; i < sol.size(); ++i) {
      std::string  pos = "positions|", forward = "forward taus|",
                  backward = "backward taus|", line = ":---:|";
      for (size_t j = 0; j < sol[i].size(); ++j) {
        const auto& [v, b, f] = sol[i][j];
        line += ":---:";
        pos += v.to_string();
        forward += std::to_string(f);
        backward += std::to_string(b);
        if (j != sol[i].size() - 1) {
          line += "|";
          pos += "|";
          forward += "|";
          backward += "|";
        }
      }

      f << pos << "\n" << line << "\n" << forward << "\n" << backward << "\n\n";

      f << "![](/steadification/" << p << "result_sub" << i << "_color_lic.png)\n\n";
    }

    f.close();
  }
}

template <typename V, typename VReal>
void calc(const field<V, VReal, 2, 2>& v, int argc, char** argv) {
  double      t0               = argc > 2 ? atof(argv[2]) : 0;
  double      btau             = argc > 3 ? atof(argv[3]) : -5;
  double      ftau             = argc > 4 ? atof(argv[4]) : 5;
  size_t      num_its          = argc > 5 ? atoi(argv[5]) : 100;
  size_t      seed_res         = argc > 6 ? atoi(argv[6]) : 3;
  double      stepsize         = argc > 7 ? atof(argv[7]) : 0.2;
  double      desired_coverage = argc > 8 ? atof(argv[8]) : 0.999;
  auto        seed_str         = argc > 9 ? argv[9]       : random_string(10);
  std::cerr << "seed: " << seed_str << '\n';

  calc(v, t0, btau, ftau, num_its, seed_res, stepsize, desired_coverage,
       seed_str);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
int main(int argc, char** argv) {
  using namespace tatooine;
  std::cerr << total_memory() / 1024.0 << "MB\n";
  std::string v = argv[1];
  if (v == "dg") {
    calc(numerical::doublegyre<double>{}, argc, argv);
  } else if (v == "fdg") {
    calc(fixed_time_field{numerical::doublegyre<double>{}, 0}, argc, argv);
  } else if (v == "sc") {
    calc(numerical::sinuscosinus<double>{}, argc, argv);
  } else if (v == "la") {
    calc(laminar<double>{}, argc, argv);
  }
  // else if (v == "cy")  { calc            (cylinder{}, argc, argv); }
  // else if (v == "fw")  { calc        (FlappingWing{}, argc, argv); }
  // else if (v == "mg")  { calc  (movinggyre<double>{}, argc, argv); }
  else if (v == "rbc") {
    calc(rbc{}, argc, argv);
  } else if (v == "bou") {
    calc(boussinesq{dataset_dir + "/boussinesq.am"}, argc, argv);
  } else if (v == "cav") {
    calc(cavity{}, argc, argv);
  } else {
    throw std::runtime_error("Dataset not recognized");
  }
}
