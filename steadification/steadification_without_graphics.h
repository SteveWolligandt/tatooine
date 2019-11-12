#ifndef STEADIFICATION_H
#define STEADIFICATION_H

#include <tatooine/cache.h>
#include <tatooine/doublegyre.h>
#include <tatooine/grid.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/parallel_for.h>
#include <tatooine/simulated_annealing.h>
#include <tatooine/streamsurface.h>
#include <boost/filesystem.hpp>
#include <boost/range/adaptors.hpp>
#include <cstdlib>
#include <vector>
#include "real_t.h"

class Steadification {
 public:
  using edge_t   = tatooine::grid_edge<real_t, 2>;
  using vertex_t = tatooine::grid_vertex<real_t, 2>;
  using grid_t   = tatooine::grid<real_t, 2>;

  using real_vec     = std::vector<real_t>;
  using edge_vec     = std::vector<edge_t>;
  using vertex_seq_t = typename grid_t::vertex_seq_t;

  //[vertex, backward_tau, forward_tau]
  using solution_t = std::vector<std::tuple<vertex_t, real_t, real_t>>;

  using listener_t = tatooine::simulated_annealing_listener<float, solution_t>;

  using ribbon_t     = tatooine::mesh<real_t, 2>;

  using ndist = std::normal_distribution<real_t>;
  using udist = std::uniform_real_distribution<real_t>;

  //============================================================================

 public:
  tatooine::grid<real_t, 2> grid;

 private:
  const real_t                  t0;
  const real_t                  btau, ftau;
  size_t                        seed_res;
  real_t                        stepsize;
  static constexpr unsigned int reduce_work_group_size = 1024;
  size_t                        seedcurve_length;
  size_t                        num_its;

 private:
  tatooine::cache<edge_t, ribbon_t> ribbon_cache;
  std::vector<solution_t>           solutions;

  std::mutex ribbon_mutex;

 public:
  //============================================================================
  Steadification(const tatooine::grid<real_t, 2>& _grid, real_t t0,
                 real_t btau, real_t ftau, size_t seed_res, real_t stepsize);

  //----------------------------------------------------------------------------
  template <typename vf_t>
  float evaluate(const vf_t& vf, const solution_t& sol) {
    return 0.0f;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto ribbon_uncached(const vf_t& vf, const edge_t& e, real_t stepsize) {
    using namespace VC::odeint;
    using vec2 = tatooine::vec<real_t, 2>;
    tatooine::streamsurface ssf{
        vf,
        t0,
        tatooine::parameterized_line<real_t, 2>{{e.first.position(), 0},
                                                {e.second.position(), 1}},
        tatooine::integration::vclibs::rungekutta43<double, 2>{
            AbsTol = 1e-6, RelTol = 1e-6, InitialStep = 0, MaxStep = stepsize},
        tatooine::interpolation::linear<real_t>{},
        tatooine::interpolation::hermite<real_t>{}};
    ssf.integrator().cache().set_max_memory_usage(1024 * 1024 * 25);
    auto        ribbon  = ssf.discretize(seed_res, stepsize, btau, ftau);
    const auto& mesh_uv = ribbon.template vertex_property<vec2>("uv");
    auto&       mesh_vf = ribbon.template add_vertex_property<vec2>("vf");

    for (auto v : ribbon.vertices()) {
      if (vf.in_domain(ribbon[v], t0 + mesh_uv[v](1))) {
        mesh_vf[v] = vf(ribbon[v], t0 + mesh_uv[v](1));
      } else {
        mesh_vf[v] = tatooine::vec<real_t, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }

    return ribbon;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto ribbon_uncached(const vf_t& vf, const edge_t& e) {
    return ribbon_uncached(vf, e, stepsize);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  const auto& ribbon(const vf_t& vf, const edge_t& e, real_t stepsize) {
    using namespace VC::odeint;
    using vec2 = tatooine::vec<real_t, 2>;
    if (auto found = ribbon_cache.find(e); found == ribbon_cache.end()) {
      auto            ribbon = ribbon_uncached(vf, e, stepsize);
      std::lock_guard lock(ribbon_mutex);
      return ribbon_cache.try_emplace(e, std::move(ribbon)).first->second;
    } else {
      return found->second;
    }
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  const auto& ribbon(const vf_t& vf, const edge_t& e) {
    return ribbon(vf, e, stepsize);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto ribbons(const vf_t& vf, const solution_t& sol) {
    std::vector<ribbon_t> rs;
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      rs.push_back(ribbon(vf, {v0, v1}));
    }
    return rs;
  }

  //------------------------------------------------------------------------
  auto clean_sequence(vertex_seq_t seq) {
    bool cleaned = false;
    while (!cleaned && seq.size() >= 4) {
      bool changed = false;
      for (auto it = begin(seq); it != prev(end(seq), 3) && !changed; ++it)
        if (grid.are_direct_neighbors(*it, *next(it, 3)) &&
            grid.are_direct_neighbors(*it, *next(it, 2)) &&
            grid.are_direct_neighbors(*next(it), *next(it, 3))) {
          seq.erase(next(it));
          changed = true;
        }

      if (!changed) cleaned = true;
    }

    cleaned = false;
    while (!cleaned && seq.size() >= 6) {
      bool changed = false;
      for (auto it0 = begin(seq); it0 != prev(end(seq), 5) && !changed; ++it0)
        for (auto it1 = next(it0, 5); it1 != end(seq) && !changed; ++it1)
          if (grid.are_direct_neighbors(*it0, *it1)) {
            seq.erase(next(it0), it1);
            changed = true;
          }

      if (!changed) cleaned = true;
    }

    return seq;
  };

  //--------------------------------------------------------------------------
  template <typename vf_t, typename RandEng>
  auto calc(const vf_t& vf, size_t _num_its, size_t _seedcurve_length,
            const std::string& path, real_t desired_coverage, RandEng&& eng,
            const std::vector<listener_t*>& listeners = {}) {
    size_t         num_pixels_in_domain = 0;
    num_its          = _num_its;
    seedcurve_length = _seedcurve_length;

    //--------------------------------------------------------------------------
    auto energy = [&vf, this](const solution_t& sol) {
      return 0.0f;
      // return evaluate(vf, sol);
    };

    //--------------------------------------------------------------------------
    // temperature must be between 0 and 1
    auto temperature = [this](real_t i) { return 1 - i / (num_its - 1); };
    // auto temperature = [num_peaks = num_its / 1000, this](real_t i) {
    //  real_t norm_i = (real_t(i) / (num_its - 1.0));
    //  auto   t      = cos(norm_i * M_PI / 2.0 * real_t(num_peaks * 2 - 1));
    //  return t * t * t * t;
    //};

    //--------------------------------------------------------------------------
    auto permute = [&, this](const solution_t& old_sol, real_t temp) {
      auto stddev = temp;

      const auto global_local_border = 0.7;
      solution_t sol;
      if (temp > global_local_border) {
        //---------------------------------------------------------------------\
        // GLOBAL CHANGE
        std::uniform_real_distribution<real_t> rand_cont{0, 1};
        auto                                   bw_tau = udist{btau, 0}(eng);
        auto                                   fw_tau = udist{0, ftau}(eng);
        ndist seq_len{real_t(seedcurve_length), real_t(2)};
        // ndist  seq_len{real_t(old_sol.size()), real_t(2)};
        size_t len = std::max<size_t>(1, seq_len(eng));

        // continue at end of previous solution
        // if (!solutions.empty() && rand_cont(eng) < 0.25) {
        //   std::uniform_int_distribution<size_t> index_dist{
        //       0, solutions.size() - 1};
        //   std::uniform_int_distribution<size_t> front_back_dist{0, 1};
        //   const auto& old_sol = solutions[index_dist(eng)];

        //   const bool front   = front_back_dist(eng) == 0;
        //   const auto start_v = front ? std::get<0>(old_sol.front())
        //                              : std::get<0>(old_sol.back());
        //   auto new_seq =
        //       clean_sequence(grid.random_straight_vertex_sequence(len, start_v, eng));

        //   for (size_t i = 0; i < new_seq.size(); ++i) {
        //     sol.push_back(std::tuple{new_seq[i], bw_tau, fw_tau});
        //   }

        //   return sol;

        // } else {  // completely random
          auto new_seq = clean_sequence(
              grid.random_straight_vertex_sequence(len, 130.0, eng));
          for (size_t i = 0; i < new_seq.size(); ++i) {
            sol.push_back(std::tuple{new_seq[i], bw_tau, fw_tau});
          }

          return sol;
        // }
        //---------------------------------------------------------------------/

      } else {
        //---------------------------------------------------------------------\
         // LOCAL CHANGE
        // ndist num_mutations{real_t(temp), 5.0};
        // auto  scaler             = std::abs(num_mutations(eng));
        // auto  num_mutation_steps = old_sol.size() * scaler;

        vertex_seq_t mutated_seq;
        boost::transform(old_sol, std::back_inserter(mutated_seq),
                         [](const auto &v) { return std::get<0>(v); });

        // permute several times
        do {
          mutated_seq = grid.mutate_seq_straight(mutated_seq, 130, 5, eng);
        } while (mutated_seq.size()>=2);


        auto clamp_pos = [this](real_t v) {
          return std::min(std::max<real_t>(v, 0), ftau);
        };
        auto clamp_neg = [this](real_t v) {
          return std::min<real_t>(std::max(v, btau), 0);
        };
        auto fw_tau =
            clamp_pos(ndist{std::get<2>(old_sol.front()), stddev}(eng));
        auto bw_tau =
            clamp_neg(ndist{std::get<1>(old_sol.front()), stddev}(eng));
        for (size_t i = 0; i < mutated_seq.size(); ++i) {
          sol.push_back(std::tuple{mutated_seq[i], bw_tau, fw_tau});
        }

        return sol;
        //---------------------------------------------------------------------/
      }
    };

    auto vertex_to_tuple = [this](auto v) { return std::tuple{v, btau, ftau}; };
    size_t i             = 0;
    real_t coverage      = 0;
    while (coverage < desired_coverage) {
      solution_t start_solution;
      boost::transform(
          grid.random_straight_vertex_sequence(seedcurve_length, 130.0, eng),
          std::back_inserter(start_solution), vertex_to_tuple);

      solutions.push_back(tatooine::simulated_annealing<std::greater>(
                              start_solution, num_its, energy, temperature,
                              permute, eng, listeners)
                              .second);
      ++i;
    }

    return solutions;
  }
};

#endif
