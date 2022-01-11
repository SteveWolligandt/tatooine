#ifndef TATOOINE_SCIVIS_CONTEST_2020_POSITIONS_IN_DOMAIN_H
#define TATOOINE_SCIVIS_CONTEST_2020_POSITIONS_IN_DOMAIN_H
//==============================================================================
#include <array>
#include "monitor.h"
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename V, typename Grid>
auto positions_in_domain(V const& v, Grid const& g) {
  struct positions_t {
    alignas(64) std::vector<typename Grid::pos_t> vector;
  };
  struct indices_t {
    alignas(64) std::vector<vec<size_t, 3>> vector;
  };
  std::vector<positions_t> xss(std::thread::hardware_concurrency());
  std::vector<indices_t>   xiss(std::thread::hardware_concurrency());
  for (auto& xs : xss) {
    xs.vector.reserve(g.vertices().size() / 5);
  }
  for (auto& xis : xiss) {
    xis.vector.reserve(g.vertices().size() / 5);
  }
  auto const         t = v.t_axis.front();
  std::atomic_size_t cnt;
  monitor(
      [&] {
        g.iterate_over_vertex_indices([&](auto const... is) {
          auto const x   = g.vertex_at(is...);
          auto const eps = 1e-4;
          if (v.in_domain(x, t) &&
              v.in_domain(vec{x(0) + eps, x(1), x(2)}, t) &&
              v.in_domain(vec{x(0) - eps, x(1), x(2)}, t) &&
              v.in_domain(vec{x(0), x(1) + eps, x(2)}, t) &&
              v.in_domain(vec{x(0), x(1) - eps, x(2)}, t) &&
              v.in_domain(vec{x(0), x(1), x(2) + eps}, t) &&
              v.in_domain(vec{x(0), x(1), x(2) - eps}, t)) {
            xss[omp_get_thread_num()].vector.push_back(x);
            xiss[omp_get_thread_num()].vector.push_back(vec<size_t, 3>{is...});
          }
          ++cnt;
        }, execution_policy::parallel);
      },
      [&] {
        return static_cast<double>(cnt) / g.vertices().size();
      },
      "collecting positions in domain");
  for (size_t i = 1; i < size(xss); ++i) {
    xss.front().vector.insert(end(xss.front().vector), begin(xss[i].vector),
                              end(xss[i].vector));
  }
  for (size_t i = 1; i < size(xiss); ++i) {
    xiss.front().vector.insert(end(xiss.front().vector), begin(xiss[i].vector),
                               end(xiss[i].vector));
  }
  xss.front().vector.shrink_to_fit();
  xiss.front().vector.shrink_to_fit();
  std::cerr << "number of points in domain: " << size(xss.front().vector)
            << '\n';
  return std::pair{std::move(xss.front().vector), xiss.front().vector};
}
  //==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
