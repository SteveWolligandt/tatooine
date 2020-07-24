#ifndef TATOOINE_SCIVIS_CONTEST_2020_POSITIONS_IN_DOMAIN_H
#define TATOOINE_SCIVIS_CONTEST_2020_POSITIONS_IN_DOMAIN_H
namespace tatooine::scivis_contest_2020 {
template <typename V, typename Grid>
auto positions_in_domain(V const& v, Grid const& g) {
  std::cerr << "collecting positions in domain...\r";
  std::vector<typename Grid::pos_t> xs;
  std::vector<vec<size_t, 3>>       xis;
  xs.reserve(g.num_vertices() / 3);
  xis.reserve(g.num_vertices() / 3);
  std::mutex                        xs_mutex;
  auto const                        t = v.t_axis.front();
  std::atomic_size_t cnt;
  bool done = false;
  std::thread                       monitor{[&] {
    while (!done) {
      std::cerr << "collecting positions in domain... "
                << static_cast<double>(cnt) / g.num_vertices() * 100
                << "%       \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{200});
    }
  }};
  g.parallel_loop_over_vertex_indices([&](auto const... is) {
    auto const x   = g.vertex_at(is...);
    auto const eps = 1e-4;
    if (v.in_domain(x, t) && v.in_domain(vec{x(0) + eps, x(1), x(2)}, t) &&
        v.in_domain(vec{x(0) - eps, x(1), x(2)}, t) &&
        v.in_domain(vec{x(0), x(1) + eps, x(2)}, t) &&
        v.in_domain(vec{x(0), x(1) - eps, x(2)}, t) &&
        v.in_domain(vec{x(0), x(1), x(2) + eps}, t) &&
        v.in_domain(vec{x(0), x(1), x(2) - eps}, t)) {
      std::lock_guard l{xs_mutex};
      xs.push_back(x);
      xis.push_back(vec<size_t, 3>{is...});
    }
    ++cnt;
  });
  done = true;
  monitor.join();
  std::cerr << "collecting positions in domain... done!         \n";
  xs.shrink_to_fit();
  xis.shrink_to_fit();
  return std::pair{xs, xis};
}
}  // namespace tatooine::scivis_contest_2020
#endif
