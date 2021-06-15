#include <tatooine/boussinesq.h>
#include <tatooine/cavity.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/random.h>
#include <tatooine/rbc.h>

using namespace tatooine;

template <typename V>
auto random_domain_position(const V& v, const boundingbox<double, 2>& domain, double t) {
  random::uniform    randx{domain.min(0), domain.max(0)};
  random::uniform    randy{domain.min(1), domain.max(1)};
  typename V::pos_t x;
  do {
    x(0) = randx();
    x(1) = randy();
  } while (!v.in_domain(x, t));
  return x;
}

template <typename Pathline>
auto resample_pathline_uniformly(const Pathline& pathline, double t0,
                                 double tau, size_t num_samples) {
  line<double, 2> l;
  for (auto t : linspace(t0, t0 + tau, num_samples)) {
    try {
      l.push_back(pathline(t));
    } catch (std::exception&) {}
  }
  return l;
}

template <typename V>
auto generate_pathlines(const V& v, const boundingbox<double, 2>& domain,
                        size_t n, double t0, double tau, size_t num_samples) {
  integration::vclibs::rungekutta43<double, 2> rk43;

  std::vector<line<double, 2>> lines;
  for (size_t i = 0; i < n; ++i) {
    auto  x        = random_domain_position(v, domain, t0);
    auto& pathline = rk43.integrate(v, x, t0, tau);
    lines.push_back(
        resample_pathline_uniformly(pathline, t0, tau, num_samples));
  }
  return lines;
}

template <typename V>
auto generate_and_write_pathlines(const V&                      v,
                                  const boundingbox<double, 2>& domain,
                                  size_t n, double t0, double tau,
                                  size_t             num_samples,
                                  const std::string& filepath) {
  auto pathlines =
      generate_pathlines(v, domain, n, t0, tau, num_samples);
  std::ofstream file{filepath};
  if (file.is_open()) {
    file << "pathlines\n";
    file << "count " << n << '\n';
    file << "domain " << domain.min(0) << ' ' << domain.max(0) << ' '
         << domain.min(1) << ' ' << domain.max(1) << ' ' << t0 << ' '
         << t0 + tau << '\n';
    file << "end_header\n";
    double dtau = tau / (num_samples - 1);
    for (const auto& pl : pathlines) {
      file << pl.num_vertices() << ' ' << t0 << ' '
           << dtau * (pl.num_vertices() - 1) << '\n';

      size_t cnt = 0;
      for (const auto& v : pl.vertices()) {
        file << v(0) << ' ' << v(1) << ' ' << t0 + dtau * cnt++ << '\n';
      }
    }
  }
}

int main() {
  size_t num_pathlines = 3000;
  size_t num_samples = 100;

  cavity v0{
      "/home/steve/vectorfield_datasets/2DCavity/"
      "Cavity2DTimeFilter3x3x7_100_bin.am"};

  generate_and_write_pathlines(v0,
                               boundingbox{vec{-1.0, -1.0}, vec{8.1164, 1.5}},
                               num_pathlines, 5, 5, num_samples, "cavity.thomas");

  rbc v1{
     "/home/steve/vectorfield_datasets/RBC/binary"};
  generate_and_write_pathlines(v1,
                              boundingbox{vec{0.00390625, 0.00390625},
                              vec{3.99609375, 0.99609375}}, num_pathlines,
                              2000, 20, num_samples, "rbc.thomas");

  boussinesq v2{"/home/steve/vectorfield_datasets/boussinesq.am"};
  generate_and_write_pathlines(v2,
                               boundingbox{vec{-0.5, -0.5}, vec{0.5, 2.5}},
                               num_pathlines, 0, 20, num_samples, "boussinesq.thomas");
}
