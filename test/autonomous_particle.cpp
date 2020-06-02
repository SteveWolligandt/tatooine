#include <tatooine/autonomous_particle.h>

#include <catch2/catch.hpp>
#include <tatooine/doublegyre.h>
#include <tatooine/vtk_legacy.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("autonomnous_particle0", "[autonomous_particle]") {
  using boost::copy;
  using boost::adaptors::transformed;
  numerical::doublegyre v;
  v.set_infinite_domain(true);
  vec const           x0{1.0, 0.3};
  double const        t0       = 0;
  double const        t1       = 5;
  double const        tau_step = 0.1;
  double const        radius   = 0.1;

  autonomous_particle p0{x0, t0, radius};
  const auto particles = p0.integrate(v, tau_step, t1);

  {
    vtk::legacy_file_writer writer{"autonomous_particle_paths_forward.vtk",
                                   vtk::POLYDATA};
    if (writer.is_open()) {
      writer.write_header();

      std::vector<std::vector<size_t>> lines;
      std::vector<vec<double, 3>>      points;
      lines.reserve(size(particles));
      for (const auto& p : particles) {
        points.push_back(vec{p.m_fromx(0), p.m_fromx(1), p.m_fromt});
        points.push_back(vec{p.m_x1(0), p.m_x1(1), p.m_t1});
        lines.push_back(std::vector{size(points) - 2, size(points) - 1});
      }
      writer.write_points(points);
      writer.write_lines(lines);

      writer.close();
    }
  }
  {
    vtk::legacy_file_writer writer{"autonomous_particle_paths_backward.vtk",
                                   vtk::POLYDATA};
    if (writer.is_open()) {
      writer.write_header();

      std::vector<std::vector<size_t>> lines;
      std::vector<vec<double, 3>>      points;
      lines.reserve(size(particles));
      for (const auto& p : particles) {
        points.push_back(vec{p.m_x0(0), p.m_x0(1), t0});
        points.push_back(vec{p.m_x1(0), p.m_x1(1), p.m_t1});
        lines.push_back(std::vector{size(points) - 2, size(points) - 1});
      }
      writer.write_points(points);
      writer.write_lines(lines);

      writer.close();
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
