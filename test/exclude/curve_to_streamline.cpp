#include <tatooine/line.h>
#include <tatooine/curve_to_streamline.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/doublegyre.h>
#include <tatooine/orbit.h>
#include <tatooine/linspace.h>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

template <typename V, typename T0, typename C2S, typename Line>
void write_step(const V& v, T0 t0, const Line& line, const C2S& c2s,
                const std::string& path) {
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  if (writer.is_open()) {
    writer.set_title("foo");
    writer.write_header();

    // write points
    std::vector<std::array<double, 3>> ps;
    ps.reserve(line.size());
    for (const auto& p : line.vertices()) { ps.push_back({p(0), p(1), p(2)}); }
    writer.write_points(ps);

    // write lines
    std::vector<std::vector<size_t>> line_seq(
        1, std::vector<size_t>(line.size()));
    boost::iota(line_seq.front(), 0);
    if (line.is_closed()) { line_seq.front().push_back(0); }
    writer.write_lines(line_seq);

    writer.write_point_data(line.size());

    // write vectorfield
    std::vector<std::vector<double>> vs;
    vs.reserve(line.size());
    for (size_t i = 0; i < line.size(); ++i) {
      auto velo = v(line.vertex_at(i), t0);
      vs.push_back({velo(0), velo(1), velo(2)});
    }
    writer.write_scalars("vectorfield", vs);

    // write tangents
    std::vector<std::vector<double>> tangents;
    tangents.reserve(c2s.tangents.size());
    for (size_t i = 0; i < c2s.tangents.size(); ++i) {
      const auto t = c2s.tangents[i];
      tangents.push_back({t(0), t(1), t(2)});
    }
    writer.write_scalars("tangents", tangents);

    // write bases
    std::vector<std::vector<double>> base0, base1;
    base0.reserve(c2s.bases.size());
    base1.reserve(c2s.bases.size());
    for (size_t i = 0; i < c2s.bases.size(); ++i) {
      const auto& b = c2s.bases[i];
      base0.push_back({b(0,0), b(1,0), b(2,0)});
      base1.push_back({b(0,1), b(1,1), b(2,1)});
    }
    writer.write_scalars("base0", base0);
    writer.write_scalars("base1", base1);

    // neg grad alpha
    std::vector<std::vector<double>> grad_alpha_plane;
    grad_alpha_plane.reserve(c2s.grad_alpha.size());
    for (size_t i = 0; i < c2s.grad_alpha.size(); ++i) {
      const auto ga = c2s.grad_alpha[i];
      grad_alpha_plane.push_back({ga(0), ga(1)});
    }
    writer.write_scalars("grad_alpha_plane", grad_alpha_plane);

    std::vector<std::vector<double>> grad_alpha;
    grad_alpha.reserve(c2s.grad_alpha.size());
    for (size_t i = 0; i < c2s.grad_alpha.size(); ++i) {
      const auto ga = c2s.bases[i] * c2s.grad_alpha[i];
      grad_alpha.push_back({-ga(0), -ga(1), -ga(2)});
    }
    writer.write_scalars("grad_alpha", grad_alpha);

    writer.close();
  }
}

//==============================================================================
TEST_CASE("curve_to_streamline_dg",
          "[curve_to_streamline][numerical][dg][doublegyre]") {
  spacetime_field stdg{numerical::doublegyre{}};

  curve_to_streamline c2s;
  const double initial_stepsize = 0.01;
  const double delta = 0.999;
  const size_t n = 10000;
  const double t0 = 0;
  auto current_stepsize = initial_stepsize;
  line<double, 3> l;
  for (auto t : linspace(0.0, 10.0, 21)) { l.push_back({1, 0.2, t}); }
  
  for (size_t i = 0; i < n; ++i) {
    auto new_l = c2s(stdg, t0, l, current_stepsize, delta, 1);
    write_step(stdg, t0, l, c2s, "stdg_line_" + std::to_string(i) + ".vtk");
    l = std::move(new_l);
    current_stepsize *= delta;
  }
}
//==============================================================================
TEST_CASE("curve_to_streamline_orbit",
          "[curve_to_streamline][numerical][orbit]") {
  numerical::orbit v;
  line<double, 3> l0, l1;
  for (auto t : linspace(0.0, 2*M_PI, 20)) {
    l0.push_back({cos(t)*2, sin(t)*2, 0});
    l1.push_back({cos(t)*0.5, sin(t)*0.5, 0});
  }
  l0.pop_back();
  l1.pop_back();
  l0.set_closed(true);
  l1.set_closed(true);

  curve_to_streamline c2s0, c2s1;
  const double initial_stepsize = 0.01;
  const double delta = 0.999;
  const size_t n = 10000;
  const double t0 = 0;
  auto current_stepsize = initial_stepsize;
  
  for (size_t i = 0; i < n; ++i) {
    auto new_l0 = c2s0(v, t0, l0, current_stepsize, delta, 1);
    auto new_l1 = c2s1(v, t0, l1, current_stepsize, delta, 1);
    write_step(v, t0, l0, c2s0, "orbit_line0_" + std::to_string(i) + ".vtk");
    write_step(v, t0, l1, c2s1, "orbit_line1_" + std::to_string(i) + ".vtk");
    l0 = std::move(new_l0);
    l1 = std::move(new_l1);
    current_stepsize *= delta;
  }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
