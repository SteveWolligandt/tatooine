#include <tatooine/counterexample_sadlo.h>
#include <tatooine/curve_to_streamline.h>
#include <tatooine/line.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/vtk_legacy.h>
#include <array>
#include <boost/range/adaptor/reversed.hpp>
#include <list>

using namespace boost;
using namespace boost::adaptors;
using namespace tatooine;
using namespace tatooine::numerical;

const std::string fullpath = FULLPATH;
const std::string filepath =
    fullpath +
    "/numerical_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk";

auto read_pv_lines() {
  struct reader : vtk::legacy_file_listener {
    std::vector<std::array<double, 3>> points;
    std::vector<std::vector<int>>      lines;

    void on_points(const std::vector<std::array<double, 3>> &points_) override {
      points = points_;
    }
    void on_lines(const std::vector<std::vector<int>> &lines_) override {
      lines = lines_;
    }
  } listener;

  vtk::legacy_file file{filepath};
  file.add_listener(listener);
  file.read();

  std::list<line<double, 3>> pv_lines;
  const auto&                vs = listener.points;
  for (const auto& line : listener.lines) {
    auto& pv_line = pv_lines.emplace_back();
    for (auto i : line) {
      pv_line.push_back({vs[i][0], vs[i][1], vs[i][2]});
    }
  }
  return pv_lines;
}

auto merge_lines() {
  auto pv_lines = read_pv_lines();
  auto merged   = pv_lines.front();
  pv_lines.pop_front();

  while (!pv_lines.empty()) {
    auto   min_d             = std::numeric_limits<double>::max();
    auto   best_it           = std::end(pv_lines);
    bool   merged_take_front = false;
    bool   it_take_front     = false;
    for (auto it = std::begin(pv_lines); it != std::end(pv_lines); ++it) {
      if (const auto d =
	      tatooine::distance(merged.front_vertex(), it->front_vertex());
	  d < min_d) {
	min_d             = d;
	best_it           = it;
	merged_take_front = true;
	it_take_front     = true;
      }
      if (const auto d =
	      tatooine::distance(merged.back_vertex(), it->front_vertex());
	  d < min_d) {
	min_d             = d;
	best_it           = it;
	merged_take_front = false;
	it_take_front     = true;
      }
      if (const auto d =
	      tatooine::distance(merged.front_vertex(), it->back_vertex());
	  d < min_d) {
	min_d             = d;
	best_it           = it;
	merged_take_front = true;
	it_take_front     = false;
      }
      if (const auto d =
	      tatooine::distance(merged.back_vertex(), it->back_vertex());
	  d < min_d) {
	min_d             = d;
	best_it           = it;
	merged_take_front = false;
	it_take_front     = false;
      }
    }

    if (merged_take_front) {
      if (it_take_front) {
	for (const auto& v : best_it->vertices()) {
	  merged.push_front(v);
	}
      } else {
	for (const auto& v : best_it->vertices() | reversed) {
	  merged.push_front(v);
	}
      }
    } else {
      if (it_take_front) {
	for (const auto& v : best_it->vertices()) {
	  merged.push_back(v);
	}
      } else {
	for (const auto& v : best_it->vertices() | reversed) {
	  merged.push_back(v);
	}
      }
    }
    pv_lines.erase(best_it);
  }

  return merged;
}

template <typename V, typename T0, typename C2S, typename Line>
void write_step(const V& v, T0 t0, const Line& line, const C2S& c2s,
                const std::string& path) {
  vtk::LegacyFileWriter<double> writer(path, vtk::POLYDATA);
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
int main() {
  spacetime_field v{counterexample_sadlo{}};

  curve_to_streamline c2s;
  const double initial_stepsize = 0.01;
  const double delta = 0.999;
  const size_t n = 10000;
  const double t0 = 0;
  auto current_stepsize = initial_stepsize;
  auto l = merge_lines();
  
  for (size_t i = 0; i < n; ++i) {
    auto new_l = c2s(v, t0, l, current_stepsize, delta, 1);
    write_step(v, t0, l, c2s, "counterexample_pv_to_streamline_progression_" + std::to_string(i) + ".vtk");
    l = std::move(new_l);
    current_stepsize *= delta;
  }
}
