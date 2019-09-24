#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H

#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/algorithm_ext/iota.hpp>
#include <cassert>
#include <deque>
#include <stdexcept>
#include "interpolation.h"
#include "tensor.h"
#include "vtk_legacy.h"

//==============================================================================
namespace tatooine {
//==============================================================================

struct forward_t {};
static constexpr inline forward_t forward;

struct backward_t {};
static constexpr inline backward_t backward;

struct central_t {};
static constexpr inline central_t central;

struct quadratic_t {};
static constexpr inline quadratic_t quadratic;

//==============================================================================
template <typename Real, size_t N>
struct line {
  struct empty_exception : std::exception {};

  //============================================================================
  using real_t          = Real;
  using vec_t           = vec<Real, N>;
  using pos_t           = vec_t;
  using vec3            = vec<Real, 3>;
  using mat3            = mat<Real, 3, 3>;
  using this_t          = line<Real, N>;
  using pos_container_t = std::deque<pos_t>;

  static constexpr auto num_dimensions() noexcept { return N; }

  //============================================================================
 private:
  pos_container_t m_vertices;
  bool            m_is_closed = false;

  //============================================================================
 public:
  line()                      = default;
  line(const line& other)     = default;
  line(line&& other) noexcept = default;
  line& operator=(const line& other) = default;
  line& operator=(line&& other) noexcept = default;

  //----------------------------------------------------------------------------
  line(const pos_container_t& data, bool is_closed = false)
      : m_vertices{data}, m_is_closed{is_closed} {}

  //----------------------------------------------------------------------------
  line(pos_container_t&& data, bool is_closed = false)
      : m_vertices{std::move(data)}, m_is_closed{is_closed} {}

  //----------------------------------------------------------------------------
  line(std::initializer_list<pos_t>&& data)
      : m_vertices{std::move(data)}, m_is_closed{false} {}

  //----------------------------------------------------------------------------
  const auto& vertices() const { return m_vertices; }
  auto&       vertices() { return m_vertices; }
  //----------------------------------------------------------------------------
  auto size() const { return m_vertices.size(); }
  //----------------------------------------------------------------------------
  auto empty() const { return m_vertices.empty(); }
  //----------------------------------------------------------------------------
  const auto& operator[](size_t i) const { return m_vertices.at(i); }
  auto&       operator[](size_t i) { return m_vertices.vertex_at(i); }
  //----------------------------------------------------------------------------
  const auto& front() const { return m_vertices.front(); }
  auto&       front() { return m_vertices.front(); }
  //----------------------------------------------------------------------------
  const auto& back() const { return m_vertices.back(); }
  auto&       back() { return m_vertices.back(); }
  //----------------------------------------------------------------------------
  auto&       at(size_t i) { return m_vertices[i]; }
  const auto& at(size_t i) const { return m_vertices[i]; }
  //----------------------------------------------------------------------------
  const auto& front_vertex() const { return m_vertices.front(); }
  auto&       front_vertex() { return m_vertices.front(); }
  //----------------------------------------------------------------------------
  const auto& back_vertex() const { return m_vertices.back(); }
  auto&       back_vertex() { return m_vertices.back(); }
  //----------------------------------------------------------------------------
  auto&       vertex_at(size_t i) { return m_vertices[i]; }
  const auto& vertex_at(size_t i) const { return m_vertices[i]; }
  //----------------------------------------------------------------------------
  void push_back(const pos_t& p) { m_vertices.push_back(p); }
  void pop_back() { m_vertices.pop_back(); }
  //----------------------------------------------------------------------------
  void push_front(const pos_t& p) { m_vertices.push_front(p); }
  void pop_front() { m_vertices.pop_front(); }
  //----------------------------------------------------------------------------
  /// calculates tangent at point i with forward differences
  auto tangent(const size_t i, forward_t /*fw*/) const {
    assert(size() > 1);
    if (is_closed()) {
      if (i == size() - 1) {
        return (front_vertex() - vertex_at(i)) /
               distance(front_vertex(), vertex_at(i));
      }
    }
    return (vertex_at(i + 1) - vertex_at(i)) /
           distance(vertex_at(i), vertex_at(i + 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with backward differences
  auto tangent(const size_t i, backward_t /*bw*/) const {
    assert(size() > 1);
    if (is_closed()) {
      if (i == 0) {
        return (vertex_at(i) - back_vertex()) /
               distance(back_vertex(), vertex_at(i));
      }
    }
    return (vertex_at(i) - vertex_at(i - 1)) /
           distance(vertex_at(i), vertex_at(i - 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with central differences
  auto tangent(const size_t i, central_t /*c*/) const {
    if (is_closed()) {
      if (i == 0) {
        return (vertex_at(i+1) - back_vertex()) /
               (distance(back_vertex(), vertex_at(i)) +
                distance(vertex_at(i), vertex_at(i+1)));
      } else if (i == size() - 1) {
        return (front_vertex() - vertex_at(i - 1)) /
               (distance(vertex_at(i - 1), vertex_at(i)) +
                distance(vertex_at(i), front_vertex()));
      }
    }
    return (vertex_at(i + 1) - vertex_at(i - 1)) /
           (distance(vertex_at(i - 1), vertex_at(i)) +
            distance(vertex_at(i), vertex_at(i + 1)));
  }

  //----------------------------------------------------------------------------
  auto tangent(const size_t i) const {
    if (is_closed()) { return tangent(i, central); }
    if (i == 0) { return tangent(i, forward); }
    if (i == size() - 1) { return tangent(i, backward); }
    return tangent(i, central);
  }

  //----------------------------------------------------------------------------
  auto length() {
    Real len = 0;
    for (size_t i = 0; i < this->size() - 1; ++i) {
      len += norm(vertex_at(i) - vertex_at(i + 1));
    }
    return len;
  }

  //----------------------------------------------------------------------------
  bool is_closed() const { return m_is_closed; }
  void set_closed(bool is_closed) { m_is_closed = is_closed; }

  ////----------------------------------------------------------------------------
  ///// filters the line and returns a vector of lines
  // template <typename Pred>
  // std::vector<line<Real, N>> filter(Pred&& pred) const;
  //
  ////----------------------------------------------------------------------------
  ///// filters out all points where the eigenvalue of the jacobian is not real
  // template <typename vf_t>
  // auto filter_only_real_eig_vals(const vf_t& vf) const {
  //  jacobian j{vf};
  //
  //  return filter([&](auto x, auto t, auto) {
  //    auto [eigvecs, eigvals] = eig(j(x, t));
  //    for (const auto& eigval : eigvals) {
  //      if (std::abs(std::imag(eigval)) > 1e-7) { return false; }
  //    }
  //    return true;
  //  });
  //}

  //----------------------------------------------------------------------------
  void write(const std::string& file);

  //----------------------------------------------------------------------------
  static void write(const std::vector<line<Real, N>>& line_set,
                    const std::string&                file);

  //----------------------------------------------------------------------------
  void write_vtk(const std::string& path,
                 const std::string& title          = "tatooine line",
                 bool               write_tangents = false) const;

  //----------------------------------------------------------------------------
  template <size_t N_ = N, std::enable_if_t<N_ == 3>...>
  static auto read_vtk(const std::string& filepath) {
    struct reader : vtk::legacy_file_listener {
      std::vector<std::array<Real, 3>> points;
      std::vector<std::vector<int>>      lines;

      void on_points(
          const std::vector<std::array<Real, 3>>& points_) override {
        points = points_;
      }
      void on_lines(const std::vector<std::vector<int>>& lines_) override {
        lines = lines_;
      }
    } listener;

    vtk::legacy_file file{filepath};
    file.add_listener(listener);
    file.read();

    std::list<line<Real, 3>> lines;
    const auto&                vs = listener.points;
    for (const auto& line : listener.lines) {
      auto& pv_line = lines.emplace_back();
      for (auto i : line) { pv_line.push_back({vs[i][0], vs[i][1], vs[i][2]}); }
    }
    return lines;
  }
};

//==============================================================================
// template <typename Real, size_t N>
// template <typename Pred>
// std::vector<line<Real, N>> line<Real, N>::filter(Pred&& pred) const {
//  std::vector<line<Real, N>> filtered_lines;
//  bool                         need_new_strip = true;
//
//  size_t i      = 0;
//  bool   closed = is_closed();
//  for (const auto [x, t] : *this) {
//    if (pred(x, t, i)) {
//      if (need_new_strip) {
//        filtered_lines.emplace_back();
//        need_new_strip = false;
//      }
//      filtered_lines.back().push_back(x, t);
//    } else {
//      closed         = false;
//      need_new_strip = true;
//      if (!filtered_lines.empty() && filtered_lines.back().size() <= 1)
//        filtered_lines.pop_back();
//    }
//    i++;
//  }
//
//  if (!filtered_lines.empty() && filtered_lines.back().size() <= 1)
//    filtered_lines.pop_back();
//  if (filtered_lines.size() == 1)
//  filtered_lines.front().set_is_closed(closed); return filtered_lines;
//}

//------------------------------------------------------------------------------
template <typename Real, size_t N>
void line<Real, N>::write_vtk(const std::string& path, const std::string& title,
                              bool write_tangents) const {
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  if (writer.is_open()) {
    writer.set_title(title);
    writer.write_header();

    // write points
    std::vector<std::array<Real, 3>> ps;
    ps.reserve(this->size());
    for (const auto& p : vertices()) {
      if constexpr (N == 3) {
        ps.push_back({p(0), p(1), p(2)});
      } else {
        ps.push_back({p(0), p(1), 0});
      }
    }
    writer.write_points(ps);

    // write lines
    std::vector<std::vector<size_t>> line_seq(
        1, std::vector<size_t>(this->size()));
    boost::iota(line_seq.front(), 0);
    if (this->is_closed()) { line_seq.front().push_back(0); }
    writer.write_lines(line_seq);

    writer.write_point_data(this->size());

    // write tangents
    if (write_tangents) {
      std::vector<std::vector<Real>> tangents;
      tangents.reserve(this->size());
      for (size_t i = 0; i < this->size(); ++i) {
        const auto t = tangent(i);
        tangents.push_back({t(0), t(1), t(2)});
      }
      writer.write_scalars("tangents", tangents);
    }

    writer.close();
  }
}

namespace detail {
template <typename LineCont>
void write_line_container_to_vtk(const LineCont& lines, const std::string& path,
                                 const std::string& title) {
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
  if (writer.is_open()) {
    size_t num_pts = 0;
    for (const auto& l : lines) num_pts += l.size();
    std::vector<std::array<typename LineCont::value_type::real_t, 3>> points;
    std::vector<std::vector<size_t>> line_seqs;
    points.reserve(num_pts);
    line_seqs.reserve(lines.size());

    size_t cur_first = 0;
    for (const auto& l : lines) {
      // add points
      for (const auto& p : l.vertices()) {
        if constexpr (LineCont::value_type::num_dimensions() == 3) {
          points.push_back({p(0), p(1), p(2)});
        } else {
          points.push_back({p(0), p(1), 0});
        }
      }

      // add lines
      boost::iota(line_seqs.emplace_back(l.size()), cur_first);
      if (l.is_closed()) { line_seqs.back().push_back(cur_first); }
      cur_first += l.size();
    }

    // write
    writer.set_title(title);
    writer.write_header();
    writer.write_points(points);
    writer.write_lines(line_seqs);
    writer.write_point_data(num_pts);
    writer.close();
  }
}

//------------------------------------------------------------------------------
template <typename Lines, typename MaxDist/*, typename MinAngle*/>
auto merge_line_container(Lines lines, MaxDist max_dist/*, MinAngle min_angle*/) {
  using line_t = typename std::decay_t<Lines>::value_type;
  std::list<line_t> merged;
  merged.emplace_back(std::move(lines.back()));
  lines.pop_back();

  while (!lines.empty()) {
    auto min_d = std::numeric_limits<typename line_t::real_t>::max();
    auto best_it           = std::end(lines);
    bool merged_take_front = false;
    bool it_take_front     = false;
    for (auto it = std::begin(lines); it != std::end(lines); ++it) {
      if (const auto d = tatooine::distance(merged.back().front_vertex(),
                                            it->front_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = true;
      }
      if (const auto d = tatooine::distance(merged.back().back_vertex(),
                                            it->front_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = true;
      }
      if (const auto d = tatooine::distance(merged.back().front_vertex(),
                                            it->back_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = true;
        it_take_front     = false;
      }
      if (const auto d = tatooine::distance(merged.back().back_vertex(),
                                            it->back_vertex());
          d < min_d && d < max_dist) {
        min_d             = d;
        best_it           = it;
        merged_take_front = false;
        it_take_front     = false;
      }
    }

    if (best_it != end(lines)) {
      if (merged_take_front) {
        if (it_take_front) {
          for (const auto& v : best_it->vertices()) {
            merged.back().push_front(v);
          }
        } else {
          for (const auto& v :
               best_it->vertices() | boost::adaptors::reversed) {
            merged.back().push_front(v);
          }
        }
      } else {
        if (it_take_front) {
          for (const auto& v : best_it->vertices()) {
            merged.back().push_back(v);
          }
        } else {
          for (const auto& v :
               best_it->vertices() | boost::adaptors::reversed) {
            merged.back().push_back(v);
          }
        }
      }
      lines.erase(best_it);
    } else {
      merged.emplace_back(std::move(lines.back()));
      lines.pop_back();
    }
  }

  return merged;
}

//------------------------------------------------------------------------------
template <typename Lines, typename Real>
auto filter_length(Lines lines, Real length) {
  for (auto it = begin(lines); it != end(lines);) {
    auto l = it->length();
    ++it;
    if (l < length) { lines.erase(prev(it)); }
  }
  return lines;
}
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
void write_vtk(const std::vector<line<Real, N>>& lines, const std::string& path,
               const std::string& title = "tatooine lines") {
  detail::write_line_container_to_vtk(lines, path, title);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
void write_vtk(const std::list<line<Real, N>>& lines, const std::string& path,
               const std::string& title = "tatooine lines") {
  detail::write_line_container_to_vtk(lines, path, title);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename MaxDist>
auto merge(const std::vector<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::merge_line_container(lines, max_dist);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename MaxDist>
auto merge(const std::list<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::merge_line_container(lines, max_dist);
}
//------------------------------------------------------------------------------
template <typename Real, size_t N, typename MaxDist>
auto filter_length(const std::vector<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::filter_length(lines, max_dist);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename MaxDist>
auto filter_length(const std::list<line<Real, N>>& lines, MaxDist max_dist) {
  return detail::filter_length(lines, max_dist);
}

//==============================================================================
template <typename Real, size_t N>
struct parameterized_line : line<Real, N> {
  using this_t   = parameterized_line<Real, N>;
  using parent_t = line<Real, N>;
  using typename parent_t::empty_exception;
  using typename parent_t::vec_t;
  using typename parent_t::pos_t;
  struct time_not_found : std::exception {};

  using parent_t::size;
  using parent_t::vertices;
  using parent_t::vertex_at;
  using parent_t::tangent;

 private:
  std::deque<Real> m_parameterization;

 public:
  parameterized_line()                              = default;
  parameterized_line(const parameterized_line&)     = default;
  parameterized_line(parameterized_line&&) noexcept = default;
  parameterized_line& operator=(const parameterized_line&) = default;
  parameterized_line& operator=(parameterized_line&&) noexcept = default;
  //----------------------------------------------------------------------------
  parameterized_line(std::initializer_list<std::pair<pos_t, Real>>&& data) {
    for (auto& [pos, param] : data) {
      push_back(std::move(pos), std::move(param));
    }
  }
  //----------------------------------------------------------------------------
  const auto& parameterization() const { return m_parameterization; }
  auto&       parameterization() { return m_parameterization; }
  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> front() {
    return {vertices().front(), m_parameterization.front()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> front() const {
    return {vertices().front(), m_parameterization.front()};
  }

  //----------------------------------------------------------------------------
  std::pair<pos_t&, Real&> back() {
    return {vertices().back(), m_parameterization.back()};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<const pos_t&, const Real&> back() const {
    return {vertices().back(), m_parameterization.back()};
  }

  //----------------------------------------------------------------------------
  std::pair<const pos_t&, const Real&> at(size_t i) const {
    return {vertex_at(i), parameterization_at(i)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<pos_t&, Real&> at(size_t i) {
    return {vertex_at(i), parameterization_at(i)};
  }
  //----------------------------------------------------------------------------
  std::pair<const pos_t&, const Real&> operator[](size_t i) const {
    return {vertex_at(i), parameterization_at(i)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  std::pair<pos_t&, Real&> operator[](size_t i) {
    return {vertex_at(i), parameterization_at(i)};
  }

  //----------------------------------------------------------------------------
  auto&       parameterization_at(size_t i) { return m_parameterization.at(i); }
  const auto& parameterization_at(size_t i) const {
    return m_parameterization.at(i);
  }
  //----------------------------------------------------------------------------
  auto&       front_parameterization() { return m_parameterization.front(); }
  const auto& front_parameterization() const {
    return m_parameterization.front();
  }

  //----------------------------------------------------------------------------
  auto&       back_parameterization() { return m_parameterization.back(); }
  const auto& back_parameterization() const {
    return m_parameterization.back();
  }

  //----------------------------------------------------------------------------
  void push_back(const pos_t& p, Real t) {
    parent_t::push_back(p);
    m_parameterization.push_back(t);
  }
  //----------------------------------------------------------------------------
  void push_back(pos_t&& p, Real t) {
    parent_t::emplace_back(std::move(p));
    m_parameterization.push_back(t);
  }
  //----------------------------------------------------------------------------
  void pop_back() {
    parent_t::pop_back();
    m_parameterization.pop_back();
  }

  //----------------------------------------------------------------------------
  void push_front(const pos_t& p, Real t) {
    parent_t::push_front(p);
    m_parameterization.push_front(t);
  }
  //----------------------------------------------------------------------------
  void push_front(pos_t&& p, Real t) {
    parent_t::emplace_front(std::move(p));
    m_parameterization.push_front(t);
  }
  //----------------------------------------------------------------------------
  void pop_front() {
    parent_t::pop_front();
    m_parameterization.pop_front();
  }

  //----------------------------------------------------------------------------
  /// sample the line via interpolation
  template <template <typename>
            typename interpolator_t = interpolation::hermite>
  auto sample(Real t) const {
    if (this->empty()) { throw empty_exception{}; }

    const auto min_time =
        std::min(m_parameterization.front(), m_parameterization.back());
    const auto max_time =
        std::max(m_parameterization.front(), m_parameterization.back());

    if (std::abs(t - min_time) < 1e-6) {
      t = min_time;
    } else if (std::abs(t - max_time) < 1e-6) {
      t = max_time;
    }

    // calculate two points t is in between
    size_t left  = std::numeric_limits<size_t>::max();
    bool   found = false;
    for (size_t i = 0; i < size() - 1; i++) {
      if ((parameterization_at(i) <= t && t <= parameterization_at(i + 1)) ||
          (parameterization_at(i + 1) <= t && t <= parameterization_at(i))) {
        left  = i;
        found = true;
        break;
      }
    }
    if (!found) { throw time_not_found{}; }
    // interpolate
    Real factor = (t - m_parameterization[left]) /
                  (m_parameterization[left + 1] - m_parameterization[left]);
    interpolator_t<Real> interp;
    return interp.interpolate_iter(next(begin(vertices()), left),
                                   next(begin(vertices()), left + 1),
                                   begin(vertices()), end(vertices()), factor);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <template <typename>
            typename interpolator_t = interpolation::hermite>
  auto operator()(const Real t) const {
    return sample<interpolator_t>(t);
  }

  //============================================================================
  void uniform_parameterization(Real t0 = 0) {
    parameterization_at(0) = t0;
    for (size_t i = 1; i < this->size(); ++i) {
      parameterization_at(i) = parameterization_at(i - 1) + 1;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void chordal_parameterization(Real t0 = 0) {
    parameterization_at(0) = t0;
    for (size_t i = 1; i < this->size(); ++i) {
      parameterization_at(i) = parameterization_at(i - 1) +
                               distance(vertex_at(i), vertex_at(i - 1));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void centripetal_parameterization(Real t0 = 0) {
    parameterization_at(0) = t0;
    for (size_t i = 1; i < this->size(); ++i) {
      parameterization_at(i) =
          parameterization_at(i - 1) +
          std::sqrt(distance(vertex_at(i), vertex_at(i - 1)));
    }
  }

  //----------------------------------------------------------------------------
  /// computes tangent assuming the line is a quadratic curve
  auto tangent(const size_t i, quadratic_t /*q*/) const {
    assert(this->size() > 1);
    // start or end point
    if (!this->is_closed()) {
      if (i == 0) { return at(1) - at(0); }
      if (i == this->size() - 1) { return at(i) - at(i - 1); }
    }

    // point in between
    // const auto& x0 = at(std::abs((i - 1) % this->size()));
    const auto& x0 = at(i - 1);
    const auto& x1 = at(i);
    const auto& x2 = at(i + 1);
    // const auto& x2 = at((i + 1) % this->size());
    const auto t = (parameterization_at(i) - parameterization_at(i - 1)) /
                   (parameterization_at(i + 1) - parameterization_at(i - 1));

    // for each component fit a quadratic curve through the neighbor points and
    // the point itself and compute the derivative
    vec_t      tangent;
    const mat3 A{{0.0, 0.0, 1.0}, {t * t, t, 1.0}, {1.0, 1.0, 1.0}};
    for (size_t j = 0; j < N; ++j) {
      vec3 b{x0(j), x1(j), x2(j)};
      auto coeffs = gesv(A, b);

      tangent(j) = 2 * coeffs(0) * t + coeffs(1);
    }
    return tangent;
  }

  //------------------------------------------------------------------------------
  void write_vtk(const std::string& path,
                 const std::string& title = "tatooine parameterized line",
                 bool               write_tangents = false) const {
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      std::vector<std::array<Real, 3>> ps;
      ps.reserve(this->size());
      for (const auto& p : vertices()) {
        if constexpr (N == 3) {
          ps.push_back({p(0), p(1), p(2)});
        } else {
          ps.push_back({p(0), p(1), 0});
        }
      }
      writer.write_points(ps);

      // write lines
      std::vector<std::vector<size_t>> line_seq(
          1, std::vector<size_t>(this->size()));
      boost::iota(line_seq.front(), 0);
      writer.write_lines(line_seq);

      writer.write_point_data(this->size());

      // write tangents
      if (write_tangents) {
        std::vector<std::vector<Real>> tangents;
        tangents.reserve(this->size());
        for (size_t i = 0; i < this->size(); ++i) {
          const auto t = tangent(i);
          tangents.push_back({t(0), t(1), t(2)});
        }
        writer.write_scalars("tangents", tangents);
      }

      // write parameterization
      std::vector<std::vector<Real>> parameterization;
      parameterization.reserve(this->size());
      for (auto t : m_parameterization) { parameterization.push_back({t}); }
      writer.write_scalars("parameterization", parameterization);

      writer.close();
    }
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
