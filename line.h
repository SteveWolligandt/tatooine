#ifndef TATOOINE_LINE_H
#define TATOOINE_LINE_H

#include <cassert>
#include <deque>
#include <stdexcept>
#include "tensor.h"
#include "field.h"
#include "interpolation.h"
//#include "vtk_legacy.h"

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
struct line  {
  struct empty_exception : std::exception {};

  //============================================================================
  using real_t   = Real;
  using vec_t    = tensor<real_t, N>;
  using pos_t    = vec_t;
  using vec3     = tensor<real_t, 3>;
  using mat3     = tensor<real_t, 3, 3>;
  using this_t   = line<real_t, N>;
  using pos_container_t = std::deque<pos_t>;

  //============================================================================
 private:
  pos_container_t m_positions;
  bool m_is_closed = false;

  //============================================================================
 public:
  line()                      = default;
  line(const line& other)     = default;
  line(line&& other) noexcept = default;
  line& operator=(const line& other) = default;
  line& operator=(line&& other) noexcept = default;

  //----------------------------------------------------------------------------
  line(const pos_container_t& data, bool is_closed = false)
      : m_positions{data}, m_is_closed{is_closed} {}

  //----------------------------------------------------------------------------
  line(pos_container_t&& data, bool is_closed = false)
      : m_positions{std::move(data)}, m_is_closed{is_closed} {}

  //----------------------------------------------------------------------------
  line(std::initializer_list<pos_t>&& data)
      : m_positions{std::move(data)}, m_is_closed{false} {}

  //----------------------------------------------------------------------------
  //explicit line(const std::string& path) { read(path); }

  //----------------------------------------------------------------------------
  auto size() const { return m_positions.size(); }
  //----------------------------------------------------------------------------
  auto&       at(size_t i) { return m_positions[i]; }
  const auto& at(size_t i) const { return m_positions[i]; }

  //----------------------------------------------------------------------------
  void push_back(const pos_t& p) { m_positions.push_back(p); }
  //----------------------------------------------------------------------------
  void push_front(const pos_t& p) {m_positions.push_front(p); }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with forward differences
  auto tangent(const size_t i, forward_t /*fw*/) const {
    assert(size() > 1);
    if (i == this->size() - 1) {
      return normalize(at(i) - at(i - 1));
    }
    return normalize(at(i + 1) - at(i));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with backward differences
  auto tangent(const size_t i, backward_t /*bw*/) const  {
    assert(size() > 1);
    if (i == 0) { return normalize(at(i + 1) - at(i)); }
    return normalize(at(i) - at(i - 1));
  }

  //----------------------------------------------------------------------------
  /// calculates tangent at point i with central differences
  auto tangent(const size_t i, central_t /*c*/) const  {
    assert(size() > 1);
    if (i == 0) { return normalize(at(i+1) - at(i)) ; }
    if (i == this->size() - 1) {
      return normalize(at(i) - at(i - 1));
    }
    return (at(i + 1) - at(i - 1)) /
           (distance(at(i - 1), at(i)) +
            distance(at(i), at(i + 1)));
  }

  //----------------------------------------------------------------------------
  auto tangent(const size_t i) const {
    if (i == 0 && !is_closed()) { return tangent(i, forward); }
    if (i == size() - 1 && !is_closed()) { return tangent(i, backward); }
    return tangent(i, central);
  }

  //----------------------------------------------------------------------------
  /// computes tangent assuming the line is a quadratic curve
  //auto tangent(const size_t i, quadratic_t [>q<]) const {
  //  assert(this->size() > 1);
  //  // start or end point
  //  if (!is_closed()) {
  //    if (i == 0) { return at(1) - at(0); }
  //    if (i == this->size() - 1) { return at(i) - at(i - 1); }
  //  }
  //
  //  // point in between
  //  // const auto& x0 = at(std::abs((i - 1) % this->size()));
  //  const auto& x0 = at(i - 1);
  //  const auto& x1 = at(i);
  //  const auto& x2 = at(i + 1);
  //  // const auto& x2 = at((i + 1) % this->size());
  //  const auto t =
  //      (time_at(i) - time_at(i - 1)) / (time_at(i + 1) - time_at(i - 1));
  //
  //  // for each component fit a quadratic curve through the neighbor points and
  //  // the point itself and compute the derivative
  //  vec_t tangent;
  //  const mat3 A{0, t * t, 1, 0, t, 1, 1, 1, 1};
  //  std::cout << A << '\n';
  //  for (size_t j = 0; j < N; ++j) {
  //    vec3 b{x0(j), x1(j), x2(j)};
  //    auto coeffs = gesv(A, b);
  //
  //    tangent(j) = 2 * coeffs(0) * t + coeffs(1);
  //  }
  //  return tangent;
  //}

  //----------------------------------------------------------------------------
  auto length() {
    real_t len = 0;
    for (size_t i = 0; i < this->size() - 1; ++i) {
      len += norm(at(i) - at(i + 1));
    }
    return len;
  }

  //----------------------------------------------------------------------------
  bool is_closed() const { return m_is_closed; }

  //----------------------------------------------------------------------------
  void set_is_closed(bool is_closed) { m_is_closed = is_closed; }

  ////----------------------------------------------------------------------------
  ///// filters the line and returns a vector of lines
  //template <typename Pred>
  //std::vector<line<real_t, N>> filter(Pred&& pred) const;
  //
  ////----------------------------------------------------------------------------
  ///// filters out all points where the eigenvalue of the jacobian is not real
  //template <typename vf_t>
  //auto filter_only_real_eig_vals(const vf_t& vf) const {
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
  //void write(const std::string& file);

  //----------------------------------------------------------------------------
  //static void write_line_set(const std::vector<line<real_t, N>>& line_set,
  //                          const std::string&                  file);

  //----------------------------------------------------------------------------
  //void read(const std::string& file);

  //----------------------------------------------------------------------------
  //static auto read_line_set(const std::string& file);

  //----------------------------------------------------------------------------
  //void write_vtk(const std::string& path, const std::string& title,
  //               bool write_tangents = false) const;

  //----------------------------------------------------------------------------
  //void write_obj(std::ostream& out, size_t first_index = 1) const;

  //----------------------------------------------------------------------------
  //void write_obj(const std::string& file, size_t first_index = 1) const;

  //----------------------------------------------------------------------------
  //void write_to_vtk_legacy(const std::string& file) const;

  //----------------------------------------------------------------------------
  //static void write_obj(const std::vector<line>& line_set,
  //                      const std::string&       file);

  //----------------------------------------------------------------------------
  //auto& print(std::ostream& out) const {
  //  for (const auto [x, t] : *this) {
  //    out << "[";
  //    for (const auto& c : x) out << c << ' ';
  //    out << ", " << t << "]\n";
  //  }
  //  return out;
  //}
};

//==============================================================================
//template <typename real_t, std::size_t N>
//template <typename Pred>
//std::vector<line<real_t, N>> line<real_t, N>::filter(Pred&& pred) const {
//  std::vector<line<real_t, N>> filtered_lines;
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
//  if (filtered_lines.size() == 1) filtered_lines.front().set_is_closed(closed);
//  return filtered_lines;
//}
//
////------------------------------------------------------------------------------
//template <typename real_t, std::size_t N>
//void line<real_t, N>::write_vtk(const std::string& path,
//                                const std::string& title,
//                                bool               write_tangents) const {
//  vtk::LegacyFileWriter<real_t> writer(path, vtk::POLYDATA);
//  if (writer.is_open()) {
//    writer.set_title(title);
//    writer.write_header();
//
//    // write points
//    std::vector<std::array<real_t, 3>> ps;
//    ps.reserve(this->size());
//    for (const auto& p : points()) {
//      if constexpr (N == 3) {
//        ps.push_back({p(0), p(1), p(2)});
//      } else {
//        ps.push_back({p(0), p(1), 0});
//      }
//    }
//    writer.write_points(ps);
//
//    // write lines
//    std::vector<std::vector<size_t>> line_seq(
//        1, std::vector<size_t>(this->size()));
//    boost::iota(line_seq.front(), 0);
//    writer.write_lines(line_seq);
//
//    writer.write_point_data(this->size());
//
//    // write tangents
//    if (write_tangents) {
//      std::vector<std::vector<real_t>> tangents;
//      tangents.reserve(this->size());
//      for (size_t i = 0; i < this->size(); ++i) {
//        const auto t = tangent_central_differences(i);
//        tangents.push_back({t(0), t(1), t(2)});
//      }
//      writer.write_scalars("tangents", tangents);
//    }
//
//    // write parameterization
//    std::vector<std::vector<real_t>> parameterization;
//    parameterization.reserve(this->size());
//    for (const auto& t : times()) { parameterization.push_back({t}); }
//    writer.write_scalars("time", parameterization);
//
//    writer.close();
//  }
//}
//
////------------------------------------------------------------------------------
//template <typename real_t, std::size_t N>
//void write_vtk(const std::vector<line<real_t, N>>& lines,
//               const std::string&                  path,
//               const std::string&                  title = "tatooine lines",
//               bool                                write_tangents = false) {
//  vtk::LegacyFileWriter<real_t> writer(path, vtk::POLYDATA);
//  if (writer.is_open()) {
//    size_t num_pts = 0;
//    for (const auto& l : lines) num_pts += l.size();
//    std::vector<std::array<real_t, 3>> points;
//    std::vector<std::vector<size_t>>   line_seqs;
//    std::vector<std::vector<real_t>>   parameterization;
//    std::vector<std::vector<real_t>>   tangents;
//    points.reserve(num_pts);
//    line_seqs.reserve(lines.size());
//    parameterization.reserve(num_pts);
//    if (write_tangents) { tangents.reserve(num_pts); }
//
//    size_t cur_first = 0;
//    for (const auto& l : lines) {
//      // add points
//      for (const auto& p : l.points()) {
//        if constexpr (N == 3) {
//          points.push_back(p);
//        } else {
//          points.push_back({p(0), p(1), 0});
//        }
//      }
//
//      // add parameterization
//      for (const auto& t : l.times()) { parameterization.push_back({t}); }
//
//      // add tangents
//      if (write_tangents) {
//        for (size_t i = 0; i < l.size(); ++i) {
//          const auto t = l.tangent_central_differences(i);
//          tangents.push_back({t(0), t(1), t(2)});
//        }
//      }
//
//      // add lines
//      boost::iota(line_seqs.emplace_back(l.size()), cur_first);
//      cur_first += l.size();
//    }
//
//    // write
//    writer.set_title(title);
//    writer.write_header();
//    writer.write_points(points);
//    writer.write_lines(line_seqs);
//    writer.write_point_data(num_pts);
//    writer.write_scalars("time", parameterization);
//    if (write_tangents) { writer.write_scalars("tangents", tangents); }
//    writer.close();
//  }
//}
//
////------------------------------------------------------------------------------
//template <typename real_t, std::size_t N>
//void line<real_t, N>::write_obj(std::ostream& out, size_t first_index) const {
//  if constexpr (N == 2)
//    for (const auto x : points())
//      out << "v " << std::setprecision(20) << x(0) << " "
//          << std::setprecision(20) << x(1) << " 0\n";
//  else
//    for (const auto x : points()) out << "v " << x.t();
//  out << '\n';
//  for (size_t i = 0; i < this->size() - 1; i++)
//    out << "l " << first_index + i << " " << first_index + i + 1 << '\n';
//  if (is_closed())
//    out << "l " << first_index + this->size() - 1 << " " << first_index << '\n';
//  out << '\n';
//}
//
////------------------------------------------------------------------------------
//template <typename real_t, std::size_t n>
//void line<real_t, N>::write_obj(const std::string& path,
//                                size_t [>first_index<]) const {
//  std::fstream fout(path, std::fstream::out);
//
//  if (fout.is_open()) {
//    write_obj(fout);
//    fout.close();
//  } else
//    throw std::runtime_error("Could not open file " + path);
//}

//------------------------------------------------------------------------------
//template <typename real_t, std::size_t N>
//void line<real_t, N>::write_to_vtk_legacy(const std::string& path) const {
//  vtk::LegacyFileWriter<real_t> writer(path, vtk::POLYDATA);
//  if (writer.is_open()) {
//    writer.set_title("line");
//    writer.write_header();
//
//    // write points
//    std::vector<std::array<real_t, 3>> ps;
//    for (const auto& p : points())
//      if constexpr (N == 3)
//        ps.push_back({p(0), p(1), p(2)});
//      else
//        ps.push_back({p(0), p(1), 0});
//    writer.write_points(ps);
//
//    // write lines
//    std::vector<std::vector<size_t>> lines;
//    lines.emplace_back(ps.size());
//    std::iota(begin(lines.back()), end(lines.back()), 0);
//    writer.write_lines(lines);
//    writer.close();
//  }
//}

//------------------------------------------------------------------------------
//template <typename real_t, std::size_t N>
//void line<real_t, N>::write_obj(const std::vector<line>& line_set,
//                                const std::string&       path) {
//  std::fstream fout(path, std::fstream::out);
//  if (fout.is_open()) {
//    std::size_t i           = 0;
//    size_t      first_index = 1;
//    for (const auto& line : line_set) {
//      fout << "o line" << std::to_string(i) << '\n';
//      i++;
//      line.write_obj(fout, first_index);
//      first_index += line.size();
//    }
//    fout.close();
//  } else
//    throw std::runtime_error("Could not open file " + path);
//}

//==============================================================================
template <typename real_t, size_t N>
struct parameteized_line : line<real_t, N> {
  using this_t = parameteized_line<real_t, N>;
  using parent_t = line<real_t, N>;
  using typename parent_t::pos_t;
  struct time_not_found : std::exception {};
  private:
  std::vector<real_t> m_parameterization;

  //----------------------------------------------------------------------------
  /// sample the line via interpolation
  template <template <typename>
            typename interpolator_t = interpolation::hermite>
  auto sample(real_t t) const {
//    if (this->empty()) throw empty_exception{};
//
//    const auto min_time = std::min(front_time(), back_time());
//    const auto max_time = std::max(front_time(), back_time());
//    if (std::abs(t - min_time) < 1e-6)
//      t = min_time;
//    else if (std::abs(t - max_time) < 1e-6)
//      t = max_time;
//    // calculate two points t is in between
//    const_iterator_point lower_it_point = begin_point(),
//                         upper_it_point = begin_point();
//    const_iterator_time lower_it_time   = begin_time(),
//                        upper_it_time   = begin_time();
//    bool found                          = false;
//    for (size_t i = 0; i < this->size() - 1; i++) {
//      if ((time_at(i) <= t && t <= time_at(i + 1)) ||
//          (time_at(i + 1) <= t && t <= time_at(i))) {
//        lower_it_time = const_iterator_time(i, this);
//        upper_it_time = const_iterator_time(i + 1, this);
//
//        lower_it_point = const_iterator_point(i, this);
//        upper_it_point = const_iterator_point(i + 1, this);
//        found          = true;
//        break;
//      }
//    }
//    if (!found) { throw time_not_found{}; }
//    // interpolate
//    interpolator_t<vec_t, real_t> interp;
//    real_t factor = (t - *lower_it_time) / (*upper_it_time - *lower_it_time);
//    auto   s = interp.interpolate(lower_it_point, upper_it_point, begin_point(),
//                                end_point(), factor);
//
    //return s;
  }
  //----------------------------------------------------------------------------
  template <template <typename>
            typename interpolator_t = interpolation::hermite>
  auto operator()(const real_t t) const {
    return sample<interpolator_t>(t);
  }
  ////----------------------------------------------------------------------------
  template <template <typename>
            typename interpolator_t = interpolation::hermite>
  auto refine(size_t level) const {
  //  line<real_t, N> refined_line;
  //  for (size_t j = 0; j < this->size() - 1; ++j) {
  //    for (size_t i = 0; i < level; ++i) {
  //      const auto& t         = time((*this)[j]);
  //      const auto  time_span = time((*this)[j + 1]) - t;
  //      real_t      step      = time_span / real_t(level);
  //      refined_line.push_back(this->sample<interpolator_t>(t + i * step),
  //                             t + i * step);
  //    }
  //  }
  //  refined_line.push_back(point(this->back()), time(this->back()));
  //  return refined_line;
  }
  
  //----------------------------------------------------------------------------
  void uniform_parameterization(real_t t0 = 0) {
    //time_at(0) = t0;
    //for (size_t i = 1; i < this->size(); ++i) {
    //  time_at(i) = time_at(i - 1) + 1;
    //}
  }

  //----------------------------------------------------------------------------
  void chordal_parameterization(real_t t0 = 0) {
    //time_at(0) = t0;
    //for (size_t i = 1; i < this->size(); ++i) {
    //  time_at(i) = time_at(i - 1) + distance(at(i), at(i - 1));
    //}
  }

  //----------------------------------------------------------------------------
  void centripetal_parameterization(real_t t0 = 0) {
    //time_at(0) = t0;
    //for (size_t i = 1; i < this->size(); ++i) {
    //  time_at(i) = time_at(i - 1) + std::sqrt(dist(at(i), at(i - 1)));
    //}
  }
};

//------------------------------------------------------------------------------
template <typename real_t, std::size_t N>
inline auto& operator<<(std::ostream& out, line<real_t, N>& line) {
  return line.print(out);
}

//==============================================================================
/// \defgroup lineiterators line Iterators
/// \{
//==============================================================================
template <typename T>
struct is_line : std::false_type {};

//------------------------------------------------------------------------------
template <typename real_t, std::size_t N>
struct is_line<line<real_t, N>> : std::true_type {};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
