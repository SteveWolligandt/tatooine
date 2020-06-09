#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/diff.h>
#include <tatooine/flowmap.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/tensor.h>

#include "concepts.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N, typename Flowmap>
struct autonomous_particle {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using real_t = Real;
  using vec_t  = vec<Real, N>;
  using mat_t  = mat<Real, N, N>;
  using pos_t  = vec_t;
  using this_t = autonomous_particle<Real, N>;

  static constexpr Real max_cond = 9.01;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  Flowmap m_flowmap;
  pos_t  m_x0, m_x1, m_prev_pos;
  Real   m_t1, m_prev_time;
  mat_t  m_flowmap_gradient1;
  size_t m_level;
  Real   m_start_radius;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename _Flowmap, arithmetic RealX0>
  autonomous_particle(_Flowmap&& flowmap, vec<RealX0, N> const& x0,
                      arithmetic auto const start_time,
                      arithmetic auto const start_radius)
      : m_flowmap{std::forward<_Forward>(flowmap)},
        m_x0{x0},
        m_x1{x0},
        m_prev_pos{x0},
        m_t1{static_cast<Real>(start_time)},
        m_prev_time{static_cast<Real>(start_time)},
        m_flowmap_gradient1{mat_t::eye()},
        m_level{0},
        m_start_radius{start_radius} {}
  //----------------------------------------------------------------------------
  autonomous_particle(const Flowmap& flowmap, pos_t const& x0, pos_t const& x1,
                      pos_t const& prev_pos, Real const t1,
                      Real const prev_time, mat_t const& flowmap_gradient,
                      size_t const level, Real const start_radius)
      : m_flowmap{flowmap},
        m_x0{x0},
        m_x1{x1},
        m_prev_pos{prev_pos},
        m_t1{t1},
        m_prev_time{prev_time},
        m_flowmap_gradient1{flowmap_gradient},
        m_level{level},
        m_start_radius{start_radius} {}
  //----------------------------------------------------------------------------
 public:
  autonomous_particle(const autonomous_particle&)     = default;
  autonomous_particle(autonomous_particle&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const autonomous_particle&)
    -> autonomous_particle& = default;
  auto operator=(autonomous_particle&&) noexcept
    -> autonomous_particle& = default;

  //----------------------------------------------------------------------------
  // getter
  //----------------------------------------------------------------------------
  auto x0() const -> auto const& { return m_x0; }
  auto x0(size_t i) const { return m_x0(i); }
  auto x1() const -> auto const& { return m_x1; }
  auto x1(size_t i) const { return m_x1(i); }
  auto previous_position(size_t i) const { return m_prev_pos(i); }
  auto previous_position() const -> auto const& { return m_prev_pos; }
  auto t1() const { return m_t1; }
  auto previous_time() const { return m_prev_time; }
  auto flowmap_gradient() const -> auto const& { return m_flowmap_gradient1; }
  auto level() const { return m_level; }
  auto current_radius() const {
    static Real const sqrt3 = std::sqrt(Real(3));
    return m_start_radius * std::pow(1 / sqrt3, m_level);
  }
  auto start_radius() const { return m_start_radius; }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto integrate(Real tau_step,
                 Real const max_t) const {
    std::vector<this_t> particles{*this};

    size_t n         = 0;
    size_t start_idx = 0;
    do {
      n                   = size(particles);
      auto const cur_size = size(particles);
      // this is necessary because after a resize of particles the currently
      // used adresses are not valid anymore
      particles.reserve(size(particles) + (cur_size - start_idx) * 3);
      for (size_t i = start_idx; i < cur_size; ++i) {
        particles[i].integrate_until_split(tau_step, particles, max_t);
      }
      start_idx = cur_size;
    } while (n != size(particles));
    return particles;
  }
  //----------------------------------------------------------------------------
  void integrate_until_split(Real tau_step,
                             std::vector<this_t>& particles,
                             Real const           max_t) const {
    if (m_t1 >= max_t) { return; }
    static Real const twothirds = Real(2) / Real(3);
    static Real const sqrt3     = std::sqrt(Real(3));

    Real tau          = 0;
    auto fmgrad_field = diff(
    auto& fm_field = fmgrad_field.internal_field();
    Real  cond     = 1;
    std::pair<mat<Real, N, N>, vec<Real, N>> eig;
    auto const&                              eigvecs = eig.first;
    auto const&                              eigvals = eig.second;

    typename decltype(fmgrad_field)::tensor_t flowmap_gradient2;
    while (cond < 9 && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) { tau = max_t - m_t1; }
      fm_field.set_tau(tau);
      flowmap_gradient2 = fmgrad_field(m_x1, m_t1);
      eig = eigenvectors_sym(flowmap_gradient2 * transpose(flowmap_gradient2));

      cond = eigvals(1) / eigvals(0);
      //if (cond > max_cond) {
      //  cond = 0;
      //  tau -= tau_step;
      //  tau_step *= 0.5;
      //}
    }
    mat_t const fmg2fmg1 = flowmap_gradient2 * m_flowmap_gradient1;
    pos_t const x2       = fm_field(m_x1, m_t1);
    Real const  t2       = m_t1 + tau;
    if (cond > 9) {
      vec_t const offset2 =
          twothirds * sqrt3 * current_radius() * normalize(eigvecs.col(1));
      vec_t const offset0    = inv(fmg2fmg1) * offset2;

      particles.emplace_back(m_x0 - offset0,
                             x2 - offset2,
                             m_x1,
                             t2,
                             m_t1,
                             fmg2fmg1,
                             m_level + 1,
                             m_start_radius);
      particles.emplace_back(m_x0,
                             x2,
                             m_x1,
                             t2,
                             m_t1,
                             fmg2fmg1,
                             m_level + 1,
                             m_start_radius);
      particles.emplace_back(m_x0 + offset0,
                             x2 + offset2,
                             m_x1,
                             t2,
                             m_t1,
                             fmg2fmg1,
                             m_level + 1,
                             m_start_radius);
    } else {
      particles.emplace_back(m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             m_start_radius);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename _Flowmap, arithmetic RealX0, size_t N,
          arithmetic RealStartTime, arithmetic RealStartRadius>
autonomous_particle(const Flowmap& flowmap, vec<RealX0, N> const& x0,
                    RealStartTime const, RealStartRadios const)
    -> autonomous_particle<
        const Flowmap&, promote_t<RealX0, RealStartTime, RealStartRadius>, N>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename _Flowmap, arithmetic RealX0, size_t N,
          arithmetic RealStartTime, arithmetic RealStartRadius>
autonomous_particle(Flowmap&& flowmap, vec<RealX0, N> const& x0,
                    RealStartTime const, RealStartRadios const)
    -> autonomous_particle<std::decay_t<Flowmap>,
                           promote_t<RealX0, RealStartTime, RealStartRadius>,
                           N>;
//==============================================================================
template <std::floating_point Real, size_t N>
void write_vtk(std::vector<autonomous_particle<Real, N>> const& particles,
               arithmetic auto const t0, std::string const& forward_path,
               std::string const& backward_path) {
  vtk::legacy_file_writer writer_forw{forward_path, vtk::POLYDATA},
      write_back{backward_path, vtk::POLYDATA};
  if (writer_forw.is_open()) {
    writer_forw.write_header();

    std::vector<std::vector<size_t>> lines;
    std::vector<vec<double, 3>>      points;
    lines.reserve(size(particles));
    for (auto const& particle : particles) {
      points.push_back(vec{particle.previous_position(0),
                           particle.previous_position(1),
                           particle.previous_time()});
      points.push_back(vec{particle.x1(0), particle.x1(1), particle.t1()});
      lines.push_back(std::vector{size(points) - 2, size(points) - 1});
    }
    writer_forw.write_points(points);
    writer_forw.write_lines(lines);

    writer_forw.close();
  }
  if (write_back.is_open()) {
    write_back.write_header();

    std::vector<std::vector<size_t>> lines;
    std::vector<vec<double, 3>>      points;
    lines.reserve(size(particles));
    for (auto const& particle : particles) {
      points.push_back(vec{particle.x0(0), particle.x0(1), t0});
      points.push_back(vec{particle.x1(0), particle.x1(1), particle.t1()});
      lines.push_back(std::vector{size(points) - 2, size(points) - 1});
    }
    write_back.write_points(points);
    write_back.write_lines(lines);

    write_back.close();
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
