#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/tensor.h>
#include <tatooine/vtk_legacy.h>

#include "concepts.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <fixed_dims_flowmap_c<2> Flowmap>
struct autonomous_particle {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using this_t    = autonomous_particle<Flowmap>;
  using flowmap_t = std::decay_t<Flowmap>;
  static constexpr auto num_dimensions() { return flowmap_t::num_dimensions(); }
  using real_t = typename flowmap_t::real_t;
  using vec_t  = vec<real_t, num_dimensions()>;
  using mat_t  = mat<real_t, num_dimensions(), num_dimensions()>;
  using pos_t  = vec_t;

  static constexpr real_t max_cond = 4.01;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  flowmap_t m_phi;
  pos_t     m_x0, m_x1, m_prev_pos;
  real_t    m_t1, m_prev_time;
  mat_t     m_nabla_phi1;
  real_t    m_current_radius;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename V, std::floating_point VReal, arithmetic RealX0>
  autonomous_particle(const vectorfield<V, VReal, 2>&      v,
                      vec<RealX0, num_dimensions()> const& x0,
                      arithmetic auto const                start_time,
                      arithmetic auto const                current_radius)
      : autonomous_particle{flowmap(v), x0, start_time, current_radius} {}
  //----------------------------------------------------------------------------
  template <arithmetic RealX0>
  autonomous_particle(flowmap_t phi, vec<RealX0, num_dimensions()> const& x0,
                      arithmetic auto const start_time,
                      arithmetic auto const current_radius)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x0},
        m_prev_pos{x0},
        m_t1{static_cast<real_t>(start_time)},
        m_prev_time{static_cast<real_t>(start_time)},
        m_nabla_phi1{mat_t::eye()},
        m_current_radius{current_radius} {}
  //----------------------------------------------------------------------------
  autonomous_particle(flowmap_t phi, pos_t const& x0, pos_t const& x1,
                      pos_t const& prev_pos, real_t const t1,
                      real_t const prev_time, mat_t const& nabla_phi1,
                      real_t const current_radius)
      : m_phi{std::move(phi)},
        m_x0{x0},
        m_x1{x1},
        m_prev_pos{prev_pos},
        m_t1{t1},
        m_prev_time{prev_time},
        m_nabla_phi1{nabla_phi1},
        m_current_radius{current_radius} {}
  //----------------------------------------------------------------------------
 public:
  autonomous_particle(const autonomous_particle&)     = default;
  autonomous_particle(autonomous_particle&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const autonomous_particle&) -> autonomous_particle& = default;
  auto operator               =(autonomous_particle&&) noexcept
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
  auto nabla_phi1() const -> auto const& { return m_nabla_phi1; }
  auto current_radius() const { return m_current_radius; }
  auto phi() const -> auto const& { return m_phi; }
  auto phi() -> auto& { return m_phi; }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto integrate(real_t tau_step, real_t const max_t) const {
    std::vector<this_t>                             particles{*this};
    std::vector<line<real_t, num_dimensions() + 1>> ellipses;

    size_t size_before = size(particles);
    size_t start_idx   = 0;
    do {
      // this is necessary because after a resize of particles the currently
      // used adresses are not valid anymore
      particles.reserve(size(particles) + size(particles) * 3);
      size_before = size(particles);
      for (size_t i = start_idx; i < size_before; ++i) {
        particles[i].integrate_until_split(tau_step, particles, ellipses,
                                           max_t);
      }
      start_idx = size_before;
    } while (size_before != size(particles));
    return std::pair{std::move(particles), std::move(ellipses)};
  }
  //----------------------------------------------------------------------------
  void integrate_until_split(
      real_t tau_step, std::vector<this_t>& particles,
      std::vector<line<real_t, num_dimensions() + 1>>& ellipses,
      real_t const                                     max_t) const {
    // add initial circle
    auto&        ellipse = ellipses.emplace_back();
    size_t const n       = 100;
    for (auto t : linspace{0.0, M_PI * 2, n}) {
      ellipse.push_back(vec{std::cos(t) * m_current_radius + m_x1(0),
                            std::sin(t) * m_current_radius + m_x1(1), t1()});
    }

    if (m_t1 >= max_t) { return; }
    static real_t const threequarters = real_t(3) / real_t(4);
    // static real_t const sqrt3     = std::sqrt(real_t(3));

    real_t tau       = 0;
    auto   nabla_phi = diff(m_phi);
    if constexpr (is_flowmap_gradient_central_differences(nabla_phi)) {
      nabla_phi.set_epsilon(current_radius());
    }
    real_t                                   cond = 1;
    typename decltype(nabla_phi)::gradient_t nabla_phi2;
    using eig_t =
        decltype(eigenvectors_sym(nabla_phi2 * transpose(nabla_phi2)));
    eig_t                        eig_left;
    [[maybe_unused]] auto const& eigvecs_left_cauchy_green = eig_left.first;
    [[maybe_unused]] auto const& eigvals_left_cauchy_green = eig_left.second;
    // eig_t                        eig_right;
    //[[maybe_unused]] auto const& eigvecs_right_cauchy_green = eig_right.first;
    //[[maybe_unused]] auto const& eigvals_right_cauchy_green =
    // eig_right.second;

    while (cond < 4 && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) { tau = max_t - m_t1; }
      nabla_phi2 = nabla_phi(m_x1, m_t1, tau);
      eig_left   = eigenvectors_sym(nabla_phi2 * transpose(nabla_phi2));
      // eig_right  = eigenvectors_sym(transpose(nabla_phi2) * nabla_phi2);

      cond = eigvals_left_cauchy_green(1) / eigvals_left_cauchy_green(0);
      if (cond > max_cond) {
        cond = 0;
        tau -= tau_step;
        tau_step *= 0.5;
      }
    }
    mat_t const  fmg2fmg1 = nabla_phi2 * m_nabla_phi1;
    pos_t const  x2       = m_phi(m_x1, m_t1, tau);
    real_t const t2       = m_t1 + tau;
    if (cond > 4) {
      vec_t const offset2 = threequarters * std::sqrt(2.0) * m_current_radius *
                            normalize(eigvecs_left_cauchy_green.col(1));
      vec_t const offset0 = inv(fmg2fmg1) * offset2;

      particles.emplace_back(m_phi, m_x0 - offset0, x2 - offset2, m_x1, t2,
                             m_t1, fmg2fmg1,
                             m_current_radius * std::sqrt(2) / 4);
      particles.emplace_back(m_phi, m_x0, x2, m_x1, t2, m_t1, fmg2fmg1,
                             m_current_radius * std::sqrt(2) / 2);
      particles.emplace_back(m_phi, m_x0 + offset0, x2 + offset2, m_x1, t2,
                             m_t1, fmg2fmg1,
                             m_current_radius * std::sqrt(2) / 4);

      auto& ellipse = ellipses.emplace_back();

      auto const a = std::sqrt(eigvals_left_cauchy_green(1)) * m_current_radius;
      auto const b = std::sqrt(eigvals_left_cauchy_green(0)) * m_current_radius;
      auto const alpha = angle(vec{1.0, 0.0}, eigvecs_left_cauchy_green.col(1));
      for (auto t : linspace{0.0, M_PI * 2, n}) {
        vec p{std::cos(t) * a, std::sin(t) * b};

        p = mat{{std::cos(alpha), -std::sin(alpha)},
                {std::sin(alpha), std::cos(alpha)}} *
            p;

        p += x2;
        ellipse.push_back(vec{p(0), p(1), t2});
      }
    } else {
      // particles.emplace_back(m_phi, m_x0, x2, m_x1, t2, m_t1, fmg2fmg1,
      //                       m_current_radius);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, arithmetic RealX0, size_t N>
autonomous_particle(const vectorfield<V, VReal, 2>& v, vec<RealX0, N> const&,
                    arithmetic auto const, arithmetic auto const)
    -> autonomous_particle<decltype(flowmap(v))>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <fixed_dims_flowmap_c<2> Flowmap, arithmetic RealX0, size_t N>
autonomous_particle(const Flowmap& flowmap, vec<RealX0, N> const&,
                    arithmetic auto const, arithmetic auto const)
    -> autonomous_particle<Flowmap>;
//==============================================================================
template <fixed_dims_flowmap_c<2> Flowmap>
void write_vtk(std::vector<autonomous_particle<Flowmap>> const& particles,
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
