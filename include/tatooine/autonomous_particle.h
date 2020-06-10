#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/tensor.h>

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

  static constexpr real_t max_cond = 9.01;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  flowmap_t m_flowmap;
  pos_t     m_x0, m_x1, m_prev_pos;
  real_t    m_t1, m_prev_time;
  mat_t     m_nabla_phi1;
  size_t    m_level;
  real_t    m_start_radius;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename V, std::floating_point VReal, arithmetic RealX0>
  autonomous_particle(const vectorfield<V, VReal, 2>&      v,
                      vec<RealX0, num_dimensions()> const& x0,
                      arithmetic auto const                start_time,
                      arithmetic auto const                start_radius)
      : autonomous_particle{flowmap(v), x0, start_time, start_radius} {}
  //----------------------------------------------------------------------------
  template <arithmetic RealX0>
  autonomous_particle(flowmap_t                            flowmap,
                      vec<RealX0, num_dimensions()> const& x0,
                      arithmetic auto const                start_time,
                      arithmetic auto const                start_radius)
      : m_flowmap{std::move(flowmap)},
        m_x0{x0},
        m_x1{x0},
        m_prev_pos{x0},
        m_t1{static_cast<real_t>(start_time)},
        m_prev_time{static_cast<real_t>(start_time)},
        m_nabla_phi1{mat_t::eye()},
        m_level{0},
        m_start_radius{start_radius} {}
  //----------------------------------------------------------------------------
  autonomous_particle(flowmap_t flowmap, pos_t const& x0, pos_t const& x1,
                      pos_t const& prev_pos, real_t const t1,
                      real_t const prev_time, mat_t const& nabla_phi1,
                      size_t const level, real_t const start_radius)
      : m_flowmap{std::move(flowmap)},
        m_x0{x0},
        m_x1{x1},
        m_prev_pos{prev_pos},
        m_t1{t1},
        m_prev_time{prev_time},
        m_nabla_phi1{nabla_phi1},
        m_level{level},
        m_start_radius{start_radius} {}
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
  auto level() const { return m_level; }
  auto current_radius() const {
    static real_t const sqrt3 = std::sqrt(real_t(3));
    return m_start_radius * std::pow(1 / sqrt3, m_level);
  }
  auto start_radius() const { return m_start_radius; }
  auto get_flowmap() const -> auto const& { return m_flowmap; }
  auto get_flowmap() -> auto& { return m_flowmap; }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto integrate(real_t tau_step, real_t const max_t) const {
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
  void integrate_until_split(real_t tau_step, std::vector<this_t>& particles,
                             real_t const max_t) const {
    if (m_t1 >= max_t) { return; }
    static real_t const twothirds = real_t(2) / real_t(3);
    static real_t const sqrt3     = std::sqrt(real_t(3));

    real_t tau       = 0;
    auto   nabla_phi = diff(m_flowmap);
    if constexpr (is_flowmap_gradient_central_differences(nabla_phi)) {
      nabla_phi.set_epsilon(current_radius());
    }
    real_t cond = 1;
    std::pair<mat<real_t, num_dimensions(), num_dimensions()>,
              vec<real_t, num_dimensions()>>
                eig;
    auto const& eigvecs = eig.first;
    auto const& eigvals = eig.second;

    typename decltype(nabla_phi)::gradient_t nabla_phi2;
    while (cond < 9 && m_t1 + tau < max_t) {
      tau += tau_step;
      if (m_t1 + tau > max_t) { tau = max_t - m_t1; }
      nabla_phi2 = nabla_phi(m_x1, m_t1, tau);
      eig        = eigenvectors_sym(nabla_phi2 * transpose(nabla_phi2));

      cond = eigvals(1) / eigvals(0);
      // if (cond > max_cond) {
      //  cond = 0;
      //  tau -= tau_step;
      //  tau_step *= 0.5;
      //}
    }
    mat_t const  fmg2fmg1 = nabla_phi2 * m_nabla_phi1;
    pos_t const  x2       = m_flowmap(m_x1, m_t1, tau);
    real_t const t2       = m_t1 + tau;
    if (cond > 9) {
      vec_t const offset2 =
          twothirds * sqrt3 * current_radius() * normalize(eigvecs.col(1));
      vec_t const offset0 = inv(fmg2fmg1) * offset2;

      particles.emplace_back(m_flowmap, m_x0 - offset0, x2 - offset2, m_x1, t2, m_t1,
                             fmg2fmg1, m_level + 1, m_start_radius);
      particles.emplace_back(m_flowmap, m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             m_start_radius);
      particles.emplace_back(m_flowmap, m_x0 + offset0, x2 + offset2, m_x1, t2, m_t1,
                             fmg2fmg1, m_level + 1, m_start_radius);
    } else {
      particles.emplace_back(m_flowmap, m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             m_start_radius);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// deduction guides
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, arithmetic RealX0, size_t N>
autonomous_particle(const vectorfield<V, VReal, 2>& v,
                    vec<RealX0, N> const&           , arithmetic auto const,
                    arithmetic auto const)
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
