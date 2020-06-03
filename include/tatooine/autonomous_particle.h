#ifndef TATOOINE_AUTONOMOUS_PARTICLES
#define TATOOINE_AUTONOMOUS_PARTICLES
//==============================================================================
#include <tatooine/diff.h>
#include <tatooine/flowmap.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
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

  static constexpr real_t max_cond = 9.01;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  pos_t  m_x0, m_x1, m_prev_pos;
  real_t m_t1, m_prev_time;
  mat_t  m_flowmap_gradient1;
  size_t m_level;
  real_t m_start_radius;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  template <typename RealX0, typename RealTS, typename RealRadius,
            enable_if_arithmetic<RealX0, RealTS, RealRadius> = true>
  autonomous_particle(vec<RealX0, N> const& x0, RealTS const ts)
      : m_x0{x0},
        m_x1{x0},
        m_prev_pos{x0},
        m_t1{static_cast<real_t>(ts)},
        m_prev_time{ts},
        m_flowmap_gradient1{mat_t::eye()},
        m_level{0},
        m_start_radius{radius} {}

  autonomous_particle(pos_t const& x0, pos_t const& x1, pos_t const& from,
                      real_t const t1, real_t const fromt,
                      const mat_t& flowmap_gradient, size_t const level, real_t const r0)
      : m_x0{x0},
        m_x1{x1},
        m_prev_pos{from},
        m_t1{t1},
        m_prev_time{fromt},
        m_flowmap_gradient1{flowmap_gradient},
        m_level{level},
        m_start_radius{r0} {}
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
  auto flowmap_gradient() const -> auto const& { return m_flowmap_gradient1; }
  auto level() const { return m_level; }
  auto current_radius() const { return m_start_radius * std::pow(1 / sqrt3, m_level); }
  auto start_radius() const { return m_start_radius; }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  template <template <typename, size_t, template <typename> typename>
            typename Integrator = integration::vclibs::rungekutta43,
            template <typename>
            typename InterpolationKernel = interpolation::hermite,
            typename V>
  auto integrate(vectorfield<V, Real, N> const& v, real_t tau_step,
                 real_t const max_t) const {
    std::vector<this_t> particles{*this};

    size_t n         = 0;
    size_t start_idx = 0;
    do {
      n                   = size(particles);
      auto const cur_size = size(particles);
      for (size_t i = start_idx; i < cur_size; ++i) {
        particles[i].integrate_until_split(v, tau_step, particles, max_t);
      }
      start_idx = cur_size;
    } while (n != size(particles));
    return particles;
  }
  //----------------------------------------------------------------------------
  template <template <typename, size_t, template <typename> typename>
            typename Integrator = integration::vclibs::rungekutta43,
            template <typename>
            typename InterpolationKernel = interpolation::linear>
  auto create_integrator() const {
    using integrator_t = Integrator<Real, N, InterpolationKernel>;
    return integrator_t{};
  }
  //----------------------------------------------------------------------------
  template <template <typename, size_t, template <typename> typename>
            typename Integrator = integration::vclibs::rungekutta43,
            template <typename>
            typename InterpolationKernel = interpolation::linear,
            typename V>
  void integrate_until_split(vectorfield<V, Real, N> const& v, real_t tau_step,
                             std::vector<this_t>& particles,
                             real_t const         max_t) const {
    if (m_t1 >= max_t) { return; }
    static real_t const twothirds = real_t(2) / real_t(3);
    static real_t const sqrt3     = std::sqrt(real_t(3));

    Real tau          = 0;
    auto fmgrad_field = diff(
        flowmap{v, create_integrator<Integrator, InterpolationKernel>(), tau},
        m_cur_radius);
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
      eig    = eigenvectors_sym(flowmap_gradient2 * transpose(flowmap_gradient2));

      cond = eigvals(1) / eigvals(0);
      if (cond > max_cond) {
        cond = 0;
        tau -= tau_step;
        tau_step *= 0.5;
      }
    }
    mat_t const  fmg2fmg1 = flowmap_gradient2 * m_flowmap_gradient1;
    pos_t const  x2       = fm_field(m_x1, m_t1);
    real_t const t2       = m_t1 + tau;
    if (cond > 9) {
      vec_t const offset2 =
          twothirds * sqrt3 * m_cur_radius * normalize(eigvecs.col(1));
      vec_t const offset0 = inv(fmg2fmg1) * offset2;
      real_t const new_radius = m_start_radius * std::pow(1 / sqrt3, m_level);

      particles.emplace_back(
          m_x0 - offset0, x2 - offset2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
          m_start_radius * std::pow(1 / sqrt3, m_level), m_start_radius);
      particles.emplace_back(m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             new_radius, m_start_radius);
      particles.emplace_back(m_x0 + offset0, x2 + offset2, m_x1, t2, m_t1,
                             fmg2fmg1, m_level + 1, new_radius, m_start_radius);
    } else {
      particles.emplace_back(m_x0, x2, m_x1, t2, m_t1, fmg2fmg1, m_level + 1,
                             m_cur_radius, m_start_radius);
    }
  }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename RealX0, size_t N, typename RealTS, typename RealRadius>
autonomous_particle(vec<RealX0, N> const& x0, RealTS const ts,
                    RealRadius const radius)
    -> autonomous_particle<promote_t<RealX0, RealTS, RealRadius>, N>;
//==============================================================================
template <typename Real, size_t N>
void write_vtk(std::vector<autonomous_particle<Real, N>> const& particles,
               Real const t0, std::string const& forward_path,
               std::string const& backward_path) {
  vtk::legacy_file_writer writer_forw{backward_path, vtk::POLYDATA},
      write_back{forward_path, vtk::POLYDATA};
  if (writer_forw.is_open()) {
    writer_forw.write_header();

    std::vector<std::vector<size_t>> lines;
    std::vector<vec<double, 3>>      points;
    lines.reserve(size(particles));
    for (auto const& p : particles) {
      points.push_back(vec{p.previous_position(0), p.previous_position(1),
                           p.previous_time()});
      points.push_back(vec{p.x1(0), p.x1(1), p.t1()});
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
    for (auto const& p : particles) {
      points.push_back(vec{p.x0(0), p.x0(1), t0});
      points.push_back(vec{p.x1(0), p.x1(1), p.t1()});
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
