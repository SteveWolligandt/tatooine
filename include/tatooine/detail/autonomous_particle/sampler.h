#ifndef TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SAMPLER_H
#define TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>
#include <tatooine/mat.h>
#include <tatooine/geometry/hyper_ellipse.h>
//==============================================================================
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct sampler {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using vec_t     = vec<Real, NumDimensions>;
  using pos_t     = vec_t;
  using mat_t     = mat<Real, NumDimensions, NumDimensions>;
  using ellipse_t = tatooine::geometry::hyper_ellipse<Real, NumDimensions>;

 private:
  //============================================================================
  // MEMBERS
  //============================================================================
  ellipse_t m_ellipse0, m_ellipse1;
  mat_t     m_nabla_phi, m_nabla_phi_inv;

 public:
  //============================================================================
  // CTORS
  //============================================================================
  sampler(sampler const&)     = default;
  sampler(sampler&&) noexcept = default;
  //============================================================================
  auto operator=(sampler const&) -> sampler& = default;
  auto operator=(sampler&&) noexcept -> sampler& = default;
  //============================================================================
  sampler()  = default;
  ~sampler() = default;
  //----------------------------------------------------------------------------
  sampler(ellipse_t const& e0, ellipse_t const& e1, mat_t const& nabla_phi)
      : m_ellipse0{e0},
        m_ellipse1{e1},
        m_nabla_phi{nabla_phi},
        m_nabla_phi_inv{*inv(nabla_phi)} {}
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto ellipse(forward_tag /*tag*/) const -> auto const& {
    return m_ellipse0;
  }
  auto ellipse(backward_tag /*tag*/) const -> auto const& {
    return m_ellipse1;
  }
  auto ellipse0() const -> auto const& { return m_ellipse0; }
  auto ellipse1() const -> auto const& { return m_ellipse1; }
  //============================================================================
  // METHODS
  //============================================================================
  auto sample_forward(pos_t const& x) const {
    return ellipse1().center() + nabla_phi() * (x - ellipse0().center());
  }
  auto operator()(pos_t const& x, forward_tag /*tag*/) const {
    return sample_forward(x);
  }
  auto sample(pos_t const& x, forward_tag /*tag*/) const {
    return sample_forward(x);
  }
  auto sample_backward(pos_t const& x) const {
    return ellipse0().center() + m_nabla_phi_inv * (x - ellipse1().center());
  }
  auto sample(pos_t const& x, backward_tag /*tag*/) const {
    return sample_backward(x);
  }
  auto operator()(pos_t const& x, backward_tag /*tag*/) const {
    return sample_backward(x);
  }
  auto is_inside0(pos_t const& x) const { return m_ellipse0.is_inside(x); }
  auto is_inside(pos_t const& x, forward_tag /*tag*/) const {
    return is_inside0(x);
  }
  auto is_inside1(pos_t const& x) const { return m_ellipse1.is_inside(x); }
  auto is_inside(pos_t const& x, backward_tag /*tag*/) const {
    return is_inside1(x);
  }
  auto center(forward_tag /*tag*/) const -> auto const& {
    return m_ellipse0.center();
  }
  auto center(backward_tag /*tag*/) const -> auto const& {
    return m_ellipse1.center();
  }
  auto opposite_center(forward_tag /*tag*/) const -> auto const& {
    return m_ellipse1.center();
  }
  auto opposite_center(backward_tag /*tag*/) const -> auto const& {
    return m_ellipse0.center();
  }
  auto distance_sqr(pos_t const& x, forward_tag tag) const {
    return tatooine::euclidean_length(nabla_phi() * (x - center(tag)));
  }
  auto distance_sqr(pos_t const& x, backward_tag tag) const {
    return tatooine::length(solve(nabla_phi(), (x - center(tag))));
  }
  auto distance(pos_t const& x, auto const tag) const {
    return gcem::sqrt(distance_sqr(x, tag));
  }
  auto nabla_phi() const -> auto const& { return m_nabla_phi; }
  auto nabla_phi_inv() const -> auto const& { return m_nabla_phi_inv; }
};
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
