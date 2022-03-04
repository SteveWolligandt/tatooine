#ifndef TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SAMPLER_H
#define TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/mat.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct sampler {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using vec_type     = vec<Real, NumDimensions>;
  using pos_type     = vec_type;
  using mat_type     = mat<Real, NumDimensions, NumDimensions>;
  using ellipse_type = tatooine::geometry::hyper_ellipse<Real, NumDimensions>;

 private:
  //============================================================================
  // MEMBERS
  //============================================================================
  ellipse_type m_ellipse0, m_ellipse1;
  mat_type     m_nabla_phi, m_nabla_phi_inv;

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
  sampler(ellipse_type const& e0, ellipse_type const& e1,
          mat_type const& nabla_phi)
      : m_ellipse0{e0},
        m_ellipse1{e1},
        m_nabla_phi{nabla_phi},
        m_nabla_phi_inv{*inv(nabla_phi)} {}
  //============================================================================
  // GETTERS / SETTERS
  //============================================================================
  auto ellipse(forward_tag /*tag*/) const -> auto const& { return m_ellipse0; }
  auto ellipse(backward_tag /*tag*/) const -> auto const& { return m_ellipse1; }
  //----------------------------------------------------------------------------
  auto S(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(tag).S();
  }
  //----------------------------------------------------------------------------
  auto nabla_phi(forward_tag const /*tag*/) const -> auto const& {
    return m_nabla_phi;
  }
  auto nabla_phi(backward_tag const /*tag*/) const -> auto const& {
    return m_nabla_phi_inv;
  }
  //============================================================================
  // METHODS
  //============================================================================
  auto local_pos(pos_type const&                    x,
                 forward_or_backward_tag auto const tag) const {
    return nabla_phi(tag) * (x - center(tag));
  }
  //----------------------------------------------------------------------------
  auto sample(pos_type const& x, forward_or_backward_tag auto const tag) const {
    return center(opposite(tag)) + local_pos(x, tag);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const&                    x,
                  forward_or_backward_tag auto const tag) const {
    return sample(x, tag);
  }
  //----------------------------------------------------------------------------
  auto is_inside(pos_type const& x, forward_or_backward_tag auto const tag) const {
    return ellipse(tag).is_inside(x);
  }
  //----------------------------------------------------------------------------
  auto center(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(tag).center();
  }
  //----------------------------------------------------------------------------
  auto distance_sqr(pos_type const&                    x,
                    forward_or_backward_tag auto const tag) const {
    return tatooine::euclidean_length(local_pos(tag));
  }
  //----------------------------------------------------------------------------
  auto distance(pos_type const& x, auto const tag) const {
    return gcem::sqrt(distance_sqr(x, tag));
  }
};
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
