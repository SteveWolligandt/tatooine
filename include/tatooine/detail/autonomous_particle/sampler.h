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
  mat_type     m_nabla_phi0, m_nabla_phi1;

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
        m_nabla_phi0{nabla_phi},
        m_nabla_phi1{*inv(nabla_phi)} {}
  //============================================================================
  /// \{
  auto ellipse(forward_tag const /*tag*/) const -> auto const& {
    return m_ellipse0;
  }
  //----------------------------------------------------------------------------
  auto ellipse(backward_tag const /*tag*/) const -> auto const& {
    return m_ellipse1;
  }
  /// \}
  //----------------------------------------------------------------------------
  /// \{
  auto nabla_phi(forward_tag const /*tag*/) const -> auto const& {
    return m_nabla_phi0;
  }
  //----------------------------------------------------------------------------
  auto nabla_phi(backward_tag const /*tag*/) const -> auto const& {
    return m_nabla_phi1;
  }
  /// \}
  //============================================================================
  auto local_pos(pos_type const&                    q,
                 forward_or_backward_tag auto const tag) const {
    return nabla_phi(tag) * (q - x0(tag));
  }
  //----------------------------------------------------------------------------
  auto sample(pos_type const& q, forward_or_backward_tag auto const tag) const {
    return phi(tag) + local_pos(q, tag);
  }
  //----------------------------------------------------------------------------
  auto operator()(pos_type const&                    q,
                  forward_or_backward_tag auto const tag) const {
    return sample(q, tag);
  }
  //----------------------------------------------------------------------------
  auto is_inside(pos_type const&                    q,
                 forward_or_backward_tag auto const tag) const {
    return ellipse(tag).is_inside(q);
  }
  //----------------------------------------------------------------------------
  auto x0(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(tag).center();
  }
  //----------------------------------------------------------------------------
  auto phi(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(opposite(tag)).center();
  }
  //----------------------------------------------------------------------------
  auto distance_sqr(pos_type const&                    q,
                    forward_or_backward_tag auto const tag) const {
    return tatooine::euclidean_length(local_pos(tag));
  }
  //----------------------------------------------------------------------------
  auto distance(pos_type const& q, auto const tag) const {
    return gcem::sqrt(distance_sqr(q, tag));
  }
  //----------------------------------------------------------------------------
  auto S(forward_or_backward_tag auto const tag) const -> auto const& {
    return ellipse(tag).S();
  }
};
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
