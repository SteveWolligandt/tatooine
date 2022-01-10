#ifndef TATOOINE_PARTICLE
#define TATOOINE_PARTICLE
//==============================================================================
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct particle {
  static constexpr auto num_dimensions() { return N; }
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using this_t = particle;
  using real_t = Real;
  using vec_t  = vec<real_t, N>;
  using pos_t  = vec_t;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  pos_t  m_x0, m_x1;
  real_t m_t1;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  particle(particle const& other);
  particle(particle&& other) noexcept;
  //----------------------------------------------------------------------------
  auto operator=(particle const& other) -> particle&;
  auto operator=(particle&& other) noexcept -> particle&;
  //----------------------------------------------------------------------------
  ~particle() = default;
  //----------------------------------------------------------------------------
  particle() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  particle(pos_t const& x0, real_t t0);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  particle(pos_t const& x0, pos_t const& x1, real_t t1);
  //----------------------------------------------------------------------------
  // getters / setters
  //----------------------------------------------------------------------------
  auto x0() -> auto& { return m_x0; }
  auto x0() const -> auto const& { return m_x0; }
  auto x0(size_t i) const { return m_x0(i); }
  auto x1() -> auto& { return m_x1; }
  auto x1() const -> auto const& { return m_x1; }
  auto x1(size_t i) const { return m_x1(i); }
  auto t1() -> auto& { return m_t1; }
  auto t1() const { return m_t1; }

  //----------------------------------------------------------------------------
  template <typename Flowmap>
  auto advect(Flowmap&& phi, real_t const tau) {
    m_x1 = phi(m_x1, m_t1, tau);
    m_t1 += tau;
  }
};
//==============================================================================
template <typename Real, size_t N>
particle<Real, N>::particle(particle const& other)
    : m_x0{other.m_x0}, m_x1{other.m_x1}, m_t1{other.m_t1} {}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
particle<Real, N>::particle(particle&& other) noexcept
    : m_x0{std::move(other.m_x0)},
      m_x1{std::move(other.m_x1)},
      m_t1{other.m_t1} {}

//----------------------------------------------------------------------------
template <typename Real, size_t N>
auto particle<Real, N>::operator=(particle const& other) -> particle& {
  if (&other == this) {
    return *this;
  };
  m_x0 = other.m_x0;
  m_x1 = other.m_x1;
  m_t1 = other.m_t1;
  return *this;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
auto particle<Real, N>::operator=(particle&& other) noexcept -> particle& {
  m_x0 = std::move(other.m_x0);
  m_x1 = std::move(other.m_x1);
  m_t1 = other.m_t1;
  return *this;
}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
particle<Real, N>::particle(pos_t const& x0, real_t const t0)
    : m_x0{x0}, m_x1{x0}, m_t1{t0} {}
//----------------------------------------------------------------------------
template <typename Real, size_t N>
particle<Real, N>::particle(pos_t const& x0, pos_t const& x1, real_t const t1)
    : m_x0{x0}, m_x1{x1}, m_t1{t1} {}
//==============================================================================
template <size_t N>
using Particle  = particle<real_t, N>;
using particle2 = Particle<2>;
using particle3 = Particle<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
