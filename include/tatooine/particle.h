#ifndef TATOOINE_PARTICLE
#define TATOOINE_PARTICLE
//==============================================================================
#include <tatooine/pointset.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct particle {
  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using this_t = particle;
  using real_t = Real;
  using vec_t  = vec<real_t, NumDimensions>;
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
  auto x0(std::size_t const i) const { return m_x0(i); }
  //----------------------------------------------------------------------------
  auto x1() -> auto& { return m_x1; }
  auto x1() const -> auto const& { return m_x1; }
  auto x1(std::size_t const i) const { return m_x1(i); }
  //----------------------------------------------------------------------------
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
template <typename Real, std::size_t NumDimensions>
particle<Real, NumDimensions>::particle(particle const& other)
    : m_x0{other.m_x0}, m_x1{other.m_x1}, m_t1{other.m_t1} {}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t NumDimensions>
particle<Real, NumDimensions>::particle(particle&& other) noexcept
    : m_x0{std::move(other.m_x0)},
      m_x1{std::move(other.m_x1)},
      m_t1{other.m_t1} {}

//----------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
auto particle<Real, NumDimensions>::operator=(particle const& other)
    -> particle& {
  if (&other == this) {
    return *this;
  };
  m_x0 = other.m_x0;
  m_x1 = other.m_x1;
  m_t1 = other.m_t1;
  return *this;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t NumDimensions>
auto particle<Real, NumDimensions>::operator=(particle&& other) noexcept
    -> particle& {
  m_x0 = std::move(other.m_x0);
  m_x1 = std::move(other.m_x1);
  m_t1 = other.m_t1;
  return *this;
}
//----------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
particle<Real, NumDimensions>::particle(pos_t const& x0, real_t const t0)
    : m_x0{x0}, m_x1{x0}, m_t1{t0} {}
//----------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
particle<Real, NumDimensions>::particle(pos_t const& x0, pos_t const& x1,
                                        real_t const t1)
    : m_x0{x0}, m_x1{x1}, m_t1{t1} {}
//==============================================================================
template <std::size_t NumDimensions>
using Particle  = particle<real_t, NumDimensions>;
using particle2 = Particle<2>;
using particle3 = Particle<3>;
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto x0_to_pointset(
    std::vector<particle<Real, NumDimensions>> const& particles) {
  auto ps = pointset<Real, NumDimensions>{};
  for (auto const& p : particles) {
    ps.insert_vertex(p.x0());
  }
  return ps;
}
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
auto x1_to_pointset(
    std::vector<particle<Real, NumDimensions>> const& particles) {
  auto ps = pointset<Real, NumDimensions>{};
  for (auto const& p : particles) {
    ps.insert_vertex(p.x1());
  }
  return ps;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
