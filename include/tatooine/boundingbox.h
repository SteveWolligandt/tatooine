#ifndef TATOOINE_BOUNDINGBOX_H
#define TATOOINE_BOUNDINGBOX_H
//==============================================================================
#include <limits>
#include <ostream>

#include "tensor.h"
#include "type_traits.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct boundingbox {
  //============================================================================
  using real_t = Real;
  using this_t = boundingbox<Real, N>;
  using pos_t  = vec<Real, N>;

  static constexpr auto num_dimensions() { return N; }
  //============================================================================
 private:
  pos_t m_min;
  pos_t m_max;
  //============================================================================
 public:
  constexpr boundingbox()                             = default;
  constexpr boundingbox(const boundingbox& other)     = default;
  constexpr boundingbox(boundingbox&& other) noexcept = default;
  constexpr auto operator=(const this_t& other) -> boundingbox& = default;
  constexpr auto operator=(this_t&& other) noexcept -> boundingbox& = default;
  ~boundingbox()                                                    = default;
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr boundingbox(vec<Real0, N>&& min, vec<Real1, N>&& max) noexcept
      : m_min{std::move(min)}, m_max{std::move(max)} {}
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr boundingbox(const vec<Real0, N>& min, const vec<Real1, N>& max)
      : m_min{min}, m_max{max} {}
  //----------------------------------------------------------------------------
  template <typename Tensor0, typename Tensor1, typename Real0, typename Real1>
  constexpr boundingbox(const base_tensor<Tensor0, Real0, N>& min,
                        const base_tensor<Tensor1, Real1, N>& max)
      : m_min{min}, m_max{max} {}
  //============================================================================
  const auto& min() const { return m_min; }
  auto&       min() { return m_min; }
  const auto& min(size_t i) const { return m_min(i); }
  auto&       min(size_t i) { return m_min(i); }
  //----------------------------------------------------------------------------
  const auto& max() const { return m_max; }
  auto&       max() { return m_max; }
  const auto& max(size_t i) const { return m_max(i); }
  auto&       max(size_t i) { return m_max(i); }
  //----------------------------------------------------------------------------
  constexpr void operator+=(const pos_t& point) {
    for (size_t i = 0; i < point.size(); ++i) {
      m_min(i) = std::min(m_min(i), point(i));
      m_max(i) = std::max(m_max(i), point(i));
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto reset() {
    for (size_t i = 0; i < N; ++i) {
      m_min(i) = std::numeric_limits<Real>::max();
      m_max(i) = -std::numeric_limits<Real>::max();
    }
  }
  //----------------------------------------------------------------------------
  constexpr auto center() const { return (m_max + m_min) * Real(0.5); }
  //----------------------------------------------------------------------------
  constexpr auto is_inside(const pos_t& p) const {
    for (size_t i = 0; i < N; ++i) {
      if (p(i) < m_min(i) || m_max(i) < p(i)) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto add_dimension(Real m_min, Real m_max) const {
    boundingbox<Real, N + 1> addeddim;
    for (size_t i = 0; i < N; ++i) {
      addeddim.m_min(i) = this->m_min(i);
      addeddim.m_max(i) = this->m_max(i);
    }
    addeddim.m_min(N) = m_min;
    addeddim.m_max(N) = m_max;
    return addeddim;
  }
  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  auto random_point(RandomEngine&& random_engine = RandomEngine{
                        std::random_device{}()}) const {
    pos_t p;
    for (size_t i = 0; i < N; ++i) {
      std::uniform_real_distribution<Real> distribution{m_min(i), m_max(i)};
      p(i) = distribution(random_engine);
    }
    return p;
  }
};
//==============================================================================
// deduction guides
//==============================================================================
template <typename Real0, typename Real1, size_t N>
boundingbox(const vec<Real0, N>&, const vec<Real1, N>&)
    -> boundingbox<promote_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N>
boundingbox(vec<Real0, N>&&, vec<Real1, N> &&)
    -> boundingbox<promote_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
boundingbox(base_tensor<Tensor0, Real0, N>&&, base_tensor<Tensor1, Real1, N> &&)
    -> boundingbox<promote_t<Real0, Real1>, N>;

//==============================================================================
// ostream output
//==============================================================================
template <typename Real, size_t N>
auto operator<<(std::ostream& out, const boundingbox<Real, N>& bb)
    -> std::ostream& {
  out << std::scientific;
  for (size_t i = 0; i < N; ++i) {
    out << "[ ";
    if (bb.m_min(i) >= 0) { out << ' '; }
    out << bb.m_min(i) << " .. ";
    if (bb.max(i) >= 0) { out << ' '; }
    out << bb.max(i) << " ]\n";
  }
  out << std::defaultfloat;
  return out;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
