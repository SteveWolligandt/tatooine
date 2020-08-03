#ifndef TATOOINE_BOUNDINGBOX_H
#define TATOOINE_BOUNDINGBOX_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ray_intersectable.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>

#include <limits>
#include <ostream>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number Real, size_t N>
struct boundingbox : ray_intersectable<Real, N> {
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
  //============================================================================
  // ray_intersectable overrides
  //============================================================================
  std::optional<intersection<Real, N>> check_intersection(
      const ray<Real, N>& r, const Real = 0) const override {
    enum Quadrant { right, left, middle };
    vec<Real, N>            coord;
    bool                    inside = true;
    std::array<Quadrant, N> quadrant;
    size_t                  which_plane;
    std::array<Real, N>     max_t;
    std::array<Real, N>     candidate_plane;

    // Find candidate planes; this loop can be avoided if rays cast all from the
    // eye(assume perpsective view)
    for (size_t i = 0; i < N; i++)
      if (r.origin(i) < min(i)) {
        quadrant[i]        = left;
        candidate_plane[i] = min(i);
        inside             = false;
      } else if (r.origin(i) > max(i)) {
        quadrant[i]        = right;
        candidate_plane[i] = max(i);
        inside             = false;
      } else {
        quadrant[i] = middle;
      }

    // Ray origin inside bounding box
    if (inside) {
      return intersection<Real, N>{this,
                                   r,
                                   Real(0),
                                   r.origin(),
                                   vec<Real, N>::zeros(),
                                   vec<Real, N - 1>::zeros()};
    }

    // Calculate T distances to candidate planes
    for (size_t i = 0; i < N; i++)
      if (quadrant[i] != middle && r.direction(i) != 0) {
        max_t[i] = (candidate_plane[i] - r.origin(i)) / r.direction(i);
      } else {
        max_t[i] = -1;
      }

    // Get largest of the max_t's for final choice of intersection
    which_plane = 0;
    for (size_t i = 1; i < N; i++)
      if (max_t[which_plane] < max_t[i]) {
        which_plane = i;
      }

    // Check final candidate actually inside box
    if (max_t[which_plane] < 0) {
      return {};
    }
    for (size_t i = 0; i < N; i++)
      if (which_plane != i) {
        coord(i) = r.origin(i) + max_t[which_plane] * r.direction(i);
        if (coord(i) < min(i) || coord(i) > max(i)) {
          return {};
        }
      } else {
        coord(i) = candidate_plane[i];
      }
    return intersection<Real, N>{this,
                                 r,
                                 max_t[which_plane],
                                 coord,
                                 vec<Real, N>::zeros(),
                                 vec<Real, N - 1>::zeros()};
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
    if (bb.min(i) >= 0) { out << ' '; }
    out << bb.min(i) << " .. ";
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
