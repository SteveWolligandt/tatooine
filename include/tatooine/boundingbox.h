#ifndef TATOOINE_BOUNDINGBOX_H
#define TATOOINE_BOUNDINGBOX_H

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
  pos_t min;
  pos_t max;
  //============================================================================
  constexpr boundingbox()                             = default;
  constexpr boundingbox(const boundingbox& other)     = default;
  constexpr boundingbox(boundingbox&& other) noexcept = default;
  constexpr boundingbox& operator=(const boundingbox& other) = default;
  constexpr boundingbox& operator=(boundingbox&& other) noexcept = default;
  ~boundingbox()                                                 = default;
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr boundingbox(vec<Real0, N>&& _min,
                        vec<Real1, N>&& _max) noexcept
      : min{std::move(_min)}, max{std::move(_max)} {}
  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  constexpr boundingbox(const vec<Real0, N>& _min, const vec<Real1, N>& _max)
      : min{_min}, max{_max} {}
  //----------------------------------------------------------------------------
  template <typename Tensor0, typename Tensor1, typename Real0, typename Real1>
  constexpr boundingbox(const base_tensor<Tensor0, Real0, N>& _min,
                        const base_tensor<Tensor1, Real1, N>& _max)
      : min{_min}, max{_max} {}
  //============================================================================
  constexpr void operator+=(const pos_t& point) {
    for (size_t i = 0; i < point.size(); ++i) {
      min(i) = std::min(min(i), point(i));
      max(i) = std::max(max(i), point(i));
    }
  }

  //----------------------------------------------------------------------------
  constexpr auto reset() {
    for (size_t i = 0; i < N; ++i) {
      min(i) = std::numeric_limits<Real>::max();
      max(i) = -std::numeric_limits<Real>::max();
    }
  }

  //----------------------------------------------------------------------------
  constexpr auto center() const { return (max + min) * Real(0.5); }

  //----------------------------------------------------------------------------
  constexpr auto is_inside(const pos_t& p) const {
    for (size_t i = 0; i < N; ++i) {
      if (p[i] < min[i] || max[i] < p[i]) { return false; }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename RandomEngine>
  auto random_point(RandomEngine& random_engine) const {
    pos_t p;
    for (size_t i = 0; i < N; ++i) {
      std::uniform_real_distribution<Real> distribution{min(i), max(i)};
      p(i) = distribution(random_engine);
    }
    return p;
  }
};
//==============================================================================
// deduction guides
//==============================================================================
#if has_cxx17_support()
template <typename Real0, typename Real1, size_t N>
boundingbox(const vec<Real0, N>&, const vec<Real1, N>&)
    ->boundingbox<promote_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, size_t N>
boundingbox(vec<Real0, N>&&, vec<Real1, N> &&)
    ->boundingbox<promote_t<Real0, Real1>, N>;
//------------------------------------------------------------------------------
template <typename Tensor0, typename Tensor1, typename Real0, typename Real1,
          size_t N>
boundingbox(base_tensor<Tensor0, Real0, N>&&, base_tensor<Tensor1, Real1, N> &&)
    ->boundingbox<promote_t<Real0, Real1>, N>;
#endif

//==============================================================================
// ostream output
//==============================================================================
template <typename Real, size_t N>
auto& operator<<(std::ostream& out, const boundingbox<Real, N>& bb) {
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
