#ifndef TATOOINE_BOUNDINGBOX_H
#define TATOOINE_BOUNDINGBOX_H

#include <limits>
#include <ostream>
#include "type_traits.h"
#include "tensor.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <size_t n, typename real_t>
struct boundingbox {
  //============================================================================
  using this_t = boundingbox<n, real_t>;
  using pos_t  = tensor<real_t, n>;

  //============================================================================
  pos_t min;
  pos_t max;

  //============================================================================
  constexpr boundingbox()                                        = default;
  constexpr boundingbox(const boundingbox& other)                = default;
  constexpr boundingbox(boundingbox&& other)           noexcept  = default;
  constexpr boundingbox& operator=(const boundingbox& other)     = default;
  constexpr boundingbox& operator=(boundingbox&& other) noexcept = default;
  ~boundingbox()                                                 = default;

  //----------------------------------------------------------------------------
  template <typename real_t0, typename real_t1>
  constexpr boundingbox(tensor<real_t0, n>&& _min, tensor<real_t1, n>&& _max) noexcept
      : min{std::move(_min)}, max{std::move(_max)} {}

  //----------------------------------------------------------------------------
  template <typename real_t0, typename real_t1>
  constexpr boundingbox(const tensor<real_t0, n>& _min,
                        const tensor<real_t1, n>& _max) noexcept
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
    for (size_t i = 0; i < n; ++i) {
      min(i) = std::numeric_limits<real_t>::max();
      max(i) = -std::numeric_limits<real_t>::max();
    }
  }

  //----------------------------------------------------------------------------
  constexpr auto center() const { return (max + min) * real_t(0.5); }

  //----------------------------------------------------------------------------
  constexpr auto is_inside(const pos_t& p) const {
    for (size_t i = 0; i < n; ++i) {
      if (p[i] < min[i] || max[i] < p[i]) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  template <typename random_engine_t>
  auto random_point(random_engine_t&& random_engine) {
    pos_t p;
    for (size_t i = 0; i < n; ++i) {
      std::uniform_real_distribution distribution{min(i), max(i)};
      p(i) = distribution(random_engine);
    }
    return p;
  }
};

//==============================================================================
// deduction guides
//==============================================================================
template <typename real_t0, typename real_t1, size_t n>
boundingbox(const tensor<real_t0, n>&, const tensor<real_t1, n>&)
    ->boundingbox<n, promote_t<real_t0, real_t1>>;

//------------------------------------------------------------------------------
template <typename real_t0, typename real_t1, size_t n>
boundingbox(tensor<real_t0, n>&&, tensor<real_t1, n> &&)
    ->boundingbox<n, promote_t<real_t0, real_t1>>;

//==============================================================================
// ostream output
//==============================================================================
template <typename real_t, size_t n>
auto& operator<<(std::ostream& out, const boundingbox<n, real_t>& bb) {
  out << std::scientific;
  for (size_t i = 0; i < n; ++i) {
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
