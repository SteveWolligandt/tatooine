#ifndef TATOOINE_NUMERICAL_FLOWMAP_H
#define TATOOINE_NUMERICAL_FLOWMAP_H
//==============================================================================
#include <tatooine/exceptions.h>
#include <tatooine/field.h>
#include <tatooine/interpolation.h>
#include <tatooine/is_cacheable.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
//#include <tatooine/ode/boost/rungekuttafehlberg78.h>
#include <tatooine/cache.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
struct numerical_flowmap {
  using this_t      = numerical_flowmap<V, ODESolver, InterpolationKernel>;
  using raw_field_t = std::remove_pointer_t<std::decay_t<V>>;
  using real_t      = typename raw_field_t::real_t;
  static constexpr auto num_dimensions() {
    return raw_field_t::num_dimensions();
  }
  using vec_t            = vec<real_t, num_dimensions()>;
  using pos_t            = vec_t;
  using integral_curve_t = line<real_t, num_dimensions()>;
  using cache_t = tatooine::cache<std::pair<real_t, pos_t>, integral_curve_t>;
  using ode_solver_t = ODESolver<real_t, num_dimensions()>;
  static constexpr auto holds_field_pointer = std::is_pointer_v<V>;
  //============================================================================
  // members
  //============================================================================
 private:
  V               m_v;
  ode_solver_t    m_ode_solver;
  mutable cache_t m_cache;
  mutable std::map<std::pair<pos_t, real_t>, std::pair<bool, bool>>
       m_on_domain_border;
  bool m_use_caching = true;
  //============================================================================
  // ctors
  //============================================================================
 public:
  numerical_flowmap(numerical_flowmap const& other)
      : m_v{other.m_v},
        m_ode_solver{other.m_ode_solver},
        m_use_caching{other.m_use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  numerical_flowmap(numerical_flowmap&& other) noexcept
      : m_v{std::move(other.m_v)},
        m_ode_solver{std::move(other.m_ode_solver)},
        m_use_caching{other.m_use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(numerical_flowmap const& other) -> numerical_flowmap& {
    m_v           = other.m_v;
    m_ode_solver  = other.m_ode_solver;
    m_use_caching = other.m_use_caching;
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(numerical_flowmap&& other) noexcept -> numerical_flowmap& {
    m_v           = std::move(other.m_v);
    m_ode_solver  = std::move(other.m_ode_solver);
    m_use_caching = std::move(other.m_use_caching);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::convertible_to<V> W, arithmetic WReal, std::size_t N,
            typename V_ = V>
  requires(!holds_field_pointer) explicit constexpr numerical_flowmap(
      vectorfield<W, WReal, N> const& w, bool const use_caching = true)
      : m_v{w.as_derived()}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::convertible_to<V> W, arithmetic WReal, std::size_t N,
            typename V_ = V>
  requires holds_field_pointer explicit constexpr numerical_flowmap(
      vectorfield<W, WReal, N> const* w, bool const use_caching = true)
      : m_v{&w->as_derived()}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic WReal, std::size_t N, typename V_ = V>
  requires holds_field_pointer explicit constexpr numerical_flowmap(
      polymorphic::vectorfield<WReal, N> const* w,
      bool const                                use_caching = true)
      : m_v{w}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic WReal, std::size_t N, typename V_ = V>
  requires holds_field_pointer explicit constexpr numerical_flowmap(
      polymorphic::vectorfield<WReal, N>* w, bool const use_caching = true)
      : m_v{w}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename V_ = V>
  requires holds_field_pointer explicit constexpr numerical_flowmap(
      bool const use_caching = true)
      : m_v{nullptr}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename W, typename WReal, std::size_t N, typename V_ = V>
  requires(!holds_field_pointer) constexpr numerical_flowmap(
      vectorfield<W, WReal, N> const& w, ode_solver_t const& ode_solver,
      bool const use_caching = true)
      : m_v{w.as_derived()},
        m_ode_solver{ode_solver},
        m_use_caching{use_caching} {}
  //============================================================================
  template <std::size_t K>
  [[nodiscard]] constexpr auto evaluate(mat<real_t, num_dimensions(), K> xs,
                                        real_t const t0, real_t tau) const {
    for (std::size_t i = 0; i < K; ++i) {
      xs.col(i) = evaluate(pos_t{xs.col(i)}, t0, tau);
    }
    return xs;
  }
  //============================================================================
  [[nodiscard]] constexpr auto evaluate(pos_t x, real_t const t0,
                                        real_t tau) const -> pos_t {
    if (tau == 0) {
      return x;
    }
    if (!m_use_caching) {
      auto callback = [t0, &x, tau](const auto& y, auto const t) {
        if (t0 + tau == t) {
          x = y;
        }
      };
      m_ode_solver.solve(vectorfield(), x, t0, tau, callback);
      return x;
    }
    // use caching
    constexpr real_t security_eps   = 1e-7;
    auto const&      integral_curve = cached_curve(x, t0, tau);
    auto             t              = t0 + tau;
    if (tau < 0 &&
        t < integral_curve
                .parameterization()[integral_curve.vertices().front()]) {
      if (t + security_eps <
          integral_curve
              .parameterization()[integral_curve.vertices().front()]) {
        t = integral_curve
                .parameterization()[integral_curve.vertices().front()];
      } else {
        throw out_of_domain_error{};
      }
    }
    if (tau > 0 &&
        t > integral_curve
                .parameterization()[integral_curve.vertices().back()]) {
      if (t - security_eps <
          integral_curve.parameterization()[integral_curve.vertices().back()]) {
        t = integral_curve.parameterization()[integral_curve.vertices().back()];
      } else {
        throw out_of_domain_error{};
      }
    }
    return integral_curve.template sampler<InterpolationKernel>()(t);
  }
  //----------------------------------------------------------------------------
  template <typename Mat, std::size_t K>
  [[nodiscard]] constexpr auto operator()(
      base_tensor<Mat, real_t, num_dimensions(), K> const& y0s, real_t const t0,
      real_t const tau) const {
    return evaluate(mat{y0s}, t0, tau);
  }
  //----------------------------------------------------------------------------
  template <typename Vec>
  [[nodiscard]] constexpr auto operator()(
      base_tensor<Vec, real_t, num_dimensions()> const& y0, real_t const t0,
      real_t const tau) const -> pos_t {
    return evaluate(vec{y0}, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto integral_curve(pos_t const& y0, real_t const t0,
                      real_t const tau) const {
    integral_curve_t c;
    auto const       v          = c.push_back(y0);
    c.parameterization()[v]     = t0;
    auto const full_integration = continue_integration(c, tau);
    return std::pair{std::move(c), full_integration};
  }
  //----------------------------------------------------------------------------
  auto integral_curve(pos_t const& y0, real_t const t0, real_t const btau,
                      real_t const ftau) const {
    integral_curve_t c;
    auto const       v      = c.push_back(y0);
    c.parameterization()[v] = t0;

    bool const backward_full = [this, &c, btau] {
      if (btau < 0) {
        return continue_integration(c, btau);
      }
      return true;
    }();
    bool const forward_full = [this, &c, ftau] {
      if (ftau > 0) {
        return continue_integration(c, ftau);
      }
      return true;
    }();
    return std::tuple{std::move(c), backward_full, forward_full};
  }
  //----------------------------------------------------------------------------
  /// Continues integration if integral_curve.
  /// If tau > 0 it takes front of integral_curve as start position and time.
  /// If tau < 0 it takes back of integral_curve as start position and time.
  /// \return true if could integrate all tau, false if hit domain border or
  /// something else went wrong.
  auto continue_integration(integral_curve_t& integral_curve,
                            real_t const      tau) const -> bool {
    auto&       tangents         = integral_curve.tangents();
    auto&       parameterization = integral_curve.parameterization();
    auto const& y0               = [&integral_curve, tau] {
      if (tau > 0) {
        return integral_curve.back_vertex();
      }
      return integral_curve.front_vertex();
    }();
    auto const& t0 = [&integral_curve, &parameterization, tau] {
      if (tau > 0) {
        return parameterization[integral_curve.vertices().back()];
      }
      return parameterization[integral_curve.vertices().front()];
    }();
    auto callback = [&integral_curve, &parameterization, &tangents, tau](
                        const auto& y, auto const t, const auto& dy) {
      if (integral_curve.num_vertices() > 0 &&
          std::abs(parameterization[integral_curve.vertices().back()] - t) <
              1e-13) {
        return;
      }
      auto const v = [&] {
        if (tau < 0) {
          return integral_curve.push_front(y);
        }
        return integral_curve.push_back(y);
      }();
      parameterization[v] = t;
      tangents[v]         = dy;
    };

    m_ode_solver.solve(vectorfield(), y0, t0, tau, callback);
    if (!integral_curve.empty()) {
      if ((tau > 0 &&
           parameterization[integral_curve.vertices().back()] < t0 + tau) ||
          (tau < 0 &&
           parameterization[integral_curve.vertices().front()] < t0 + tau)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0) const -> auto const& {
    return *m_cache.emplace({t0, y0}).first->second;
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0, real_t const tau) const
      -> auto const& {
    return cached_curve(y0, t0, tau < 0 ? tau : 0, tau > 0 ? tau : 0);
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0, real_t btau,
                    real_t ftau) const -> auto const& {
    auto [curve_it, new_integral_curve] = m_cache.emplace({t0, y0});
    auto& curve                         = curve_it->second;

    auto& [backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    if (new_integral_curve || curve.empty()) {
      // integral_curve not yet integrated
      auto [fresh_curve, fullback, fullforw] =
          integral_curve(y0, t0, btau, ftau);
      curve              = std::move(fresh_curve);
      backward_on_border = !fullback;
      forward_on_border  = !fullforw;
    } else {
      if (btau > ftau) {
        std::swap(btau, ftau);
      }
      if (auto const tf = curve.parameterization()[curve.vertices().front()];
          btau < 0 && tf > t0 + btau && !backward_on_border) {
        // continue integration in backward time
        bool const full    = continue_integration(curve, t0 + btau - tf);
        backward_on_border = !full;
      }
      if (auto const tb = curve.parameterization()[curve.vertices().back()];
          ftau > 0 && tb < t0 + ftau && !forward_on_border) {
        // continue integration in forward time
        bool const full   = continue_integration(curve, t0 + ftau - tb);
        forward_on_border = !full;
      }
    }
    return curve;
  }
  //============================================================================
  auto vectorfield() const -> auto const& {
    if constexpr (holds_field_pointer) {
      return *m_v;
    } else {
      return m_v;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto vectorfield() -> auto& {
    if constexpr (holds_field_pointer) {
      return *m_v;
    } else {
      return m_v;
    }
  }
  //----------------------------------------------------------------------------
  template <typename = void>
  requires holds_field_pointer auto set_vectorfield(
      polymorphic::vectorfield<real_t, num_dimensions()>* w) {
    m_v = w;
  }
  //----------------------------------------------------------------------------
  auto ode_solver() const -> auto const& { return *m_ode_solver; }
  auto ode_solver() -> auto& { return *m_ode_solver; }
  //----------------------------------------------------------------------------
  auto use_caching(bool const b = true) -> void { m_use_caching = b; }
  auto is_using_caching() const { return m_use_caching; }
  auto is_using_caching() -> auto& { return m_use_caching; }
  //----------------------------------------------------------------------------
  auto invalidate_cache() const {
    m_cache.clear();
    m_on_domain_border.clear();
  }
};

//==============================================================================
template <typename V, typename Real, std::size_t N>
numerical_flowmap(vectorfield<V, Real, N> const&)
    -> numerical_flowmap<V const&, ode::vclibs::rungekutta43,
                         interpolation::cubic>;
//-> numerical_flowmap<V, ode::boost::rungekuttafehlberg78,
// interpolation::cubic>;
//------------------------------------------------------------------------------
template <typename V, typename Real, std::size_t N,
          template <typename, std::size_t> typename ODESolver>
numerical_flowmap(vectorfield<V, Real, N> const&, ODESolver<Real, N> const&)
    -> numerical_flowmap<V const&, ODESolver, interpolation::cubic>;
//==============================================================================
template <
    // template <typename, std::size_t> typename ODESolver =
    // ode::boost::rungekuttafehlberg78,
    template <typename, std::size_t>
    typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t N>
auto flowmap(vectorfield<V, Real, N> const& v, tag::numerical_t /*tag*/) {
  return numerical_flowmap<V const&, ODESolver, InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    // template <typename, std::size_t> typename ODESolver =
    // ode::boost::rungekuttafehlberg78,
    template <typename, std::size_t>
    typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t N>
auto flowmap(vectorfield<V, Real, N> const& v) {
  return numerical_flowmap<V const&, ODESolver, InterpolationKernel>(v);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    // template <typename, std::size_t> typename ODESolver =
    // ode::boost::rungekuttafehlberg78,
    template <typename, std::size_t>
    typename ODESolver = ode::vclibs::rungekutta43,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename Real, std::size_t N>
auto flowmap(polymorphic::vectorfield<Real, N> const& v) {
  return numerical_flowmap<polymorphic::vectorfield<Real, N> const*, ODESolver,
                           InterpolationKernel>(&v);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include "flowmap_gradient_central_differences.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel,
          arithmetic EpsReal = typename V::real_t>
auto diff(numerical_flowmap<V, ODESolver, InterpolationKernel> const& flowmap,
          tag::central_t /*tag*/, EpsReal epsilon = 1e-7) {
  return flowmap_gradient_central_differences<
      numerical_flowmap<V, ODESolver, InterpolationKernel>>{flowmap, epsilon};
}
//------------------------------------------------------------------------------
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel,
          std::floating_point EpsReal>
auto diff(numerical_flowmap<V, ODESolver, InterpolationKernel> const& flowmap,
          tag::central_t /*tag*/, vec<EpsReal, V::num_dimensions()> epsilon) {
  return flowmap_gradient_central_differences<
      numerical_flowmap<V, ODESolver, InterpolationKernel>>{flowmap, epsilon};
}
//==============================================================================
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel,
          arithmetic EpsReal = typename std::decay_t<V>::real_t>
auto diff(numerical_flowmap<V, ODESolver, InterpolationKernel> const& flowmap,
          EpsReal epsilon = 1e-7) {
  return diff(flowmap, tag::central, epsilon);
}
//------------------------------------------------------------------------------
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel,
          std::floating_point EpsReal>
auto diff(numerical_flowmap<V, ODESolver, InterpolationKernel> const& flowmap,
          vec<EpsReal, V::num_dimensions()>                           epsilon) {
  return diff(flowmap, tag::central, epsilon);
}
//==============================================================================
// typedefs
//==============================================================================
template <arithmetic Real, std::size_t N,
          template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
using numerical_flowmap_field_pointer =
    numerical_flowmap<polymorphic::vectorfield<Real, N>*, ODESolver,
                      InterpolationKernel>;
//==============================================================================
// type traits
//==============================================================================
template <typename T>
struct is_numerical_flowmap : std::false_type {};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
struct is_numerical_flowmap<
    numerical_flowmap<V, ODESolver, InterpolationKernel>> : std::true_type {};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename T>
static constexpr auto is_numerical_flowmap_v = is_numerical_flowmap<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
