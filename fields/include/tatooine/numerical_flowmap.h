#ifndef TATOOINE_NUMERICAL_FLOWMAP_H
#define TATOOINE_NUMERICAL_FLOWMAP_H
//==============================================================================
#include <tatooine/exceptions.h>
#include <tatooine/field.h>
#include <tatooine/interpolation.h>
#include <tatooine/is_cacheable.h>
#include <tatooine/ode/boost/rungekuttafehlberg78.h>
#include <tatooine/cache.h>
#include <tatooine/tags.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
struct numerical_flowmap {
  using this_type      = numerical_flowmap<V, ODESolver, InterpolationKernel>;
  using raw_field_type = std::remove_pointer_t<std::decay_t<V>>;
  using real_type      = typename raw_field_type::real_type;
  static constexpr auto num_dimensions() {
    return raw_field_type::num_dimensions();
  }
  using vec_type            = vec<real_type, num_dimensions()>;
  using pos_type            = vec_type;
  using integral_curve_type = line<real_type, num_dimensions()>;
  using cache_type =
      tatooine::cache<std::pair<real_type, pos_type>, integral_curve_type>;
  using ode_solver_type = ODESolver<real_type, num_dimensions()>;
  using domain_border_flags_type =
      std::map<std::pair<pos_type, real_type>, std::pair<bool, bool>>;
  static constexpr auto holds_field_pointer = std::is_pointer_v<V>;
  static constexpr auto default_use_caching = false;
  //============================================================================
  // members
  //============================================================================
 private:
  V                                m_v;
  ode_solver_type                  m_ode_solver;
  mutable cache_type               m_cache;
  mutable domain_border_flags_type m_on_domain_border;
  bool                             m_use_caching = default_use_caching;
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
    if (&other == this) {
      return *this;
    }
    m_v           = other.m_v;
    m_ode_solver  = other.m_ode_solver;
    m_use_caching = other.m_use_caching;
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(numerical_flowmap&& other) noexcept -> numerical_flowmap& {
    m_v           = std::move(other.m_v);
    m_ode_solver  = std::move(other.m_ode_solver);
    m_use_caching = other.m_use_caching;
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~numerical_flowmap() = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::convertible_to<V> W, arithmetic WReal,
            std::size_t NumDimensions, typename V_ = V>
  requires(!holds_field_pointer)
  explicit constexpr numerical_flowmap(
      vectorfield<W, WReal, NumDimensions> const& w,
      bool const use_caching = default_use_caching)
      : m_v{w.as_derived()}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::convertible_to<V> W, arithmetic WReal,
            std::size_t NumDimensions, typename V_ = V>
  requires holds_field_pointer
  explicit constexpr numerical_flowmap(
      vectorfield<W, WReal, NumDimensions> const* w,
      bool const use_caching = default_use_caching)
      : m_v{&w->as_derived()}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic WReal, std::size_t NumDimensions, typename V_ = V>
  requires holds_field_pointer
  explicit constexpr numerical_flowmap(
      polymorphic::vectorfield<WReal, NumDimensions> const* w,
      bool const use_caching = default_use_caching)
      : m_v{w}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic WReal, std::size_t NumDimensions, typename V_ = V>
  requires holds_field_pointer
  explicit constexpr numerical_flowmap(
      polymorphic::vectorfield<WReal, NumDimensions>* w,
      bool const use_caching = default_use_caching)
      : m_v{w}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename V_ = V>
  requires holds_field_pointer
  explicit constexpr numerical_flowmap(
      bool const use_caching = default_use_caching)
      : m_v{nullptr}, m_use_caching{use_caching} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename W, typename WReal, std::size_t NumDimensions,
            typename V_ = V>
  requires(!holds_field_pointer)
  constexpr numerical_flowmap(
      vectorfield<W, WReal, NumDimensions> const& w,
      ode_solver_type const&                      ode_solver,
      bool const use_caching = default_use_caching)
      : m_v{w.as_derived()},
        m_ode_solver{ode_solver},
        m_use_caching{use_caching} {}
  //============================================================================
  [[nodiscard]] constexpr auto evaluate(
      fixed_num_rows_mat<num_dimensions()> auto const& x0s, real_type const t0,
      real_type tau) const {
    auto xes = mat<real_type, x0s.dimension(0), x0s.dimension(1)>{x0s};
    for (std::size_t i = 0; i < xes.dimension(1); ++i) {
      xes.col(i) = evaluate(pos_type{xes.col(i)}, t0, tau);
    }
    return xes;
  }
  //============================================================================
  [[nodiscard]] constexpr auto evaluate(
      fixed_size_vec<num_dimensions()> auto const& x0, real_type const t0,
      real_type const tau) const -> pos_type {
    if (tau == 0) {
      return pos_type{x0};
    }
    auto x1 = pos_type::fill(0.0 / 0.0);
    auto const t_end = t0 + tau;
    if (!m_use_caching) {
      auto callback = [t_end, &x1](const auto& y, auto const t) {
        if (t_end == t) {
          x1 = y;
        }
      };
      m_ode_solver.solve(vectorfield(), x0, t0, tau, callback);
      return x1;
    }
    // use caching
    constexpr real_type security_eps   = 1e-7;
    auto const&         integral_curve = cached_curve(x0, t0, tau);
    auto                t              = t0 + tau;
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
  [[nodiscard]] constexpr auto operator()(
      fixed_num_rows_mat<num_dimensions()> auto const& x0s, real_type const t0,
      real_type const tau) const {
    return evaluate(x0s, t0, tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator()(
      fixed_size_vec<num_dimensions()> auto const& y0, real_type const t0,
      real_type const tau) const -> pos_type {
    return evaluate(vec{y0}, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto integral_curve(pos_type const& y0, real_type const t0,
                      real_type const tau) const {
    integral_curve_type c;
    auto const          v       = c.push_back(y0);
    c.parameterization()[v]     = t0;
    auto const full_integration = continue_integration(c, tau);
    return std::pair{std::move(c), full_integration};
  }
  //----------------------------------------------------------------------------
  auto integral_curve(pos_type const& y0, real_type const t0,
                      real_type const btau, real_type const ftau) const {
    integral_curve_type c;
    auto const          v   = c.push_back(y0);
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
  auto continue_integration(integral_curve_type& integral_curve,
                            real_type const      tau) const -> bool {
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
      if (integral_curve.vertices().size() > 0 &&
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

    auto solver_copy = m_ode_solver;
    solver_copy.solve(vectorfield(), y0, t0, tau, callback);
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
  auto cached_curve(pos_type const& y0, real_type const t0) const
      -> auto const& {
    return *m_cache.emplace({t0, y0}).first->second;
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_type const& y0, real_type const t0,
                    real_type const tau) const -> auto const& {
    return cached_curve(y0, t0, tau < 0 ? tau : 0, tau > 0 ? tau : 0);
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_type const& y0, real_type const t0, real_type btau,
                    real_type ftau) const -> auto const& {
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
      polymorphic::vectorfield<real_type, num_dimensions()>* w) {
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
template <typename V, typename Real, std::size_t NumDimensions>
numerical_flowmap(vectorfield<V, Real, NumDimensions> const&)
    -> numerical_flowmap<V const&, ode::boost::rungekuttafehlberg78,
                         interpolation::cubic>;
//------------------------------------------------------------------------------
template <typename V, typename Real, std::size_t NumDimensions>
numerical_flowmap(vectorfield<V, Real, NumDimensions>&&)
    -> numerical_flowmap<V, ode::boost::rungekuttafehlberg78, interpolation::cubic>;
//------------------------------------------------------------------------------
template <typename V, typename Real, std::size_t NumDimensions,
          template <typename, std::size_t> typename ODESolver>
numerical_flowmap(vectorfield<V, Real, NumDimensions> const&,
                  ODESolver<Real, NumDimensions> const&)
    -> numerical_flowmap<V const&, ODESolver, interpolation::cubic>;
//------------------------------------------------------------------------------
template <typename V, typename Real, std::size_t NumDimensions,
          template <typename, std::size_t> typename ODESolver>
numerical_flowmap(vectorfield<V, Real, NumDimensions>&&,
                  ODESolver<Real, NumDimensions> const&)
    -> numerical_flowmap<V, ODESolver, interpolation::cubic>;
//==============================================================================
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions> const& v,
             tag::numerical_t /*tag*/) {
  return numerical_flowmap<V const&, ODESolver, InterpolationKernel>{v};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t> typename ODESolver,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions> const& v,
             ODESolver<Real, NumDimensions> const& ode_solver,
                 tag::numerical_t /*tag*/) {
  return numerical_flowmap<V const&, ODESolver, InterpolationKernel>{
      v, ode_solver};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions>&& v,
             tag::numerical_t /*tag*/) {
  return numerical_flowmap<V, ODESolver, InterpolationKernel>{std::move(v)};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t> typename ODESolver,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions>&& v,
             ODESolver<Real, NumDimensions> const& ode_solver,
                 tag::numerical_t /*tag*/) {
  return numerical_flowmap<V, ODESolver, InterpolationKernel>{std::move(v),
                                                              ode_solver};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions> const& v) {
  return numerical_flowmap<V const&, ODESolver, InterpolationKernel>(v);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t> typename ODESolver,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions> const& v,
             ODESolver<Real, NumDimensions> const&      ode_solver) {
  return numerical_flowmap<V const&, ODESolver, InterpolationKernel>{
      v, ode_solver};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions>&& v,
             ODESolver<Real, NumDimensions> const& ode_solver) {
  return numerical_flowmap<V, ODESolver, InterpolationKernel>(std::move(v),
                                                              ode_solver);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename V, typename Real, std::size_t NumDimensions>
auto flowmap(vectorfield<V, Real, NumDimensions>&& v) {
  return numerical_flowmap<V, ODESolver, InterpolationKernel>(std::move(v));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename Real, std::size_t NumDimensions>
auto flowmap(polymorphic::vectorfield<Real, NumDimensions> const& v) {
  return numerical_flowmap<polymorphic::vectorfield<Real, NumDimensions> const*,
                           ODESolver, InterpolationKernel>(&v);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <
    template <typename, std::size_t>
    typename ODESolver = ode::boost::rungekuttafehlberg78,
    template <typename> typename InterpolationKernel = interpolation::cubic,
    typename Real, std::size_t NumDimensions>
auto flowmap(polymorphic::vectorfield<Real, NumDimensions> const& v,
             ODESolver<Real, NumDimensions> const&                ode_solver) {
  return numerical_flowmap<polymorphic::vectorfield<Real, NumDimensions> const*,
                           ODESolver, InterpolationKernel>(&v, ode_solver);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/differentiated_flowmap.h>
//==============================================================================
namespace tatooine {
//==============================================================================
// typedefs
//==============================================================================
template <arithmetic Real, std::size_t NumDimensions,
          template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
using numerical_flowmap_field_pointer =
    numerical_flowmap<polymorphic::vectorfield<Real, NumDimensions>*, ODESolver,
                      InterpolationKernel>;
//==============================================================================
// type traits
//==============================================================================
template <typename T>
struct is_numerical_flowmap_impl : std::false_type {};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename V, template <typename, std::size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
struct is_numerical_flowmap_impl<
    numerical_flowmap<V, ODESolver, InterpolationKernel>> : std::true_type {};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename T>
static constexpr auto is_numerical_flowmap =
    is_numerical_flowmap_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
