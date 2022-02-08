#ifndef VC_ODEINT_HERMITESPLINE_HH
#define VC_ODEINT_HERMITESPLINE_HH
//=============================================================================
# include <cassert>
# include <cmath>     // std::isfinite
# include <vector>    // container
# include <tuple>
# include <algorithm> // std::lower_bound
# include <iostream>  // define operator>>
//=============================================================================
namespace VC {
namespace odeint {
namespace hermite {
//=============================================================================

/** Cubic Hermite interpolation.

    Interpolate at `_t` for given `_x0(_t0), _dx0(_t0), _x1(_t1),
    _dx1(_t1),`.

    \tparam T data type
    \tparam R real number type
    \tparam OnlyValue Don't compute `dx(t)` if `true`
*/

template <typename T, typename R, bool OnlyValue = false>
struct interpolator_t {
  const T x0, dx0, x1, dx1;
  const R t0, t1;

  interpolator_t(const T& _x0, const T& _dx0,
                         const T& _x1, const T& _dx1,
                         R _t0, R _t1)
      : x0(_x0), dx0(_dx0), x1(_x1), dx1(_dx1), t0(_t0), t1(_t1) {}

  interpolator_t(const interpolator_t<T, R, OnlyValue>&) = default;
  interpolator_t(interpolator_t<T, R, OnlyValue>&&) = default;

  // Note: This is a bit complex, because the instance is immutable
  //       (see const attributes); I;m unsure whether this it's worth it.
  auto& operator=(const interpolator_t<T, R, OnlyValue>& _other) {
    new(this) interpolator_t<T, R, OnlyValue> { _other };
    return *this;
  }

  /* Evaluate at `t`
     \return `(x(t), dx(y))` as `std::tuple<>` or `x(t)` if `OnlyValue`
  */
  auto operator()(R _t) const {
    // Exact interpolation at end points (no rounding error), and
    // handle degenerate case (empty interval).
    if (_t == t0  || t0 == t1) {
      if constexpr (OnlyValue) return x0;
      else                     return std::make_tuple(x0, dx0);
    }
    if (_t == t1) {
      if constexpr (OnlyValue) return x1;
      else                     return std::make_tuple(x1, dx1);
    }

    const R h = t1 - t0;
    const R t = (_t - t0) / h;

    assert(std::isfinite(t));

    const R cx0  = (1 + 2*t) * ((1-t)*(1-t));
    const R cdx0 = t * ((1-t)*(1-t));
    const R cx1  = (t*t) * (3 - 2*t);
    const R cdx1 = (t*t) * (t-1);

    const T x = x0*cx0 + dx0*(cdx0*h) + x1*cx1 + dx1*(cdx1*h);

    if constexpr (OnlyValue) {
      return x;
    }
    else {
      const R ex0  = t * (t-1) * 6;
      const R edx0 = (t-1) * (3*t - 1);
      const R ex1  = -ex0;
      const R edx1 = (3*t - 2) * t;

      const T dx =x0*(ex0/h) + dx0*edx0 + x1*(ex1/h) + dx1*edx1;

      return std::make_tuple(x, dx);
    }
  }

  bool is_in(R _t) const {
    return ((t0<=t1) && (t0<=_t && _t<=t1)) || (t0>=_t && _t>=t1);

  }
};

//-----------------------------------------------------------------------------

// TODO: split vector<real_type> vector<pair<vec_t,vec_t>> (faster search) -- push_back, pop_front, reverse, resize

template <typename T, typename R,
          typename Container = std::vector<std::tuple<R, T, T>>>
struct spline_t : public Container {
  using vec_t = T;
  using real_type = R;
  using container_t = Container;
  using value_type = typename container_t::value_type;

  real_type t(int i) const {
    assert(i<int(this->size()));
    return std::get<0>((*this)[i]);
  }
  vec_t x(int i) const {
        assert(i<int(this->size()));
        return std::get<1>((*this)[i]);
  }
  vec_t dx(int i) const {
    assert(i<int(this->size()));
    return std::get<2>((*this)[i]);
  }

  auto span() const {
    assert(!this->empty());
    return std::make_tuple(std::get<0>(this->front()),
                           std::get<0>(this->back()));
  }

  bool ascending() const { return this->size() < 2 || t(0) <= t(1); }

  bool is_monotone() const {
    if (this->size() < 2)
      return true;

    real_type d = t(1) - t(0);
    for (size_t i=1;i<this->size()-1;++i) {
      real_type d1 = t(i+1)-t(i);
      if (d*d1 < 0)
        return false;
      d = d1;
    }

    return true;
  }

  void push_back(value_type&& _p) {
    container_t::push_back(std::forward<value_type>(_p));
  }
  void push_back(real_type _t, const vec_t& _x, const vec_t& _dx) {
    this->emplace_back(_t, _x, _dx);
  }

  template <bool OnlyValue = false>
  auto interpolator(int i) const {
    assert(!this->empty());
    i = std::max(0,i);
    int j = std::min(i+1, int(this->size()));
    // VC_DBG_P(i);
    // VC_DBG_P(j);
    // VC_DBG_P(std::get<0>((*this)[i]));
    // VC_DBG_P(std::get<0>((*this)[j]));

    return interpolator_t<vec_t, real_type, OnlyValue> {
      std::get<1>((*this)[i]), std::get<2>((*this)[i]), // x0, dx0
      std::get<1>((*this)[j]), std::get<2>((*this)[j]), // x1, dx1
      std::get<0>((*this)[i]), std::get<0>((*this)[j]), // t0, t1
    };
  }

  int index(real_type _t) const {
    assert(!this->empty());

    const int n = int(this->size());

    if (n <= 2) {
      return 0;
    }
    // else if (n < 8) {
    // TODO
    // }
    auto ii =
        ascending() ?
        std::lower_bound(this->begin(), this->end()-1, _t,
                         [](const value_type& a, real_type b) {
                           return std::get<0>(a) < b;
                         })
        :
        std::lower_bound(this->begin(), this->end()-1, _t,
                         [](const value_type& a,real_type b) {
                           return std::get<0>(a) > b;
                         });

    const int d = std::distance(this->begin(), ii);
    return std::max(0, std::min(n-2, d-1));
  }

  template <bool OnlyValue = false>
  auto interpolator(real_type _t) const {
    return interpolator<OnlyValue>(index(_t));
  }


  template <bool OnlyValue = false>
  auto&
  update_interpolator(real_type _t,
                      interpolator_t<vec_t, real_type, OnlyValue>& _last) const {
    if (!_last.is_in(_t))
      _last = interpolator<OnlyValue>(_t);

    return _last;
  }

  auto operator()(real_type _t) const {
    return interpolator(_t)(_t);
  }
};

//-----------------------------------------------------------------------------

template <typename T, typename R>
std::ostream&
operator<<(std::ostream& _out, const spline_t<T,R>&  _s) {
  _out << "[ ";

  for (int i=0;i<int(_s.size());++i) {
    _out << _s.t(i) << "  " << _s.x(i) << "  " << _s.dx(i);
    if (i+1 < int(_s.size()))
      _out << "; ";
  }

  return _out << " ]";
}

//=============================================================================
} // namespace hermite
} // namespace odeint
} // namespace VC
//=============================================================================
#endif // VC_ODEINT_HERMITESPLINE_HH
