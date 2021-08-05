#ifndef VC_ODEINT_TESTS_HH
#define VC_ODEINT_TESTS_HH
//=============================================================================
# include <cassert>
# include <cmath>
//-----------------------------------------------------------------------------
# include "odeint.hh"
//=============================================================================
namespace VC {
namespace odeint {
namespace tests {
//=============================================================================

/** Non-stiff test problems.

    From

    Enright and Pryce, <em>Two FORTRAN packages for assessing initial
    value methods</em>, ACM Transactions on Mathematical Software,
    13(1), 1987, 193--218
*/
template <typename R>
struct detest {

  using real_t = R;

  // A: single equations

  struct A1_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static auto dy(real_t t, vec_t y) { return -y; }
    static constexpr real_t t1 = 20;
    static constexpr real_t y0 = 1;
    static auto y(real_t t) { return std::exp(-t); }
  };
  struct A2_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static auto dy(real_t t, vec_t y) { return -(y*y*y)/2; }
    static constexpr real_t t1 = 20;
    static constexpr real_t y0 = 1;
    static auto y(real_t t) { return real_t(1)/std::sqrt(1+t); }
  };
  struct A3_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static auto dy(real_t t, vec_t y) { return y*cos(t); }
    static constexpr real_t t1 = 20;
    static constexpr real_t y0 = 1;
    static auto y(real_t t) { return std::exp(sin(t)); }
  };
  struct A4_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static auto dy(real_t t, vec_t y) { return y/4*(1-y/20); }
    static constexpr real_t t1 = 20;
    static constexpr real_t y0 = 1;
    static auto y(real_t t) {
      return real_t(20)/(1+19*std::exp(-t/4));
    }
  };
  struct A5_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static auto dy(real_t t, vec_t y) { return (y-t)/(y+t); }
    static constexpr real_t t1 = 20;
    static constexpr real_t y0 = 4;
  };

  // B: small systems

  struct B1_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { 2 * (y[0] - y[0] * y[1]), -(y[1] - y[0] * y[1]) };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= {1, 3};
  };
  struct B2_t {
    static constexpr int dim = 3;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { -y[0] + y[1], y[0] - 2 * y[1] + y[2], y[1] - y[2] };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 = {2, 0, 1};
  };
  struct B3_t {
    static constexpr int dim = 3;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { -y[0], y[0] - 2 * y[1] + y[2], y[1] - y[2] };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 = {1, 0, 0};
  };
  struct B4_t {
    static constexpr int dim = 3;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      const real_t d = std::hypot(y[0], y[1]);
      return { -y[1]-(y[0]*y[2])/d, y[0]-(y[1]*y[2])/d, y[0]/d };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= {3, 0, 0};
  };
  struct B5_t {
    static constexpr int dim = 3;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { y[1] * y[2], -y[0] * y[2], -real_t(0.51) * y[0] * y[1] };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= {0, 1, 1};
  };

  //  C: moderate systems

  struct C1_t {
    static constexpr int dim = 10;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      vec_t r;
      r[0] = -y[0];
      for (int i=1;i<9;++i)
        r[i] = y[i-1]-y[i];
      r[9] = y[8];
      return r;
    }
    static constexpr real_t t1 = 20;
   static inline const vec_t y0= { 1, 0 /* ... */};
  };

  struct C2_t {
    static constexpr int dim = 10;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      vec_t r;
      r[0] = -y[0];
      for (int i=1;i<9;++i)
        r[i] = i*y[i-1]-(i+1)*y[i];
      r[9] = 9*y[8];
      return r;
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 1, 0 /* ... */};
  };

  struct C3_t {
    static constexpr int dim = 10;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      vec_t r;
      r[0] = -2*y[0]+y[1];
      for (int i=1;i<9;++i)
        r[i] = y[i-1] -2*y[i] + y[i+1];
      r[9] = y[8] -2*y[9];
      return r;
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 1, 0 /* ... */};
  };

  struct C4_t {
    static constexpr int dim = 51;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      vec_t r;
      r[0] = -2*y[0]+y[1];
      for (int i=1;i<50;++i)
        r[i] = y[i-1] -2*y[i] + y[i+1];
      r[50] = y[49] -2*y[50];
      return r;
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 { 1, 0 /* ... */};
  };

  // MISSING: C5_t

  // D: orbit equations

  template <int I>
  struct Di {
    static constexpr int dim = 4;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static constexpr real_t EPS = real_t(I)/10;
    static vec_t dy(real_t t, const vec_t& y) {
      const real_t d = pow(y[0]*y[0]+y[1]*y[1], 1.5);
      return { y[2], y[3], -y[0]/d, -y[1]/d };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 { 1 - EPS, 0, 0, std::sqrt((1 + EPS) / (1 - EPS)) };
  };
  using D1_t = Di<1>;
  using D2_t = Di<3>;
  using D3_t = Di<5>;
  using D4_t = Di<7>;
  using D5_t = Di<9>;

  // E: higher order equations

  struct E1_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { y[1], -(y[1]/(t+1)+(1-real_t(0.25)/((t+1)*(t+1)))*y[0]) };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= {
      real_t(0.6713967071418030), real_t(0.09540051444747446)
    };
  };

  struct E2_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { y[1], (1-y[0]*y[0])*y[1]-y[0]};
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 2, 0 };
  };

  struct E3_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return {
        y[1], y[0] * y[0] * y[0] / 6 - y[0] + 2 * sin(real_t(2.78535) * t)
      };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 0, 0 };
  };

  struct E4_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { y[1], real_t(0.032) - real_t(0.4)*y[1]*y[1] };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 30, 0 };
  };

  struct E5_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return { y[1], std::sqrt(1 + y[1] * y[1]) / (25 - t) };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 0, 0 };
  };

  // F: problems with discontinuities

  struct F1_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      constexpr real_t a = 0.1;
      real_t dy2 = 2*a*y[1]-(real_t(M_PI*M_PI)+a*a)*y[0];

      if (int(std::floor(t))%2 == 0)
        dy2 += 1;
      else
        dy2 -= 1;

      return { y[1], dy2 };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 0, 0 };
  };

  struct F2_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static vec_t dy(real_t t, vec_t y) {
      return 55 - (int(std::floor(t))%2 == 0 ? 3*y/2 : y/2);
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 = 110;
  };

  struct F3_t {
    static constexpr int dim = 2;
    using vec_t = VC::vecn::vecn<dim, real_t>;
    static vec_t dy(real_t t, vec_t y) {
      return {
        y[1], real_t(0.01)*(1-y[0]*y[0])-y[0]-std::abs(sin(real_t(M_PI)*t))
      };
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0= { 0, 0 };
  };

  struct F4_t {
    static constexpr int dim = 1;
    using vec_t = real_t;

    // Note: This one is very sensitive to step size base relerr -> 0)!
    static auto dy(real_t t, vec_t y) -> maybe<vec_t> {
      if (!(0<=t && t<=20))
        return OutOfDomain;
      if (t<=10) {
        return -real_t(2)/21 - 120*(t-5) /(1+4*(t-5)*(t-5));
      }
      else {
        return -2*y;
      }
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 = 1;
  };

  struct F5_t {
    static constexpr int dim = 1;
    using vec_t = real_t;
    static real_t c() {
      real_t sum = 0;
      for (int i=0;i<19;++i)
        sum += std::pow(real_t(i),real_t(4)/3);
      return sum;
    };
    static vec_t dy(real_t t, vec_t y) {
      static const real_t c = F5_t::c();
      real_t dp = 0;
      for (int i=0;i<19;++i)
        dp += (real_t(4)/3)*std::pow(std::abs(t-i), real_t(1)/3); // abs?!
      return 1/c * dp * y;
    }
    static constexpr real_t t1 = 20;
    static inline const vec_t y0 = 1;
  };

  //---------------------------------------------------------------------------

  template <typename Test> static auto make_ode() {
    if constexpr (Test::dim == 1) {
      return VC::odeint::ode_t<1, real_t, real_t> {};
    }
    else {
      return VC::odeint::ode_t<Test::dim, real_t> {};
    }
  }

  template <typename Test>
  using ode_t = decltype(make_ode<Test>());

  template <typename Test, typename Solver>
  static auto test(const Test&, Solver& _solver) {
    auto success =  _solver.integrate(Test::dy, 0, Test::t1, Test::y0);
    assert(success == evstate_t::OK);
    return _solver.y;
  }

  template <typename Test, typename Stepper, typename... Opts>
  static auto test_with(const Test&, const Stepper& stepper, Opts&&... o) {
    using ode = ode_t<Test>;
    auto opts = ode::make_options(std::forward<Opts>(o)...);
    //VC_DBG_P(opts);
    auto solver = ode::solver(stepper, opts);
    auto success = solver.integrate(Test::dy, 0, Test::t1, Test::y0);
    assert(success == evstate_t::OK);
    return solver.y;
  }

  template <typename Test>
  static auto reference(const Test& t) {
    return test_with(t, RK43,
                     MaxStep= 1e-5,
                     AbsTol= 1e-4, RelTol= 1e-4,
                     MaxNumSteps= 100000000);
  }

  template <typename Test, typename... Args>
  static auto values(const Test& t, Args&&... args) {
    auto yt = test_with(t, std::forward<Args>(args)...);
    auto yr = reference(t);
    return std::make_pair(yt, yr);
  }

  template <typename Test, typename... Args>
  static auto diff(const Test&t , Args&&... args) {
    auto [yt, yr] = values(t, std::forward<Args>(args)...);
    return yr-yt;
  }

  template <typename Test, typename... Args>
  static auto reldiff(const Test&t , Args&&... args) {
    auto [yt, yr] = values(t, std::forward<Args>(args)...);
    return (yr-yt)/std::max(1e-3, std::max(norm(yt), norm(yr)));
  }

  template <int D>
  static real_t norm(const VC::vecn::vecn<D, real_t>& _x) {
    return VC::vecn::norm(_x);
  }
  static real_t norm(real_t _x) { return std::abs(_x); }

  template <typename Test, typename... Args>
  static auto error(const Test& t, Args&&... args) {
    return norm(reldiff(t, std::forward<Args>(args)...) / Test::dim);
  }

  //---------------------------------------------------------------------------

  inline static const A1_t A1 {};
  inline static const A2_t A2 {};
  inline static const A3_t A3 {};
  inline static const A4_t A4 {};
  inline static const A5_t A5 {};

  inline static const B1_t B1 {};
  inline static const B2_t B2 {};
  inline static const B3_t B3 {};
  inline static const B4_t B4 {};
  inline static const B5_t B5 {};

  inline static const C1_t C1 {};
  inline static const C2_t C2 {};
  inline static const C3_t C3 {};
  inline static const C4_t C4 {};
  //inline static const C5_T c5 {};

  inline static const D1_t D1 {};
  inline static const D2_t D2 {};
  inline static const D3_t D3 {};
  inline static const D4_t D4 {};
  inline static const D5_t D5 {};

  inline static const E1_t E1 {};
  inline static const E2_t E2 {};
  inline static const E3_t E3 {};
  inline static const E4_t E4 {};
  inline static const E5_t E5 {};

  inline static const F1_t F1 {};
  inline static const F2_t F2 {};
  inline static const F3_t F3 {};
  inline static const F4_t F4 {};
  inline static const F5_t F5 {};

};  // struct detest

//=============================================================================
}  // namespace tests
}  // namespace odeint
}  // namespace VC
//=============================================================================
#endif  // VC_ODEINT_TEST_HH
