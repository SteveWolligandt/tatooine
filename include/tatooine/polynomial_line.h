#ifndef TATOOINE_POLYNOMIAL_LINE_H
#define TATOOINE_POLYNOMIAL_LINE_H
//==============================================================================
#include <tatooine/available_libraries.h>
#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/math.h>
#include <tatooine/polynomial.h>

#include <array>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t N, std::size_t Degree>
class polynomial_line {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using vec_type        = vec<Real, N>;
  using polynomial_type = tatooine::polynomial<Real, Degree>;

  //----------------------------------------------------------------------------
  // static methods
  //----------------------------------------------------------------------------
 public:
  static constexpr auto num_dimensions() { return N; }
  static constexpr auto degree() { return Degree; }

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  std::array<polynomial_type, N> m_polynomials;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr polynomial_line()
      : m_polynomials{make_array<polynomial_type, N>()} {}
  constexpr polynomial_line(const polynomial_line& other) = default;
  constexpr polynomial_line(polynomial_line&& other)      = default;
  constexpr polynomial_line& operator=(const polynomial_line& other) = default;
  constexpr polynomial_line& operator=(polynomial_line&& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Polynomials>
  requires(is_polynomial<Polynomials> && ...)
  constexpr polynomial_line(Polynomials&&... polynomials)
      : m_polynomials{std::forward<Polynomials>(polynomials)...} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  auto&       polynomial(std::size_t i) { return m_polynomials[i]; }
  const auto& polynomial(std::size_t i) const { return m_polynomials[i]; }
  //----------------------------------------------------------------------------
  auto&       polynomials() { return m_polynomials; }
  const auto& polynomials() const { return m_polynomials; }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto evaluate(Real t, std::index_sequence<Is...> /*is*/) const {
    return vec_type{m_polynomials[Is](t)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto evaluate(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto diff(std::index_sequence<Is...> /*is*/) const {
    return polynomial_line{diff(m_polynomials[Is])...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto diff() const { return diff(std::make_index_sequence<N>{}); }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto tangent(Real t, std::index_sequence<Is...> /*is*/) const {
    return vec_type{m_polynomials[Is].diff()(t)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto tangent(Real t) const {
    return tangent(t, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Is>
  constexpr auto second_derivative(Real t,
                                   std::index_sequence<Is...> /*is*/) const {
    return vec_type{m_polynomials[Is].diff().diff()(t)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto second_derivative(Real t) const {
    return second_derivative(t, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  constexpr auto curvature(Real t) const {
    return curvature(tangent(t), second_derivative(t));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto curvature(Real t, const vec_type& tang) const {
    return curvature(tang, second_derivative(t));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto curvature(const vec_type& tang, const vec_type& snd_der) const
      -> Real {
    const auto ltang = length(tang);
    if (std::abs(ltang) < 1e-10) {
      return 0;
    }
    if constexpr (N == 2) {
      return std::abs(tang(0) * snd_der(1) - tang(1) * snd_der(0)) /
             (ltang * ltang * ltang);
    } else if constexpr (N == 3) {
      return length(cross(tang, snd_der)) / (ltang * ltang * ltang);
    }
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel,
            floating_point Real_>
  constexpr auto evaluate(linspace<Real_> const& ts) const {
    auto  discretized = line<Real, N>{};
    auto& param       = discretized.parameterization();
    auto& tang        = discretized.tangents();
    auto& snd_der =
        discretized.template vertex_property<vec<Real, N>>("second_derivative");
    auto& curv = discretized.scalar_vertex_property("curvature");
    for (auto const t : ts) {
      discretized.push_back(evaluate(t));
      param.back()   = t;
      tang.back()    = tangent(t);
      snd_der.back() = second_derivative(t);
      curv.back()    = curvature(tang.back(), snd_der.back());
    }
    return discretized;
  }
  //----------------------------------------------------------------------------
  constexpr auto arc_length(const linspace<Real>& range) const {
    Real l = 0;
    for (std::size_t i = 0; i < size(range) - 1; ++i) {
      l += distance(evaluate(range[i]), evaluate(range[i + 1]));
    }
    return l;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Polynomials>
polynomial_line(Polynomials&&...)
    -> polynomial_line<common_type<typename Polynomials::real_type...>,
                       sizeof...(Polynomials), max(Polynomials::degree()...)>;
//==============================================================================
template <typename Real, std::size_t N, std::size_t Degree>
auto operator<<(std::ostream& out, const polynomial_line<Real, N, Degree>& line)
    -> std::ostream& {
  out << "[(" << line.polynomial(0) << ")";
  for (std::size_t i = 1; i < N; ++i) {
    out << ", (" << line.polynomial(i) << ")";
  }
  out << "]";
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
