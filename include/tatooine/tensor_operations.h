#ifndef TATOOINE_TENSOR_OPERATIONS_H
#define TATOOINE_TENSOR_OPERATIONS_H
//==============================================================================
#include <tatooine/vec.h>
#include <tatooine/mat.h>
#include <tatooine/transposed_tensor.h>

#include <optional>
//==============================================================================
namespace tatooine {
//==============================================================================
/// invert symmetric matrix
/// A = [a,b]
///     [b,c]
template <typename Tensor, typename Real,
          enable_if<is_floating_point<Real>> = true>
constexpr auto inv_sym(base_tensor<Tensor, Real, 2, 2> const& A)
    -> std::optional<mat<Real, 2, 2>> {
  decltype(auto) a   = A(0, 0);
  decltype(auto) b   = A(1, 0);
  decltype(auto) c   = A(1, 1);
  auto const     det = (a * c - b * b);
  if (std::abs(det) < 1e-10) {
    return {};
  }
  auto const d = 1 / det;
  auto const e = -b * d;
  return mat{{c * d, e}, {e, a * d}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b]
///     [c,d]
#ifdef __cpp_concepts
template <typename Tensor, floating_point Real>
#else
template <typename Tensor, typename Real,
          enable_if<is_floating_point<Real>> = true>
#endif
constexpr auto inv(base_tensor<Tensor, Real, 2, 2> const& A)
    -> std::optional<mat<Real, 2, 2>> {
  decltype(auto) b = A(0, 1);
  decltype(auto) c = A(1, 0);
  if (std::abs(b - c) < 1e-10) {
    return inv_sym(A);
  }
  decltype(auto) a   = A(0, 0);
  decltype(auto) d   = A(1, 1);
  auto const     det = (a * d - b * c);
  if (std::abs(det) < 1e-10) {
    return {};
  }
  auto const e = 1 / det;
  return mat{{d, -b}, {-c, a}} * e;
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c]
///     [b,d,e]
///     [c,e,f]
#ifdef __cpp_concepts
template <typename Tensor, floating_point Real>
#else
template <typename Tensor, typename Real,
          enable_if<is_floating_point<Real>> = true>
#endif
constexpr auto inv_sym(base_tensor<Tensor, Real, 3, 3> const& A)
    -> std::optional<mat<Real, 3, 3>> {
  decltype(auto) a = A(0, 0);
  decltype(auto) b = A(1, 0);
  decltype(auto) c = A(2, 0);
  decltype(auto) d = A(1, 1);
  decltype(auto) e = A(2, 1);
  decltype(auto) f = A(2, 2);
  auto const     det =
      ((a * d - b * b) * f - a * e * e + 2 * b * c * e - c * c * d);
  if (std::abs(det) < 1e-10) {
    return {};
  }
  auto const div = 1 / det;
  return mat{
      {(d * f - e * e) * div, -(b * f - c * e) * div, (b * e - c * d) * div},
      {-(b * f - c * e) * div, (a * f - c * c) * div, -(a * e - b * c) * div},
      {(b * e - c * d) * div, -(a * e - b * c) * div, (a * d - b * b) * div}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b,c]
///     [d,e,f]
///     [g,h,i]
#ifdef __cpp_concepts
template <typename Tensor, floating_point Real>
#else
template <typename Tensor, typename Real,
          enable_if<is_floating_point<Real>> = true>
#endif
constexpr auto inv(base_tensor<Tensor, Real, 3, 3> const& A)
    -> std::optional<mat<Real, 3, 3>> {
  decltype(auto) b = A(0, 1);
  decltype(auto) c = A(0, 2);
  decltype(auto) d = A(1, 0);
  decltype(auto) g = A(2, 0);
  decltype(auto) f = A(1, 2);
  decltype(auto) h = A(2, 1);
  if (std::abs(b - d) < 1e-10 && std::abs(c - g) < 1e-10 &&
      std::abs(f - h) < 1e-10) {
    return inv_sym(A);
  }
  decltype(auto) a = A(0, 0);
  decltype(auto) e = A(1, 1);
  decltype(auto) i = A(2, 2);
  auto const     det =
      ((a * e - b * d) * i + (c * d - a * f) * h + (b * f - c * e) * g);
  if (std::abs(det) < 1e-10) {
    return {};
  }
  auto const div = 1 / det;
  return mat{{(e * i - f * h), -(b * i - c * h), (b * f - c * e)},
             {-(d * i - f * g), (a * i - c * g), -(a * f - c * d)},
             {(d * h - e * g), -(a * h - b * g), (a * e - b * d)}} *
         div;
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c,d]
///     [b,e,f,g]
///     [c,f,h,i]
///     [d,g,i,j]
#ifdef __cpp_concepts
template <typename Tensor, floating_point Real>
#else
template <typename Tensor, typename Real,
          enable_if<is_floating_point<Real>> = true>
#endif
constexpr auto inv_sym(base_tensor<Tensor, Real, 4, 4> const& A) {
  decltype(auto) a = A(0, 0);
  decltype(auto) b = A(1, 0);
  decltype(auto) c = A(2, 0);
  decltype(auto) d = A(3, 0);
  decltype(auto) e = A(1, 1);
  decltype(auto) f = A(2, 1);
  decltype(auto) g = A(3, 1);
  decltype(auto) h = A(2, 2);
  decltype(auto) i = A(3, 2);
  decltype(auto) j = A(3, 3);
  auto const     div =
      1 / (((a * e - b * b) * h - a * f * f + 2 * b * c * f - c * c * e) * j +
           (b * b - a * e) * i * i +
           ((2 * a * f - 2 * b * c) * g - 2 * b * d * f + 2 * c * d * e) * i +
           (-a * g * g + 2 * b * d * g - d * d * e) * h + c * c * g * g -
           2 * c * d * f * g + d * d * f * f);
  return mat{
      {((e * h - f * f) * j - e * i * i + 2 * f * g * i - g * g * h) * div,
       -((b * h - c * f) * j - b * i * i + (c * g + d * f) * i - d * g * h) *
           div,
       ((b * f - c * e) * j + (d * e - b * g) * i + c * g * g - d * f * g) *
           div,
       -((b * f - c * e) * i + (d * e - b * g) * h + c * f * g - d * f * f) *
           div},
      {-((b * h - c * f) * j - b * i * i + (c * g + d * f) * i - d * g * h) *
           div,
       ((a * h - c * c) * j - a * i * i + 2 * c * d * i - d * d * h) * div,
       -((a * f - b * c) * j + (b * d - a * g) * i + c * d * g - d * d * f) *
           div,
       ((a * f - b * c) * i + (b * d - a * g) * h + c * c * g - c * d * f) *
           div},
      {((b * f - c * e) * j + (d * e - b * g) * i + c * g * g - d * f * g) *
           div,
       -((a * f - b * c) * j + (b * d - a * g) * i + c * d * g - d * d * f) *
           div,
       ((a * e - b * b) * j - a * g * g + 2 * b * d * g - d * d * e) * div,
       -((a * e - b * b) * i + (b * c - a * f) * g + b * d * f - c * d * e) *
           div},
      {-((b * f - c * e) * i + (d * e - b * g) * h + c * f * g - d * f * f) *
           div,
       ((a * f - b * c) * i + (b * d - a * g) * h + c * c * g - c * d * f) *
           div,
       -((a * e - b * b) * i + (b * c - a * f) * g + b * d * f - c * d * e) *
           div,
       ((a * e - b * b) * h - a * f * f + 2 * b * c * f - c * c * e) * div}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b,c,d]
///     [e,f,g,h]
///     [i,j,k,l]
///     [m,n,o,p]
#ifdef __cpp_concepts
template <typename Tensor, floating_point Real>
#else
template <typename Tensor, typename Real,
          enable_if<is_floating_point<Real>> = true>
#endif
constexpr auto inv(base_tensor<Tensor, Real, 4, 4> const& A)
    -> std::optional<mat<Real, 4, 4>> {
  decltype(auto) b = A(0, 1);
  decltype(auto) c = A(0, 2);
  decltype(auto) d = A(0, 3);
  decltype(auto) e = A(1, 0);
  decltype(auto) g = A(1, 2);
  decltype(auto) h = A(1, 3);
  decltype(auto) i = A(2, 0);
  decltype(auto) j = A(2, 1);
  decltype(auto) l = A(2, 3);
  decltype(auto) m = A(3, 0);
  decltype(auto) n = A(3, 1);
  decltype(auto) o = A(3, 2);

  if (std::abs(b - e) < 1e-10 && std::abs(c - i) < 1e-10 &&
      std::abs(d - m) < 1e-10 && std::abs(g - j) < 1e-10 &&
      std::abs(h - n) < 1e-10 && std::abs(l - o) < 1e-10) {
    return inv_sym(A);
  }

  decltype(auto) a = A(0, 0);
  decltype(auto) f = A(1, 1);
  decltype(auto) k = A(2, 2);
  decltype(auto) p = A(3, 3);
  const auto     det =
      ((((a * f - b * e) * k + (c * e - a * g) * j + (b * g - c * f) * i) * p +
        ((b * e - a * f) * l + (a * h - d * e) * j + (d * f - b * h) * i) * o +
        ((a * g - c * e) * l + (d * e - a * h) * k + (c * h - d * g) * i) * n +
        ((c * f - b * g) * l + (b * h - d * f) * k + (d * g - c * h) * j) * m));
  if (std::abs(det) < 1e-10) {
    return {};
  }
  const auto div = 1 / det;
  return mat{
      {((f * k - g * j) * p + (h * j - f * l) * o + (g * l - h * k) * n) * div,
       -((b * k - c * j) * p + (d * j - b * l) * o + (c * l - d * k) * n) * div,
       ((b * g - c * f) * p + (d * f - b * h) * o + (c * h - d * g) * n) * div,
       -((b * g - c * f) * l + (d * f - b * h) * k + (c * h - d * g) * j) *
           div},
      {-((e * k - g * i) * p + (h * i - e * l) * o + (g * l - h * k) * m) * div,
       ((a * k - c * i) * p + (d * i - a * l) * o + (c * l - d * k) * m) * div,
       -((a * g - c * e) * p + (d * e - a * h) * o + (c * h - d * g) * m) * div,
       ((a * g - c * e) * l + (d * e - a * h) * k + (c * h - d * g) * i) * div},
      {((e * j - f * i) * p + (h * i - e * l) * n + (f * l - h * j) * m) * div,
       -((a * j - b * i) * p + (d * i - a * l) * n + (b * l - d * j) * m) * div,
       ((a * f - b * e) * p + (d * e - a * h) * n + (b * h - d * f) * m) * div,
       -((a * f - b * e) * l + (d * e - a * h) * j + (b * h - d * f) * i) *
           div},
      {-((e * j - f * i) * o + (g * i - e * k) * n + (f * k - g * j) * m) * div,
       ((a * j - b * i) * o + (c * i - a * k) * n + (b * k - c * j) * m) * div,
       -((a * f - b * e) * o + (c * e - a * g) * n + (b * g - c * f) * m) * div,
       ((a * f - b * e) * k + (c * e - a * g) * j + (b * g - c * f) * i) *
           div}};
}
//==============================================================================
/// Returns the cosine of the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
constexpr auto cos_angle(base_tensor<Tensor0, T0, N> const& v0,
                         base_tensor<Tensor1, T1, N> const& v1) {
  return dot(normalize(v0), normalize(v1));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, size_t... Dims>
constexpr auto abs(base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [](auto const& component) { return std::abs(component); }, t);
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
auto angle(base_tensor<Tensor0, T0, N> const& v0,
           base_tensor<Tensor1, T1, N> const& v1) {
  return std::acos(cos_angle(v0, v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
auto min_angle(base_tensor<Tensor0, T0, N> const& v0,
               base_tensor<Tensor1, T1, N> const& v1) {
  return std::min(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
auto max_angle(base_tensor<Tensor0, T0, N> const& v0,
               base_tensor<Tensor1, T1, N> const& v1) {
  return std::max(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename T0,
          typename T1, typename T2, size_t N>
constexpr auto cos_angle(base_tensor<Tensor0, T0, N> const& v0,
                         base_tensor<Tensor1, T1, N> const& v1,
                         base_tensor<Tensor2, T2, N> const& v2) {
  return cos_angle(v0 - v1, v2 - v1);
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename T0,
          typename T1, typename T2, size_t N>
auto angle(base_tensor<Tensor0, T0, N> const& v0,
           base_tensor<Tensor1, T1, N> const& v1,
           base_tensor<Tensor2, T2, N> const& v2) {
  return std::acos(cos_angle(v0, v1, v2));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t... Dims>
constexpr auto min(base_tensor<Tensor, T, Dims...> const& t) {
  T m = std::numeric_limits<T>::max();
  t.for_indices([&](auto const... is) { m = std::min(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t... Dims>
constexpr auto max(base_tensor<Tensor, T, Dims...> const& t) {
  T m = -std::numeric_limits<T>::max();
  t.for_indices([&](auto const... is) { m = std::max(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto norm(base_tensor<Tensor, T, N> const& t, unsigned p = 2) -> T {
  T n = 0;
  for (size_t i = 0; i < N; ++i) {
    n += std::pow(t(i), p);
  }
  return std::pow(n, T(1) / T(p));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto norm_inf(base_tensor<Tensor, T, N> const& t) -> T {
  T norm = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < N; ++i) {
    norm = std::max(norm, std::abs(t(i)));
  }
  return norm;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto norm1(base_tensor<Tensor, T, N> const& t) {
  return sum(abs(t));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto sqr_length(base_tensor<Tensor, T, N> const& t_in) {
  return dot(t_in, t_in);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto length(base_tensor<Tensor, T, N> const& t_in) -> T {
  return std::sqrt(sqr_length(t_in));
}
//------------------------------------------------------------------------------
/// squared p-norm of a rank-2 tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto sqr_norm(base_tensor<Tensor, T, M, N> const& A,
                        unsigned int const                  p) {
  if (p == 2) {
    return eigenvalues_sym(transposed(A) * A)(N - 1);
  }
  return T(0) / T(0);
}
//------------------------------------------------------------------------------
/// p-norm of a rank-2 tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& A,
                    unsigned int const                  p) {
  return std::sqrt(sqr_norm(A, p));
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto sqr_norm(base_tensor<Tensor, T, M, N> const& mat,
                        tag::frobenius_t) {
  T n = 0;
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < M; ++i) {
      n += std::abs(mat(i, j));
    }
  }
  return n;
}
//------------------------------------------------------------------------------
/// Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& mat, tag::frobenius_t) {
  return std::sqrt(sqr_norm(mat, tag::frobenius));
}
//------------------------------------------------------------------------------
/// 1-norm of a MxN Tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto norm1(base_tensor<Tensor, T, M, N> const& mat) {
  T          max    = -std::numeric_limits<T>::max();
  auto const absmat = abs(mat);
  for (size_t i = 0; i < N; ++i) {
    max = std::max(max, sum(absmat.template slice<1>(i)));
  }
  return max;
}
//------------------------------------------------------------------------------
/// infinity-norm of a MxN tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto norm_inf(base_tensor<Tensor, T, M, N> const& mat) {
  T max = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < M; ++i) {
    max = std::max(max, sum(abs(mat.template slice<0>(i))));
  }
  return max;
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto sqr_norm(base_tensor<Tensor, T, M, N> const& mat) {
  return sqr_norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& mat) {
  return norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto normalize(base_tensor<Tensor, T, N> const& t_in) -> vec<T, N> {
  auto const l = length(t_in);
  if (std::abs(l) < 1e-13) {
    return vec<T, N>::zeros();
  }
  return t_in / l;
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t N>
constexpr auto sqr_distance(base_tensor<Tensor0, T0, N> const& lhs,
                            base_tensor<Tensor1, T1, N> const& rhs) {
  return sqr_length(rhs - lhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t N>
constexpr auto distance(base_tensor<Tensor0, T0, N> const& lhs,
                        base_tensor<Tensor1, T1, N> const& rhs) {
  return length(rhs - lhs);
}
//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename Tensor, typename T, size_t VecDim>
constexpr auto sum(base_tensor<Tensor, T, VecDim> const& v) {
  T s = 0;
  for (size_t i = 0; i < VecDim; ++i) {
    s += v(i);
  }
  return s;
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t N>
constexpr auto dot(base_tensor<Tensor0, T0, N> const& lhs,
                   base_tensor<Tensor1, T1, N> const& rhs) {
  common_type<T0, T1> d = 0;
  for (size_t i = 0; i < N; ++i) {
    d += lhs(i) * rhs(i);
  }
  return d;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T>
constexpr auto det(base_tensor<Tensor, T, 2, 2> const& m) -> T {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T>
constexpr auto detAtA(base_tensor<Tensor, T, 2, 2> const& m) -> T {
  return m(0, 0) * m(0, 0) * m(1, 1) * m(1, 1) +
         m(0, 1) * m(0, 1) * m(1, 0) * m(1, 0) -
         2 * m(0, 0) * m(1, 0) * m(0, 1) * m(1, 1);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T>
constexpr auto det(base_tensor<Tensor, T, 3, 3> const& m) -> T {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1>
constexpr auto cross(base_tensor<Tensor0, T0, 3> const& lhs,
                     base_tensor<Tensor1, T1, 3> const& rhs) {
  return vec<common_type<T0, T1>, 3>{lhs(1) * rhs(2) - lhs(2) * rhs(1),
                                     lhs(2) * rhs(0) - lhs(0) * rhs(2),
                                     lhs(0) * rhs(1) - lhs(1) * rhs(0)};
}
//------------------------------------------------------------------------------
template <typename F, typename Tensor, typename T, size_t N>
constexpr auto unary_operation(F&& f, base_tensor<Tensor, T, N> const& t_in) {
  using TOut         = typename std::result_of<decltype(f)(T)>::type;
  vec<TOut, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, typename T, size_t M, size_t N>
constexpr auto unary_operation(F&&                                 f,
                               base_tensor<Tensor, T, M, N> const& t_in) {
  using TOut = typename std::result_of<decltype(f)(T)>::type;
  auto t_out = mat<TOut, M, N>{t_in};
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, typename T, size_t... Dims>
constexpr auto unary_operation(F&&                                    f,
                               base_tensor<Tensor, T, Dims...> const& t_in) {
  using TOut                  = typename std::result_of<decltype(f)(T)>::type;
  tensor<TOut, Dims...> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename Tensor0, typename T0, typename Tensor1,
          typename T1, size_t N>
constexpr auto binary_operation(F&& f, base_tensor<Tensor0, T0, N> const& lhs,
                                base_tensor<Tensor1, T1, N> const& rhs) {
  using TOut         = typename std::result_of<decltype(f)(T0, T1)>::type;
  vec<TOut, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, typename T0, typename Tensor1,
          typename T1, size_t M, size_t N>
constexpr auto binary_operation(F&&                                   f,
                                base_tensor<Tensor0, T0, M, N> const& lhs,
                                base_tensor<Tensor1, T1, M, N> const& rhs) {
  using TOut = typename std::result_of<decltype(f)(T0, T1)>::type;
  auto t_out = mat<TOut, M, N>{lhs};
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, typename T0, typename Tensor1,
          typename T1, size_t... Dims>
constexpr auto binary_operation(F&&                                      f,
                                base_tensor<Tensor0, T0, Dims...> const& lhs,
                                base_tensor<Tensor1, T1, Dims...> const& rhs) {
  using TOut = typename std::result_of<decltype(f)(T0, T1)>::type;
  tensor<TOut, Dims...> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
//------------------------------------------------------------------------------
template <typename TensorLhs, typename TensorRhs, typename T, size_t... Dims>
constexpr auto operator==(base_tensor<TensorLhs, T, Dims...> const& lhs,
                          base_tensor<TensorRhs, T, Dims...> const& rhs) {
  bool equal = true;
  for_loop(
      [&](auto const... is) {
        if (lhs(is...) != rhs(is...)) {
          equal = false;
          return;
        }
      },
      Dims...);
  return equal;
}
//------------------------------------------------------------------------------
template <typename TensorLhs, typename TensorRhs, typename T, size_t... Dims>
constexpr auto operator!=(base_tensor<TensorLhs, T, Dims...> const& lhs,
                          base_tensor<TensorRhs, T, Dims...> const& rhs) {
  return !(lhs == rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t... Dims>
constexpr auto operator-(base_tensor<Tensor, T, Dims...> const& t) {
  return unary_operation([](auto const& c) { return -c; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename T1, size_t... Dims,
          enable_if<is_arithmetic_or_complex<T1>> = true>
constexpr auto operator+(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         T1                                       scalar) {
  return unary_operation([scalar](auto const& c) { return c + scalar; }, lhs);
}
//------------------------------------------------------------------------------
/// matrix-matrix multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t M, size_t N, size_t O>
constexpr auto operator*(base_tensor<Tensor0, T0, M, N> const& lhs,
                         base_tensor<Tensor1, T1, N, O> const& rhs) {
  mat<common_type<T0, T1>, M, O> product;
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < O; ++c) {
      product(r, c) = dot(lhs.template slice<0>(r), rhs.template slice<1>(c));
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// component-wise multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t... Dims, enable_if<(sizeof...(Dims) != 2)> = true>
constexpr auto operator*(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::multiplies<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t... Dims>
constexpr auto operator/(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::divides<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t... Dims>
constexpr auto operator+(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::plus<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true, size_t... Dims>
constexpr auto operator*(base_tensor<Tensor, TensorT, Dims...> const& t,
                         Scalar const                                 scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true, size_t... Dims>
constexpr auto operator*(Scalar const                                 scalar,
                         base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true, size_t... Dims>
constexpr auto operator/(base_tensor<Tensor, TensorT, Dims...> const& t,
                         Scalar const                                 scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename TensorT, typename Scalar,
          enable_if<is_arithmetic_or_complex<Scalar>> = true, size_t... Dims>
constexpr auto operator/(Scalar const                                 scalar,
                         base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [scalar](auto const& component) { return scalar / component; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t... Dims>
constexpr auto operator-(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::minus<common_type<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t M, size_t N>
constexpr auto operator*(base_tensor<Tensor0, T0, M, N> const& lhs,
                         base_tensor<Tensor1, T1, N> const&    rhs) {
  vec<common_type<T0, T1>, M> product;
  for (size_t i = 0; i < M; ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}
//------------------------------------------------------------------------------
/// vector-matrix-multiplication
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t M, size_t N>
constexpr auto operator*(base_tensor<Tensor0, T0, M> const&    lhs,
                         base_tensor<Tensor1, T1, M, N> const& rhs) {
  vec<common_type<T0, T1>, N> product;
  for (size_t i = 0; i < N; ++i) {
    product(i) = dot(lhs, rhs.template slice<1>(i));
  }
  return product;
}
//------------------------------------------------------------------------------
template <typename T0, typename T1>
constexpr auto reflect(vec<T0, 3> const& incidentVec,
                       vec<T1, 3> const& normal) {
  return incidentVec - 2 * dot(incidentVec, normal) * normal;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/tensor_lapack_utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N>
auto solve(base_tensor<TensorA, Real, M, N> const& A,
           base_tensor<TensorB, Real, M> const&    b) {
  if constexpr (M == N) {
    return solve_lu(A, b);
  } else if constexpr (M > N) {
    return solve_qr(A, b);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t N>
auto solve_lu(base_tensor<TensorA, Real, N, N> const& A_base,
              base_tensor<TensorB, Real, N> const&    b_base) {
  auto                  A    = mat<Real, N, N>{A_base};
  auto                  b    = vec<Real, N>{b_base};
  auto                  ipiv = vec<int, N>{};
  [[maybe_unused]] auto info = lapack::gesv(A, b, ipiv);
  return b;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N>
auto solve_qr(base_tensor<TensorA, Real, M, N> const& A_base,
              base_tensor<TensorB, Real, M> const&    b_base) {
  auto A   = mat<Real, M, N>{A_base};
  auto b   = vec<Real, M>{b_base};
  auto tau = vec<Real, (M < N ? M : N)>{};

  // Q * R = A
  lapack::geqrf(A, tau);
  // R * x = Q^T * b
  lapack::ormqr(A, b, tau, ::lapack::Side::Left, ::lapack::Op::Trans);
  // Use back-substitution using the upper right part of A
  lapack::trtrs(A, b, ::lapack::Uplo::Upper, ::lapack::Op::NoTrans,
                ::lapack::Diag::NonUnit);
  for (size_t i = 0; i < tau.dimension(0); ++i) {
    tau(i) = b(i);
  }
  return tau;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N,
          size_t K>
auto solve(base_tensor<TensorA, Real, M, N> const& A,
           base_tensor<TensorB, Real, M, K> const& B) {
  if constexpr (M == N) {
    return solve_lu(A, B);
  } else if constexpr (M > N) {
    return solve_qr(A, B);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t N, size_t K>
auto solve_lu(base_tensor<TensorA, Real, N, N> const& A_base,
              base_tensor<TensorB, Real, N, K> const& B_base) {
  auto                  A    = mat<Real, N, N>{A_base};
  auto                  B    = mat<Real, N, K>{B_base};
  auto                  ipiv = vec<int, N>{};
  [[maybe_unused]] auto info = lapack::gesv(A, B, ipiv);
  return B;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N,
          size_t K>
auto solve_qr(base_tensor<TensorA, Real, M, N> const& A_base,
              base_tensor<TensorB, Real, M, K> const& B_base) {
  auto A   = mat<Real, M, N>{A_base};
  auto B   = mat<Real, M, K>{B_base};
  auto tau = vec<Real, (M < N ? M : N)>{};
  auto X   = mat<Real, N, K>{};

  // Q * R = A
  lapack::geqrf(A, tau);
  // R * x = Q^T * B
  lapack::ormqr(A, B, tau, ::lapack::Side::Left, ::lapack::Op::Trans);
  // Use back-substitution using the upper right part of A
  lapack::trtrs(A, B, ::lapack::Uplo::Upper, ::lapack::Op::NoTrans,
                ::lapack::Diag::NonUnit);
  for_loop(
      [&, i = size_t(0)](auto const... is) mutable { X(is...) = B(is...); }, N,
      K);
  return X;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, size_t... Dims>
constexpr auto sqrt(base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [](auto const& component) { return std::sqrt(component); }, t);
}
//------------------------------------------------------------------------------
///// compute condition number
//#ifdef __cpp_concepts
//template <typename T, size_t N, integral P = int>
//#else
//template <typename T, size_t N, typename P = int,
//          enable_if<is_integral<P>> = true>
//#endif
//auto condition_number(tensor<T, N, N> const& A, P const p = 2) {
//  if (p == 1) {
//    return 1 / lapack::gecon(tensor{A});
//  } else if (p == 2) {
//    auto const s = singular_values(A);
//    return s(0) / s(N - 1);
//  } else {
//    throw std::runtime_error{"p = " + std::to_string(p) +
//                             " is no valid base. p must be either 1 or 2."};
//  }
//}
//------------------------------------------------------------------------------
//template <typename Tensor, typename T, size_t N, typename PReal>
//auto condition_number(base_tensor<Tensor, T, N, N> const& A, PReal p) {
//  return condition_number(tensor{A}, p);
//}
//==============================================================================
// template <typename Tensor, typename Real>
// constexpr auto eigenvectors_sym(base_tensor<Tensor, Real, 2, 2> const& A) {
//  decltype(auto) b     = A(1, 0);
//  if (b == 0) {
//    return std::pair{mat<Real, 2, 2>::eye(), vec<Real, 2>{A(0, 0), A(1, 1)}};
//  }
//
//  decltype(auto) a     = A(0, 0);
//  decltype(auto) d     = A(1, 1);
//  auto const     e_sqr = d * d - 2 * a * d + 4 * b * b + a * a;
//  auto const     e     = std::sqrt(e_sqr);
//  constexpr auto half  = 1 / Real(2);
//  auto const     b2inv = 1 / (2 * b);
//  std::pair      out{mat<Real, 2, 2>{{Real(1), Real(1)},
//                                {-(e - d + a) * b2inv, (e + d - a) * b2inv}},
//                vec<Real, 2>{-(e - d - a) * half, (e + d + a) * half}};
//  if (out.second(0) > out.second(1)) {
//    std::swap(out.first(1, 0), out.first(1, 1));
//    std::swap(out.second(0), out.second(1));
//  }
//  if (out.first(1, 0) < 0) {
//    out.first.col(0) *= -1;
//  }
//  if (out.first(1, 1) < 0) {
//    out.first.col(1) *= -1;
//  }
//  out.first.col(0) /= std::sqrt(1 + out.first(1, 0) * out.first(1, 0));
//  out.first.col(1) /= std::sqrt(1 + out.first(1, 1) * out.first(1, 1));
//  return out;
//}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
auto eigenvectors_sym(base_tensor<Tensor, Real, N, N> const& A) {
  auto W = std::pair{mat<Real, N, N>{A}, vec<Real, N>{}};
  lapack::syev(::lapack::Job::Vec, ::lapack::Uplo::Upper,
               W.first, W.second);
  return W;
}
//==============================================================================
template <typename Tensor, typename Real>
constexpr auto eigenvalues_sym(base_tensor<Tensor, Real, 2, 2> const& A) {
  decltype(auto) b = A(1, 0);
  if (std::abs(b) <= 1e-11) {
    return vec<Real, 2>{A(0, 0), A(1, 1)};
  }
  decltype(auto) a      = A(0, 0);
  decltype(auto) d      = A(1, 1);
  auto const     e_sqr  = d * d - 2 * a * d + 4 * b * b + a * a;
  auto const     e      = std::sqrt(e_sqr);
  constexpr auto half   = 1 / Real(2);
  auto           lambda = vec<Real, 2>{-e + d + a, e + d + a} * half;
  if (lambda(0) > lambda(1)) {
    std::swap(lambda(0), lambda(1));
  }
  return lambda;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto eigenvalues_sym(base_tensor<Tensor, Real, N, N> const& A) {
  auto W = vec<Real, N>{};
  lapack::syev(::lapack::Job::NoVec, ::lapack::Uplo::Upper, mat<Real, N, N>{A},
               W);
  return W;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real>
constexpr auto eigenvalues_22(base_tensor<Tensor, Real, 2, 2> const& A)
    -> vec<std::complex<Real>, 2> {
  decltype(auto) b = A(1, 0);
  decltype(auto) c = A(0, 1);
  // if (std::abs(b - c) < 1e-10) {
  //  return eigenvalues_22_sym(A);
  //}
  decltype(auto) a   = A(0, 0);
  decltype(auto) d   = A(1, 1);
  auto const     sqr = d * d - 2 * a * d + 4 * b * c + a * a;

  vec<std::complex<Real>, 2> s;
  if (sqr >= 0) {
    s(0).real(-(std::sqrt(sqr) - d - a) / 2);
    s(1).real((std::sqrt(sqr) + d + a) / 2);
  } else {
    s(0).real((d + a) / 2);
    s(1).real(s(0).real());
    s(0).imag(std::sqrt(std::abs(sqr)) / 2);
    s(1).imag(-s(0).imag());
  }
  return s;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t N>
constexpr auto eigenvalues(base_tensor<Tensor, Real, N, N> const& A) {
  return eigenvalues(tensor<Real, N, N>{A});
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
constexpr auto eigenvalues(tensor<Real, N, N> A) -> vec<std::complex<Real>, N> {
  if constexpr (N == 2) {
    return eigenvalues_22(A);
  } else {
    auto                        eig  = tensor<std::complex<Real>, N>{};
    [[maybe_unused]] auto const info = lapack::geev(A, eig);

    return eig;
  }
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
auto eigenvectors(base_tensor<Tensor, Real, N, N> const& A) {
  return eigenvectors(tensor<Real, N, N>{A});
}
template <typename Real, size_t N>
auto eigenvectors(tensor<Real, N, N> A) {
  std::pair<mat<std::complex<Real>, N, N>, vec<std::complex<Real>, N>> eig;
  auto&                             V = eig.first;
  auto&                             W = eig.second;
  mat<Real, N, N>                   VR;
  [[maybe_unused]] auto const info = lapack::geev_right(A, W, VR);

  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      if (W[j].imag() == 0) {
        V(i, j) = {VR[i + j * N], 0};
      } else {
        V(i, j)     = {VR[i + j * N], VR[i + (j + 1) * N]};
        V(i, j + 1) = {VR[i + j * N], -VR[i + (j + 1) * N]};
        if (i == N - 1) {
          ++j;
        }
      }
    }
  }

  return eig;
}
//==============================================================================
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd(base_tensor<Tensor, T, M, N> const& A_base, tag::full_t [>tag<]) {
//  auto A = mat<T, M, N>{A_base};
//  return lapack::gesvd(A, lapack::job::A, lapack::job::A);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd(base_tensor<Tensor, T, M, N> const& A_base, tag::economy_t [>tag<]) {
//  auto A = mat<T, M, N>{A_base};
//  return lapack::gesvd(A, lapack::job::S, lapack::job::S);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd(base_tensor<Tensor, T, M, N> const& A) {
//  return svd(A, tag::full);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd_left(base_tensor<Tensor, T, M, N> const& A_base, tag::full_t [>tag<]) {
//  auto A = mat<T, M, N>{A_base};
//  return lapack::gesvd(A, lapack::job::A, lapack::job::N);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd_left(base_tensor<Tensor, T, M, N> const& A_base,
//              tag::economy_t [>tag<]) {
//  auto A = mat<T, M, N>{A_base};
//  return lapack::gesvd(A, lapack::job::S, lapack::job::N);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd_left(base_tensor<Tensor, T, M, N> const& A) {
//  return svd_left(A, tag::full);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd_right(base_tensor<Tensor, T, M, N> const& A_base,
//               tag::full_t [>tag<]) {
//  auto A = mat<T, M, N>{A_base};
//  return lapack::gesvd(A, lapack::job::N, lapack::job::A);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd_right(base_tensor<Tensor, T, M, N> const& A_base,
//               tag::economy_t [>tag<]) {
//  auto A = mat<T, M, N>{A_base};
//  return lapack::gesvd(A, lapack::job::N, lapack::job::S);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//auto svd_right(base_tensor<Tensor, T, M, N> const& A) {
//  return svd_right(A, tag::full);
//}
////------------------------------------------------------------------------------
//template <typename Tensor, typename T>
//constexpr auto singular_values22(base_tensor<Tensor, T, 2, 2> const& A) {
//  auto const a = A(0, 0);
//  auto const b = A(0, 1);
//  auto const c = A(1, 0);
//  auto const d = A(1, 1);
//
//  auto const aa     = a * a;
//  auto const bb     = b * b;
//  auto const cc     = c * c;
//  auto const dd     = d * d;
//  auto const s1     = aa + bb + cc + dd;
//  auto const s2     = std::sqrt((aa + bb - cc - dd) * (aa + bb - cc - dd) +
//                            4 * (a * c + b * d) * (a * c + b * d));
//  auto const sigma1 = std::sqrt((s1 + s2) / 2);
//  auto const sigma2 = std::sqrt((s1 - s2) / 2);
//  return vec{tatooine::max(sigma1, sigma2), tatooine::min(sigma1, sigma2)};
//}
////------------------------------------------------------------------------------
//template <typename T, size_t M, size_t N>
//constexpr auto singular_values(tensor<T, M, N>&& A) {
//  if constexpr (M == 2 && N == 2) {
//    return singular_values22(A);
//  } else {
//    return gesvd(A, lapack::job::N, lapack::job::N);
//  }
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Tensor, typename T, size_t M, size_t N>
//constexpr auto singular_values(base_tensor<Tensor, T, M, N> const& A_base) {
//  if constexpr (M == 2 && N == 2) {
//    return singular_values22(A_base);
//  } else {
//    auto A = mat<T, M, N>{A_base};
//    return lapack::gesvd(A, lapack::job::N, lapack::job::N);
//  }
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
