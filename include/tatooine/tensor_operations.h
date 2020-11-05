#ifndef TATOOINE_TENSOR_OPERATIONS_H
#define TATOOINE_TENSOR_OPERATIONS_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// invert symmetric matrix
/// A = [a,b]
///     [b,c]
template <typename Tensor, real_number T>
constexpr auto inv_sym(base_tensor<Tensor, T, 2, 2> const& A) {
  auto const& a = A(0, 0);
  auto const& b = A(1, 0);
  auto const& c = A(1, 1);
  auto const  d = 1 / (a * c - b * b);
  auto const  e = -b * d;
  return mat{{c * d, e}, {e, a * d}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b]
///     [c,d]
template <typename Tensor, real_number T>
constexpr auto inv(base_tensor<Tensor, T, 2, 2> const& A) {
  auto const& b = A(0, 1);
  auto const& c = A(1, 0);
  if (std::abs(b - c) < 1e-10) {
    return inv_sym(A);
  }
  auto const& a = A(0, 0);
  auto const& d = A(1, 1);
  auto const  e = 1 / (a * d - b * c);
  return mat{{d * e, -b * e}, {-c * e, a * e}};
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c]
///     [b,d,e]
///     [c,e,f]
template <typename Tensor, real_number T>
constexpr auto inv_sym(base_tensor<Tensor, T, 3, 3> const& A) {
  auto const& a = A(0, 0);
  auto const& b = A(1, 0);
  auto const& c = A(2, 0);
  auto const& d = A(1, 1);
  auto const& e = A(2, 1);
  auto const& f = A(2, 2);
  auto const  div =
      1 / ((a * d - b * b) * f - a * e * e + 2 * b * c * e - c * c * d);
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
template <typename Tensor, real_number T>
constexpr auto inv(base_tensor<Tensor, T, 3, 3> const& A) {
  auto const& b = A(0, 1);
  auto const& c = A(0, 2);
  auto const& d = A(1, 0);
  auto const& g = A(2, 0);
  auto const& f = A(1, 2);
  auto const& h = A(2, 1);
  if (std::abs(b - d) < 1e-10 && std::abs(c - g) < 1e-10 &&
      std::abs(f - h) < 1e-10) {
    return inv_sym(A);
  }
  auto const& a = A(0, 0);
  auto const& e = A(1, 1);
  auto const& i = A(2, 2);
  auto const  div =
      1 / ((a * e - b * d) * i + (c * d - a * f) * h + (b * f - c * e) * g);
  return mat{
      {(e * i - f * h) * div, -(b * i - c * h) * div, (b * f - c * e) * div},
      {-(d * i - f * g) * div, (a * i - c * g) * div, -(a * f - c * d) * div},
      {(d * h - e * g) * div, -(a * h - b * g) * div, (a * e - b * d) * div}};
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c,d]
///     [b,e,f,g]
///     [c,f,h,i]
///     [d,g,i,j]
template <typename Tensor, real_number T>
constexpr auto inv_sym(base_tensor<Tensor, T, 4, 4> const& A) {
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
template <typename Tensor, real_number T>
constexpr auto inv(base_tensor<Tensor, T, 4, 4> const& A) {
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
  const auto     div =
      1 /
      ((((a * f - b * e) * k + (c * e - a * g) * j + (b * g - c * f) * i) * p +
        ((b * e - a * f) * l + (a * h - d * e) * j + (d * f - b * h) * i) * o +
        ((a * g - c * e) * l + (d * e - a * h) * k + (c * h - d * g) * i) * n +
        ((c * f - b * g) * l + (b * h - d * f) * k + (d * g - c * h) * j) * m));
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
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
constexpr auto cos_angle(base_tensor<Tensor0, T0, N> const& v0,
                         base_tensor<Tensor1, T1, N> const& v1) {
  return dot(normalize(v0), normalize(v1));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto abs(base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [](auto const& component) { return std::abs(component); }, t);
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
auto angle(base_tensor<Tensor0, T0, N> const& v0,
           base_tensor<Tensor1, T1, N> const& v1) {
  return std::acos(cos_angle(v0, v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
auto min_angle(base_tensor<Tensor0, T0, N> const& v0,
               base_tensor<Tensor1, T1, N> const& v1) {
  return std::min(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
auto max_angle(base_tensor<Tensor0, T0, N> const& v0,
               base_tensor<Tensor1, T1, N> const& v1) {
  return std::max(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, real_number T0,
          real_number T1, real_number T2, size_t N>
constexpr auto cos_angle(base_tensor<Tensor0, T0, N> const& v0,
                         base_tensor<Tensor1, T1, N> const& v1,
                         base_tensor<Tensor2, T2, N> const& v2) {
  return cos_angle(v0 - v1, v2 - v1);
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, real_number T0,
          real_number T1, real_number T2, size_t N>
auto angle(base_tensor<Tensor0, T0, N> const& v0,
           base_tensor<Tensor1, T1, N> const& v1,
           base_tensor<Tensor2, T2, N> const& v2) {
  return std::acos(cos_angle(v0, v1, v2));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t... Dims>
constexpr auto min(base_tensor<Tensor, T, Dims...> const& t) {
  T m = std::numeric_limits<T>::max();
  t.for_indices([&](auto const... is) { m = std::min(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t... Dims>
constexpr auto max(base_tensor<Tensor, T, Dims...> const& t) {
  T m = -std::numeric_limits<T>::max();
  t.for_indices([&](auto const... is) { m = std::max(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto norm(base_tensor<Tensor, T, N> const& t, unsigned p = 2) -> T {
  T n = 0;
  for (size_t i = 0; i < N; ++i) {
    n += std::pow(t(i), p);
  }
  return std::pow(n, T(1) / T(p));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto norm_inf(base_tensor<Tensor, T, N> const& t) -> T {
  T norm = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < N; ++i) {
    norm = std::max(norm, std::abs(t(i)));
  }
  return norm;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto norm1(base_tensor<Tensor, T, N> const& t) {
  return sum(abs(t));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto sqr_length(base_tensor<Tensor, T, N> const& t_in) {
  return dot(t_in, t_in);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto length(base_tensor<Tensor, T, N> const& t_in) -> T {
  return std::sqrt(sqr_length(t_in));
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
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
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& mat, tag::frobenius_t) {
  return std::sqrt(sqr_norm(mat, tag::frobenius));
}
//------------------------------------------------------------------------------
/// 1-norm of a MxN Tensor
template <typename Tensor, real_number T, size_t M, size_t N>
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
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm_inf(base_tensor<Tensor, T, M, N> const& mat) {
  T max = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < M; ++i) {
    max = std::max(max, sum(abs(mat.template slice<0>(i))));
  }
  return max;
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto sqr_norm(base_tensor<Tensor, T, M, N> const& mat) {
  return sqr_norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& mat) {
  return norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto normalize(base_tensor<Tensor, T, N> const& t_in) -> vec<T, N> {
  auto const l = length(t_in);
  if (std::abs(l) < 1e-13) {
    return vec<T, N>::zeros();
  }
  return t_in / l;
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_number T0, typename Tensor1, real_number T1,
          size_t N>
constexpr auto sqr_distance(base_tensor<Tensor0, T0, N> const& lhs,
                            base_tensor<Tensor1, T1, N> const& rhs) {
  return sqr_length(rhs - lhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_number T0, typename Tensor1, real_number T1,
          size_t N>
constexpr auto distance(base_tensor<Tensor0, T0, N> const& lhs,
                        base_tensor<Tensor1, T1, N> const& rhs) {
  return length(rhs - lhs);
}
//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename Tensor, real_or_complex_number T, size_t VecDim>
constexpr auto sum(base_tensor<Tensor, T, VecDim> const& v) {
  T s = 0;
  for (size_t i = 0; i < VecDim; ++i) {
    s += v(i);
  }
  return s;
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t N>
constexpr auto dot(base_tensor<Tensor0, T0, N> const& lhs,
                   base_tensor<Tensor1, T1, N> const& rhs) {
  promote_t<T0, T1> d = 0;
  for (size_t i = 0; i < N; ++i) {
    d += lhs(i) * rhs(i);
  }
  return d;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T>
constexpr auto det(base_tensor<Tensor, T, 2, 2> const& m) -> T {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T>
constexpr auto detAtA(base_tensor<Tensor, T, 2, 2> const& m) -> T {
  return m(0, 0) * m(0, 0) * m(1, 1) * m(1, 1) +
         m(0, 1) * m(0, 1) * m(1, 0) * m(1, 0) -
         2 * m(0, 0) * m(1, 0) * m(0, 1) * m(1, 1);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_or_complex_number T>
constexpr auto det(base_tensor<Tensor, T, 3, 3> const& m) -> T {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1>
constexpr auto cross(base_tensor<Tensor0, T0, 3> const& lhs,
                     base_tensor<Tensor1, T1, 3> const& rhs) {
  return vec<promote_t<T0, T1>, 3>{lhs(1) * rhs(2) - lhs(2) * rhs(1),
                                   lhs(2) * rhs(0) - lhs(0) * rhs(2),
                                   lhs(0) * rhs(1) - lhs(1) * rhs(0)};
}
//------------------------------------------------------------------------------
template <typename F, typename Tensor, real_or_complex_number T, size_t N>
constexpr auto unary_operation(F&& f, base_tensor<Tensor, T, N> const& t_in) {
  using TOut         = typename std::result_of<decltype(f)(T)>::type;
  vec<TOut, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, real_or_complex_number T, size_t M,
          size_t N>
constexpr auto unary_operation(F&&                                 f,
                               base_tensor<Tensor, T, M, N> const& t_in) {
  using TOut            = typename std::result_of<decltype(f)(T)>::type;
  mat<TOut, M, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, real_or_complex_number T, size_t... Dims>
constexpr auto unary_operation(F&&                                    f,
                               base_tensor<Tensor, T, Dims...> const& t_in) {
  using TOut                  = typename std::result_of<decltype(f)(T)>::type;
  tensor<TOut, Dims...> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename Tensor0, real_or_complex_number T0,
          typename Tensor1, real_or_complex_number T1, size_t N>
constexpr auto binary_operation(F&& f, base_tensor<Tensor0, T0, N> const& lhs,
                                base_tensor<Tensor1, T1, N> const& rhs) {
  using TOut         = typename std::result_of<decltype(f)(T0, T1)>::type;
  vec<TOut, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, real_or_complex_number T0,
          typename Tensor1, real_or_complex_number T1, size_t M, size_t N>
constexpr auto binary_operation(F&&                                   f,
                                base_tensor<Tensor0, T0, M, N> const& lhs,
                                base_tensor<Tensor1, T1, M, N> const& rhs) {
  using TOut            = typename std::result_of<decltype(f)(T0, T1)>::type;
  mat<TOut, M, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, real_or_complex_number T0,
          typename Tensor1, real_or_complex_number T1, size_t... Dims>
constexpr auto binary_operation(F&&                                      f,
                                base_tensor<Tensor0, T0, Dims...> const& lhs,
                                base_tensor<Tensor1, T1, Dims...> const& rhs) {
  using TOut = typename std::result_of<decltype(f)(T0, T1)>::type;
  tensor<TOut, Dims...> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T, size_t... Dims>
constexpr auto operator-(base_tensor<Tensor, T, Dims...> const& t) {
  return unary_operation([](auto const& c) { return -c; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator+(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         T1                                       scalar) {
  return unary_operation([scalar](auto const& c) { return c + scalar; }, lhs);
}
//------------------------------------------------------------------------------
/// matrix-matrix multiplication
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t M, size_t N, size_t O>
constexpr auto operator*(base_tensor<Tensor0, T0, M, N> const& lhs,
                         base_tensor<Tensor1, T1, N, O> const& rhs) {
  mat<promote_t<T0, T1>, M, O> product;
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < O; ++c) {
      product(r, c) = dot(lhs.template slice<0>(r), rhs.template slice<1>(c));
    }
  }
  return product;
}
//------------------------------------------------------------------------------
/// component-wise multiplication
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims,
          std::enable_if_t<(sizeof...(Dims) != 2), bool> = true>
constexpr auto operator%(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::multiplies<promote_t<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator/(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::divides<promote_t<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator+(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::plus<promote_t<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto operator*(base_tensor<Tensor, TensorT, Dims...> const& t,
                         real_or_complex_number auto const            scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto operator*(real_or_complex_number auto const            scalar,
                         base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [scalar](auto const& component) { return component * scalar; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto operator/(base_tensor<Tensor, TensorT, Dims...> const& t,
                         real_or_complex_number auto const            scalar) {
  return unary_operation(
      [scalar](auto const& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_or_complex_number TensorT,
          real_or_complex_number ScalarT, size_t... Dims>
constexpr auto operator/(ScalarT const                                scalar,
                         base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [scalar](auto const& component) { return scalar / component; }, t);
}
//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator-(base_tensor<Tensor0, T0, Dims...> const& lhs,
                         base_tensor<Tensor1, T1, Dims...> const& rhs) {
  return binary_operation(std::minus<promote_t<T0, T1>>{}, lhs, rhs);
}
//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t M, size_t N>
constexpr auto operator*(base_tensor<Tensor0, T0, M, N> const& lhs,
                         base_tensor<Tensor1, T1, N> const&    rhs) {
  vec<promote_t<T0, T1>, M> product;
  for (size_t i = 0; i < M; ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}
//------------------------------------------------------------------------------
/// vector-matrix-multiplication
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t M, size_t N>
constexpr auto operator*(base_tensor<Tensor0, T0, M> const&    lhs,
                         base_tensor<Tensor1, T1, M, N> const& rhs) {
  vec<promote_t<T0, T1>, N> product;
  for (size_t i = 0; i < N; ++i) {
    product(i) = dot(lhs, rhs.template slice<1>(i));
  }
  return product;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
