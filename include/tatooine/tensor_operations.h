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
constexpr auto inv_sym(const base_tensor<Tensor, T, 2, 2>& A) {
  const auto& a = A(0, 0);
  const auto& b = A(1, 0);
  const auto& c = A(1, 1);
  const auto  d = 1 / (a * c - b * b);
  const auto  e = -b * d;
  return mat{{c * d, e}, {e, a * d}};
}
//------------------------------------------------------------------------------
/// invert matrix
/// A = [a,b]
///     [c,d]
template <typename Tensor, real_number T>
constexpr auto inv(const base_tensor<Tensor, T, 2, 2>& A) {
  const auto& b = A(0, 1);
  const auto& c = A(1, 0);
  if (std::abs(b - c) < 1e-10) { return inv_sym(A); }
  const auto& a = A(0, 0);
  const auto& d = A(1, 1);
  const auto  e = 1 / (a * d - b * c);
  return mat{{d * e, -b * e}, {-c * e, a * e}};
}
//------------------------------------------------------------------------------
/// invert symmetric matrix
/// A = [a,b,c]
///     [b,d,e]
///     [c,e,f]
template <typename Tensor, real_number T>
constexpr auto inv_sym(const base_tensor<Tensor, T, 3, 3>& A) {
  const auto& a = A(0, 0);
  const auto& b = A(1, 0);
  const auto& c = A(2, 0);
  const auto& d = A(1, 1);
  const auto& e = A(2, 1);
  const auto& f = A(2, 2);
  const auto  div =
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
constexpr auto inv(const base_tensor<Tensor, T, 3, 3>& A) {
  const auto& b = A(0, 1);
  const auto& c = A(0, 2);
  const auto& d = A(1, 0);
  const auto& g = A(2, 0);
  const auto& f = A(1, 2);
  const auto& h = A(2, 1);
  if (std::abs(b - d) < 1e-10 && std::abs(c - g) < 1e-10 &&
      std::abs(f - h) < 1e-10) {
    return inv_sym(A);
  }
  const auto& a = A(0, 0);
  const auto& e = A(1, 1);
  const auto& i = A(2, 2);
  const auto  div =
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
constexpr auto inv_sym(const base_tensor<Tensor, T, 4, 4>& A) {
  const auto& a = A(0, 0);
  const auto& b = A(1, 0);
  const auto& c = A(2, 0);
  const auto& d = A(3, 0);
  const auto& e = A(1, 1);
  const auto& f = A(2, 1);
  const auto& g = A(3, 1);
  const auto& h = A(2, 2);
  const auto& i = A(3, 2);
  const auto& j = A(3, 3);
  const auto  div =
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
constexpr auto inv(const base_tensor<Tensor, T, 4, 4>& A) {
  const auto& b = A(0, 1);
  const auto& c = A(0, 2);
  const auto& d = A(0, 3);
  const auto& e = A(1, 0);
  const auto& g = A(1, 2);
  const auto& h = A(1, 3);
  const auto& i = A(2, 0);
  const auto& j = A(2, 1);
  const auto& l = A(2, 3);
  const auto& m = A(3, 0);
  const auto& n = A(3, 1);
  const auto& o = A(3, 2);

  if (std::abs(b - e) < 1e-10 && std::abs(c - i) < 1e-10 &&
      std::abs(d - m) < 1e-10 && std::abs(g - j) < 1e-10 &&
      std::abs(h - n) < 1e-10 && std::abs(l - o) < 1e-10) {
    return inv_sym(A);
  }

  const auto& a = A(0, 0);
  const auto& f = A(1, 1);
  const auto& k = A(2, 2);
  const auto& p = A(3, 3);
  const auto  div =
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
template <typename Tensor0, typename Tensor1, real_number T0,
          real_number T1, size_t N>
constexpr auto cos_angle(const base_tensor<Tensor0, T0, N>& v0,
                         const base_tensor<Tensor1, T1, N>& v1) {
  return dot(normalize(v0), normalize(v1));
}

//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
auto angle(const base_tensor<Tensor0, T0, N>& v0,
           const base_tensor<Tensor1, T1, N>& v1) {
  return std::acos(cos_angle(v0, v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
auto min_angle(const base_tensor<Tensor0, T0, N>& v0,
               const base_tensor<Tensor1, T1, N>& v1) {
  return std::min(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, real_number T0, real_number T1,
          size_t N>
auto max_angle(const base_tensor<Tensor0, T0, N>& v0,
               const base_tensor<Tensor1, T1, N>& v1) {
  return std::max(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, real_number T0,
          real_number T1, real_number T2, size_t N>
constexpr auto cos_angle(const base_tensor<Tensor0, T0, N>& v0,
                         const base_tensor<Tensor1, T1, N>& v1,
                         const base_tensor<Tensor2, T2, N>& v2) {
  return cos_angle(v0 - v1, v2 - v1);
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, real_number T0,
          real_number T1, real_number T2, size_t N>
auto angle(const base_tensor<Tensor0, T0, N>& v0,
           const base_tensor<Tensor1, T1, N>& v1,
           const base_tensor<Tensor2, T2, N>& v2) {
  return std::acos(cos_angle(v0, v1, v2));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t... Dims>
constexpr auto min(const base_tensor<Tensor, T, Dims...>& t) {
  T m = std::numeric_limits<T>::max();
  t.for_indices([&](const auto... is) { m = std::min(m, t(is...)); });
  return m;
}

//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t... Dims>
constexpr auto max(const base_tensor<Tensor, T, Dims...>& t) {
  T m = -std::numeric_limits<T>::max();
  t.for_indices([&](const auto... is) { m = std::max(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto norm(const base_tensor<Tensor, T, N>& t, unsigned p = 2) -> T {
  T n = 0;
  for (size_t i = 0; i < N; ++i) { n += std::pow(t(i), p); }
  return std::pow(n, T(1) / T(p));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto norm_inf(const base_tensor<Tensor, T, N>& t) -> T {
  T norm = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < N; ++i) { norm = std::max(norm, std::abs(t(i))); }
  return norm;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto norm1(const base_tensor<Tensor, T, N>& t) {
  return sum(abs(t));
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto sqr_length(const base_tensor<Tensor, T, N>& t_in) {
  return dot(t_in, t_in);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto length(const base_tensor<Tensor, T, N>& t_in) -> T {
  return std::sqrt(sqr_length(t_in));
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto sqr_norm(const base_tensor<Tensor, T, M, N>& mat,
                        tag::frobenius_t) {
  T n = 0;
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < M; ++i) { n += std::abs(mat(i, j)); }
  }
  return n;
}
//------------------------------------------------------------------------------
/// Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm(const base_tensor<Tensor, T, M, N>& mat, tag::frobenius_t) {
  return std::sqrt(sqr_norm(mat, tag::frobenius));
}
//------------------------------------------------------------------------------
/// 1-norm of a MxN Tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm1(const base_tensor<Tensor, T, M, N>& mat) {
  T          max    = -std::numeric_limits<T>::max();
  const auto absmat = abs(mat);
  for (size_t i = 0; i < N; ++i) {
    max = std::max(max, sum(absmat.template slice<1>(i)));
  }
  return max;
}
//------------------------------------------------------------------------------
/// infinity-norm of a MxN tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm_inf(const base_tensor<Tensor, T, M, N>& mat) {
  T max = -std::numeric_limits<T>::max();
  for (size_t i = 0; i < M; ++i) {
    max = std::max(max, sum(abs(mat.template slice<0>(i))));
  }
  return max;
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto sqr_norm(const base_tensor<Tensor, T, M, N>& mat) {
  return sqr_norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, real_number T, size_t M, size_t N>
constexpr auto norm(const base_tensor<Tensor, T, M, N>& mat) {
  return norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number T, size_t N>
constexpr auto normalize(const base_tensor<Tensor, T, N>& t_in) -> vec<T, N> {
  const auto l = length(t_in);
  if (std::abs(l) < 1e-13) { return vec<T, N>::zeros(); }
  return t_in / l;
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_number T0, typename Tensor1, real_number T1,
          size_t N>
constexpr auto distance(const base_tensor<Tensor0, T0, N>& lhs,
                        const base_tensor<Tensor1, T1, N>& rhs) {
  return length(rhs - lhs);
}

//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename Tensor, real_or_complex_number T, size_t VecDim>
constexpr auto sum(const base_tensor<Tensor, T, VecDim>& v) {
  T s = 0;
  for (size_t i = 0; i < VecDim; ++i) { s += v(i); }
  return s;
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t N>
constexpr auto dot(const base_tensor<Tensor0, T0, N>& lhs,
                   const base_tensor<Tensor1, T1, N>& rhs) {
  promote_t<T0, T1> d = 0;
  for (size_t i = 0; i < N; ++i) { d += lhs(i) * rhs(i); }
  return d;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T>
constexpr auto det(const base_tensor<Tensor, T, 2, 2>& m) -> T {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T>
constexpr auto detAtA(const base_tensor<Tensor, T, 2, 2>& m) -> T {
  return m(0, 0) * m(0, 0) * m(1, 1) * m(1, 1) +
         m(0, 1) * m(0, 1) * m(1, 0) * m(1, 0) -
         2 * m(0, 0) * m(1, 0) * m(0, 1) * m(1, 1);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_or_complex_number T>
constexpr auto det(const base_tensor<Tensor, T, 3, 3>& m) -> T {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1>
constexpr auto cross(const base_tensor<Tensor0, T0, 3>& lhs,
                     const base_tensor<Tensor1, T1, 3>& rhs) {
  return vec<promote_t<T0, T1>, 3>{lhs(1) * rhs(2) - lhs(2) * rhs(1),
                                   lhs(2) * rhs(0) - lhs(0) * rhs(2),
                                   lhs(0) * rhs(1) - lhs(1) * rhs(0)};
}

//------------------------------------------------------------------------------
template <typename F, typename Tensor, real_or_complex_number T, size_t N>
constexpr auto unary_operation(F&& f, const base_tensor<Tensor, T, N>& t_in) {
  using TOut         = typename std::result_of<decltype(f)(T)>::type;
  vec<TOut, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, real_or_complex_number T, size_t M,
          size_t N>
constexpr auto unary_operation(F&&                                 f,
                               const base_tensor<Tensor, T, M, N>& t_in) {
  using TOut            = typename std::result_of<decltype(f)(T)>::type;
  mat<TOut, M, N> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
template <typename F, typename Tensor, real_or_complex_number T, size_t... Dims>
constexpr auto unary_operation(F&&                                    f,
                               const base_tensor<Tensor, T, Dims...>& t_in) {
  using TOut                  = typename std::result_of<decltype(f)(T)>::type;
  tensor<TOut, Dims...> t_out = t_in;
  t_out.unary_operation(std::forward<F>(f));
  return t_out;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename Tensor0, real_or_complex_number T0,
          typename Tensor1, real_or_complex_number T1, size_t N>
constexpr auto binary_operation(F&& f, const base_tensor<Tensor0, T0, N>& lhs,
                                const base_tensor<Tensor1, T1, N>& rhs) {
  using TOut         = typename std::result_of<decltype(f)(T0, T1)>::type;
  vec<TOut, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, real_or_complex_number T0,
          typename Tensor1, real_or_complex_number T1, size_t M, size_t N>
constexpr auto binary_operation(F&&                                   f,
                                const base_tensor<Tensor0, T0, M, N>& lhs,
                                const base_tensor<Tensor1, T1, M, N>& rhs) {
  using TOut            = typename std::result_of<decltype(f)(T0, T1)>::type;
  mat<TOut, M, N> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}
template <typename F, typename Tensor0, real_or_complex_number T0,
          typename Tensor1, real_or_complex_number T1, size_t... Dims>
constexpr auto binary_operation(F&&                                      f,
                                const base_tensor<Tensor0, T0, Dims...>& lhs,
                                const base_tensor<Tensor1, T1, Dims...>& rhs) {
  using TOut = typename std::result_of<decltype(f)(T0, T1)>::type;
  tensor<TOut, Dims...> t_out = lhs;
  t_out.binary_operation(std::forward<F>(f), rhs);
  return t_out;
}

//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number T, size_t... Dims>
constexpr auto operator-(const base_tensor<Tensor, T, Dims...>& t) {
  return unary_operation([](const auto& c) { return -c; }, t);
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator+(const base_tensor<Tensor0, T0, Dims...>& lhs,
                         T1                                       scalar) {
  return unary_operation([scalar](const auto& c) { return c + scalar; }, lhs);
}

//------------------------------------------------------------------------------
/// matrix-matrix multiplication
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t M, size_t N, size_t O>
constexpr auto operator*(const base_tensor<Tensor0, T0, M, N>& lhs,
                         const base_tensor<Tensor1, T1, N, O>& rhs) {
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
constexpr auto operator%(const base_tensor<Tensor0, T0, Dims...>& lhs,
                         const base_tensor<Tensor1, T1, Dims...>& rhs) {
  return binary_operation(std::multiplies<promote_t<T0, T1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator/(const base_tensor<Tensor0, T0, Dims...>& lhs,
                         const base_tensor<Tensor1, T1, Dims...>& rhs) {
  return binary_operation(std::divides<promote_t<T0, T1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator+(const base_tensor<Tensor0, T0, Dims...>& lhs,
                         const base_tensor<Tensor1, T1, Dims...>& rhs) {
  return binary_operation(std::plus<promote_t<T0, T1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto operator*(const base_tensor<Tensor, TensorT, Dims...>& t,
                         real_or_complex_number auto const            scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto operator*(real_or_complex_number auto const            scalar,
                         const base_tensor<Tensor, TensorT, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}

//------------------------------------------------------------------------------
template <typename Tensor, real_or_complex_number TensorT, size_t... Dims>
constexpr auto operator/(const base_tensor<Tensor, TensorT, Dims...>& t,
                         real_or_complex_number auto const            scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_or_complex_number TensorT,
          real_or_complex_number ScalarT, size_t... Dims>
constexpr auto operator/(const ScalarT                                scalar,
                         const base_tensor<Tensor, TensorT, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return scalar / component; }, t);
}

//------------------------------------------------------------------------------
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t... Dims>
constexpr auto operator-(const base_tensor<Tensor0, T0, Dims...>& lhs,
                         const base_tensor<Tensor1, T1, Dims...>& rhs) {
  return binary_operation(std::minus<promote_t<T0, T1>>{}, lhs, rhs);
}

//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename Tensor0, real_or_complex_number T0, typename Tensor1,
          real_or_complex_number T1, size_t M, size_t N>
constexpr auto operator*(const base_tensor<Tensor0, T0, M, N>& lhs,
                         const base_tensor<Tensor1, T1, N>&    rhs) {
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
constexpr auto operator*(const base_tensor<Tensor0, T0, M>&    lhs,
                         const base_tensor<Tensor1, T1, M, N>& rhs) {
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
