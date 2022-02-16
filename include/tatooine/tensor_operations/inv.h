#ifndef TATOOINE_TENSOR_OPERATIONS_INV_H
#define TATOOINE_TENSOR_OPERATIONS_INV_H
//==============================================================================
namespace tatooine {
//==============================================================================
/// invert symmetric matrix
/// A = [a,b]
///     [b,c]
template <typename Tensor, floating_point Real>
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
template <typename Tensor, floating_point Real>
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
template <typename Tensor, floating_point Real>
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
template <typename Tensor, floating_point Real>
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
template <typename Tensor, floating_point Real>
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
template <typename Tensor, floating_point Real>
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
}  // namespace tatooine
//==============================================================================
#endif
