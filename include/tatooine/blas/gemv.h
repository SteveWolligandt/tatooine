
#ifndef TATOOINE_BLAS_GEMV_H
#define TATOOINE_BLAS_GEMV_H
//==============================================================================
#include <tatooine/blas/base.h>
//==============================================================================
namespace tatooine::blas {
//==============================================================================
/// \defgroup blas_gemv GEMV
/// \brief General Matrix-Vector-Multiplication
/// \ingroup blas2
///
///  **GEMV**  performs one of the matrix-vector operations
///
/// \f[\vy = \alpha\cdot\mA\cdot\vx + \beta*\vy\ \ \text{ or }\ \ \vy = \alpha\cdot\mA^\top\cdot\vx + \beta*\vy\f]
///
///  where \f$\alpha\f$ and \f$\beta\f$ are scalars, \f$\vx\f$ and \f$\vy\f$ are
///  vectors and \f$\mA\f$ is an \f$m \times n\f$ matrix.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html'>BLAS
/// documentation</a>
/// \{
//==============================================================================
using ::blas::gemv;
//==============================================================================
/// See \ref blas_gemv.
template <typename Real>
auto gemv(blas::Op trans, Real const alpha, tensor<Real> const& A,
          tensor<Real> const& x, Real const beta, tensor<Real>& y) {
  assert(A.rank() == 2);
  assert(x.rank() == 1);
  assert(y.rank() == 1);
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  assert(N == x.dimension(0));
  assert(y.dimension(0) == M);

  return gemv(Layout::ColMajor, trans, M, N, alpha, A.data_ptr(), M,
              x.data_ptr(), 1, beta, y.data_ptr(), 1);
}
//------------------------------------------------------------------------------
/// See \ref blas_gemv.
template <typename Real>
auto gemv(Real const alpha, tensor<Real> const& A,
          tensor<Real> const& x, Real const beta, tensor<Real>& y) {
  return gemv(Op::NoTrans, alpha, A, x, beta, y);
}
/// \}
//==============================================================================
}  // namespace tatooine::blas
//==============================================================================
#endif
