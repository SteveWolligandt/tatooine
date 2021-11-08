#ifndef TATOOINE_TENSOR_OPERATIONS_SOLVE_H
#define TATOOINE_TENSOR_OPERATIONS_SOLVE_H
//==============================================================================
#include <tatooine/tensor_operations/determinant.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename TensorA, typename TensorB, typename Real, std::size_t K>
auto solve_direct(base_tensor<TensorA, Real, 2, 2> const& A,
           base_tensor<TensorB, Real, 2, K> const& B) {
  auto const p = 1 / (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
  auto       X = mat<Real, 2, K>{};
  for (std::size_t i = 0; i < K; ++i) {
    X(0, i) = -(A(0, 1) * B(1, i) - A(1, 1) * B(0, i)) * p;
    X(1, i) = (A(0, 0) * B(1, i) - A(1, 0) * B(0, i)) * p;
  }
  return X;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real>
auto solve_direct(base_tensor<TensorA, Real, 2, 2> const& A,
           base_tensor<TensorB, Real, 2> const&    b) {
  auto const p = 1 / (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
  return vec<Real, 2>{-(A(0, 1) * b(1) - A(1, 1) * b(0)) * p,
                      (A(0, 0) * b(1) - A(1, 0) * b(0)) * p};
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename Real, std::size_t N>
auto solve_cramer(base_tensor<TensorA, Real, N, N> const& A,
                  base_tensor<TensorB, Real, N> const&    b) {
  auto const det_inv = 1 / det(A);
  auto Y       = mat<Real, N, N>{A};
  auto tmp     = vec<Real, N>{};
  auto result  = vec<Real, N>{};
  for (size_t i = 0; i < N; ++i) {
    tmp       = Y.col(i);
    Y.col(i)  = b;
    result(i) = det(Y) * det_inv;
    Y.col(i)  = tmp;
  }
  return result;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename Real, size_t N>
auto solve_lu_lapack(base_tensor<TensorA, Real, N, N> const& A_base,
              base_tensor<TensorB, Real, N> const&    b_base) {
  auto                  A    = mat<Real, N, N>{A_base};
  auto                  b    = vec<Real, N>{b_base};
  auto                  ipiv = vec<std::int64_t, N>{};
  [[maybe_unused]] auto info = lapack::gesv(A, b, ipiv);
  return b;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t N, size_t K>
auto solve_lu_lapack(base_tensor<TensorA, Real, N, N> const& A_base,
              base_tensor<TensorB, Real, N, K> const& B_base) {
  auto                  A    = mat<Real, N, N>{A_base};
  auto                  B    = mat<Real, N, K>{B_base};
  auto                  ipiv = vec<int, N>{};
  [[maybe_unused]] auto info = lapack::gesv(A, B, ipiv);
  return B;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
auto solve_lu_lapack(tensor<Real> A, tensor<Real> B) {
  assert(A.rank() == 2);
  assert(A.dimension(0) == A.dimension(1));
  assert(A.dimension(0) == B.dimension(0));
  assert(B.rank() == 1 || B.rank() == 2);
  auto                  ipiv = tensor<std::int64_t>{A.dimension(0)};
  [[maybe_unused]] auto info = lapack::gesv(A, B, ipiv);
  return B;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N,
          size_t K>
auto solve_qr_lapack(base_tensor<TensorA, Real, M, N> const& A_base,
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
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N>
auto solve_qr_lapack(base_tensor<TensorA, Real, M, N> const& A_base,
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
template <typename Real>
auto solve_qr_lapack(tensor<Real> A, tensor<Real> B) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == B.dimension(0));
  auto const M   = A.dimension(0);
  auto const N   = A.dimension(1);
  auto const K   = (B.rank() == 1 ? 1 : B.dimension(1));
  auto       tau = tensor<Real>{min(M, N)};
  auto       X   = K > 1 ? tensor<Real>{N, K} : tensor<Real>{N};

  // Q * R = A
  lapack::geqrf(A, tau);
  // R * x = Q^T * B
  lapack::ormqr(A, B, tau, ::lapack::Side::Left, ::lapack::Op::Trans);
  // Use back-substitution using the upper right part of A
  lapack::trtrs(A, B, ::lapack::Uplo::Upper, ::lapack::Op::NoTrans,
                ::lapack::Diag::NonUnit);
  for_loop(
      [&](auto const i, auto const j) {
        if (B.rank() == 1) {
          X(i) = B(i);
        } else {
          X(i, j) = B(i, j);
        }
      },
      N, K);
  return X;
}
//------------------------------------------------------------------------------
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N,
          size_t K>
auto solve(base_tensor<TensorA, Real, M, N> const& A,
           base_tensor<TensorB, Real, M, K> const& B) {
  if constexpr (M == N) {
    return solve_lu_lapack(A, B);
  } else if constexpr (M > N) {
    return solve_qr_lapack(A, B);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, std::size_t K>
auto solve(base_tensor<TensorA, Real, 2, 2> const& A,
           base_tensor<TensorB, Real, 2, K> const& B) {
  return solve_direct(A, B);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real, size_t M, size_t N>
auto solve(base_tensor<TensorA, Real, M, N> const& A,
           base_tensor<TensorB, Real, M> const&    b) {
  if constexpr (M == N) {
    return solve_lu_lapack(A, b);
  } else if constexpr (M > N) {
    return solve_qr_lapack(A, b);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real>
auto solve(base_tensor<TensorA, Real, 2, 2> const& A,
           base_tensor<TensorB, Real, 2> const& b) {
  return solve_direct(A, b);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename TensorA, typename TensorB, typename Real>
auto solve(base_tensor<TensorA, Real, 3, 3> const& A,
           base_tensor<TensorB, Real, 3> const& b) {
  return solve_cramer(A, b);
}
//------------------------------------------------------------------------------
template <typename T>
auto solve(tensor<T> const& A, tensor<T> const& B) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(B.dimension(0) == A.dimension(0));
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  if (M == N) {
    return solve_lu_lapack(A, B);
  } else if (M > N) {
    return solve_qr_lapack(A, B);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
