#ifndef TATOOINE_TENSOR_OPERATIONS_SOLVE_H
#define TATOOINE_TENSOR_OPERATIONS_SOLVE_H
//==============================================================================
namespace tatooine {
//==============================================================================
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
template <typename Real>
auto solve_lu(tensor<Real> A, tensor<Real> B) {
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
template <typename Real>
auto solve_qr(tensor<Real> A, tensor<Real> B) {
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
    return solve_lu(A, B);
  } else if constexpr (M > N) {
    return solve_qr(A, B);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
template <typename T>
auto solve(tensor<T> const& A, tensor<T> const& B) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(B.dimension(0) == A.dimension(0));
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  if (M == N) {
    return solve_lu(A, B);
  } else if (M > N) {
    return solve_qr(A, B);
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
