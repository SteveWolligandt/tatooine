#ifndef TATOOINE_TENSOR_OPERATIONS_SOLVE_H
#define TATOOINE_TENSOR_OPERATIONS_SOLVE_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
#include <tatooine/tensor_operations/determinant.h>
#include <tatooine/utility.h>

#include <cstdint>
//==============================================================================
namespace tatooine {
//==============================================================================
auto copy_or_keep_if_rvalue_tensor_solve(dynamic_tensor auto&& x) -> decltype(auto) {
  return tensor<tensor_value_type<decltype(x)>>{std::forward<decltype(x)>(x)};
}
template <typename T>
auto copy_or_keep_if_rvalue_tensor_solve(tensor<T>&& x) -> decltype(auto) {
  return std::move(x);
}
template <typename T, std::size_t... Dims>
auto copy_or_keep_if_rvalue_tensor_solve(tensor<T, Dims...>&& x) -> decltype(auto) {
  return std::move(x);
}
template <typename T>
auto copy_or_keep_if_rvalue_tensor_solve(tensor<T> const& x) {
  return tensor<T>{x};
}
template <typename T, std::size_t... Dims>
auto copy_or_keep_if_rvalue_tensor_solve(tensor<T, Dims...> const& x) {
  return tensor<T, Dims...>{x};
}
template <fixed_size_quadratic_mat<2> MatA, static_mat MatB>
requires(tensor_dimension<MatB, 0> == 2)
auto constexpr solve_direct(MatA&& A, MatB&& B)
    -> std::optional<
        mat<common_type<tensor_value_type<MatA>, tensor_value_type<MatB>>, 2,
            tensor_dimension<MatB, 1>>> {
      using out_value_type =
          common_type<tensor_value_type<MatA>, tensor_value_type<MatB>>;
      auto constexpr K = tensor_dimension<MatB, 1>;
      auto const div   = (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
      if (div == 0) {
        return std::nullopt;
      }
      auto const p = 1 / div;
      auto       X = mat<out_value_type, 2, K>{};
      for (std::size_t i = 0; i < K; ++i) {
        X(0, i) = -(A(0, 1) * B(1, i) - A(1, 1) * B(0, i)) * p;
        X(1, i) = (A(0, 0) * B(1, i) - A(1, 0) * B(0, i)) * p;
      }
      return X;
    }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <fixed_size_mat<2, 2> MatA, fixed_size_vec<2> VecB>
auto solve_direct(MatA&& A, VecB&& b) -> std::optional<
    vec<common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>, 2>> {
  using out_value_type =
      common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>;
  auto const div = (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
  if (div == 0) {
    return std::nullopt;
  }
  auto const p = 1 / div;
  return vec<out_value_type, 2>{-(A(0, 1) * b(1) - A(1, 1) * b(0)) * p,
                                (A(0, 0) * b(1) - A(1, 0) * b(0)) * p};
}
//------------------------------------------------------------------------------
template <static_quadratic_mat MatA, static_vec VecB>
requires(tensor_dimension<MatA, 0> ==
         tensor_dimension<VecB, 0>)
auto solve_cramer(MatA const& A, VecB const& b) -> std::optional<
    vec<common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>,
        tensor_dimension<MatA, 1>>> {
  static constexpr auto N = tensor_dimension<MatA, 1>;
  using out_value_type =
      common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>;
  auto const det_inv = 1 / det(A);
  auto       Y       = mat<out_value_type, N, N>{A};
  auto       tmp     = vec<out_value_type, N>{};
  auto       result  = vec<out_value_type, N>{};
  for (std::size_t i = 0; i < N; ++i) {
    tmp       = Y.col(i);
    Y.col(i)  = b;
    result(i) = det(Y) * det_inv;
    Y.col(i)  = tmp;
  }
  return result;
}
//------------------------------------------------------------------------------
template <static_quadratic_mat MatA, static_vec VecB>
requires(tensor_dimension<MatA, 1> ==
         tensor_dimension<VecB, 0>)
auto solve_lu_lapack(MatA& A_, VecB&& b_)
    -> std::optional<
        tensor<common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>>> {
  using out_value_type =
      common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>;
  static constexpr auto N = tensor_dimension<MatA, 1>;
  auto                  A    = mat<out_value_type, N, N>{A_};
  auto                  b    = vec<out_value_type, N>{b_};
  auto                  ipiv = vec<std::int64_t, N>{};
  [[maybe_unused]] auto info = lapack::gesv(A, b, ipiv);
  if (info != 0) {
    return std::nullopt;
  }
  return b;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_quadratic_mat MatA, static_mat MatB>
requires(tensor_dimension<MatA, 1> ==
         tensor_dimension<MatB, 0>)
auto solve_lu_lapack(MatA& A_, MatB& B_)
    -> std::optional<
        tensor<common_type<tensor_value_type<MatA>, tensor_value_type<MatB>>>> {
  using out_value_type =
      common_type<tensor_value_type<MatA>, tensor_value_type<MatB>>;
  static constexpr auto N = tensor_dimension<MatA, 1>; 
  static constexpr auto K = tensor_dimension<MatB, 1>; 
  auto                  A    = mat<out_value_type, N, N>{A_};
  auto                  B    = mat<out_value_type, N, K>{B_};
  auto                  ipiv = vec<std::int64_t, N>{};
  [[maybe_unused]] auto info = lapack::gesv(A, B, ipiv);
  if (info != 0) {
    return std::nullopt;
  }
  return B;
}
//------------------------------------------------------------------------------
template <static_mat MatA, static_mat MatB>
requires(tensor_dimension<MatA, 0> == tensor_dimension<MatB, 0>)
auto solve_qr_lapack(MatA&& A_, MatB&& B_)
    -> std::optional<
        mat<common_type<tensor_value_type<MatA>, tensor_value_type<MatB>>,
            tensor_dimension<MatA, 1>, tensor_dimension<MatB, 1>>> {
      using out_value_type =
          common_type<tensor_value_type<MatA>, tensor_value_type<MatB>>;
      static auto constexpr M = tensor_dimension<MatA, 0>;
      static auto constexpr N = tensor_dimension<MatA, 1>;
      static auto constexpr K = tensor_dimension<MatB, 1>;

      auto A   = mat<out_value_type, M, N>{A_};
      auto B   = mat<out_value_type, M, K>{B_};
      auto tau = vec<out_value_type, (M < N ? M : N)>{};
      auto X   = mat<out_value_type, N, K>{};

      // Q * R = A
      if (lapack::geqrf(A, tau) != 0) {
        return std::nullopt;
      }
      // R * x = Q^T * B
      if (lapack::ormqr(A, B, tau, ::lapack::Side::Left, ::lapack::Op::Trans) !=
          0) {
        return std::nullopt;
      }
      // Use back-substitution using the upper right part of A
      if (lapack::trtrs(A, B, ::lapack::Uplo::Upper, ::lapack::Op::NoTrans,
                        ::lapack::Diag::NonUnit)) {
        return std::nullopt;
      }
      for_loop([&, i = std::size_t(0)](
                   auto const... is) mutable { X(is...) = B(is...); },
               N, K);
      return X;
    }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_mat MatA, static_vec VecB>
requires (tensor_dimension<MatA, 0> == tensor_dimension<VecB, 0>)
auto solve_qr_lapack(MatA&& A_, VecB&& b_) -> std::optional<
    vec<common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>,
        (tensor_dimension<MatA, 0> < tensor_dimension<MatA, 1>
             ? tensor_dimension<MatA, 0>
             : tensor_dimension<MatA, 1>)>> {
  using out_value_type =
      common_type<tensor_value_type<MatA>, tensor_value_type<VecB>>;
  static auto constexpr M = tensor_dimension<MatA, 0>;
  static auto constexpr N = tensor_dimension<MatA, 1>;
  auto A   = mat<out_value_type, M, N>{A_};
  auto b   = vec<out_value_type, M>{b_};
  auto tau = vec<out_value_type, (M < N ? M : N)>{};

  // Q * R = A
  if (lapack::geqrf(A, tau) != 0) {
    return std::nullopt;
  }
  // R * x = Q^T * b
  if (lapack::ormqr(A, b, tau, ::lapack::Side::Left, ::lapack::Op::Trans) != 0) {
    return std::nullopt;
  }
  // Use back-substitution using the upper right part of A
  if (lapack::trtrs(A, b, ::lapack::Uplo::Upper, ::lapack::Op::NoTrans,
                ::lapack::Diag::NonUnit) != 0) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < tau.dimension(0); ++i) {
    tau(i) = b(i);
  }
  return tau;
}
//------------------------------------------------------------------------------
template <static_mat MatA, static_mat MatB>
requires (tensor_dimension<MatA, 0> == tensor_dimension<MatB, 0>)
auto solve(MatA&& A, MatB&& B) {
  static auto constexpr M = tensor_dimension<MatA, 0>;
  static auto constexpr N = tensor_dimension<MatA, 1>;
  static auto constexpr K = tensor_dimension<MatB, 1>;
  if constexpr (M == 2 && N == 2 && K >= M) {
    return solve_direct(std::forward<MatA>(A), std::forward<MatB>(B));
  } else if constexpr (M == N) {
    return solve_lu_lapack(std::forward<MatA>(A), std::forward<MatB>(B));
  } else if constexpr (M > N) {
    return solve_qr_lapack(std::forward<MatA>(A), std::forward<MatB>(B));
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
//------------------------------------------------------------------------------
template <dynamic_tensor TensorA, dynamic_tensor TensorB>
auto solve_lu_lapack(TensorA&& A_, TensorB&& B_) -> std::optional<tensor<
    common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>>> {
  assert(A_.rank() == 2);
  assert(A_.dimension(0) == A_.dimension(1));
  assert(A_.dimension(0) == B_.dimension(0));
  assert(B_.rank() == 1 || B_.rank() == 2);
  auto ipiv   = tensor<std::int64_t>::zeros(A_.dimension(0));
  if constexpr (same_as<tensor_value_type<TensorA>,
                        tensor_value_type<TensorB>>) {
    decltype(auto) A = copy_or_keep_if_rvalue_tensor_solve(A_);
    decltype(auto) B = copy_or_keep_if_rvalue_tensor_solve(B_);
    if (lapack::gesv(A, B, ipiv) != 0) {
      return std::nullopt;
    }
    return std::forward<decltype(B)>(B);
  } else {
    using out_value_type =
        common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>;
    auto A = tensor<out_value_type>{A_};
    auto B = tensor<out_value_type>{B_};
    if (lapack::gesv(A, B, ipiv) != 0) {
      return std::nullopt;
    }
    return B;
  }
}
//------------------------------------------------------------------------------
template <dynamic_tensor TensorA, dynamic_tensor TensorB>
auto solve_qr_lapack(TensorA&& A_, TensorB&& B_)
    -> std::optional<tensor<
        common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>>> {
  using out_value_type =
      common_type<tensor_value_type<TensorA>, tensor_value_type<TensorB>>;
  assert(A_.rank() == 2);
  assert(B_.rank() == 1 || B_.rank() == 2);
  assert(A_.dimension(0) == B_.dimension(0));
  auto A = copy_or_keep_if_rvalue_tensor_solve(A_);
  auto B = copy_or_keep_if_rvalue_tensor_solve(B_);
  auto const M   = A.dimension(0);
  auto const N   = A.dimension(1);
  auto const K   = (B.rank() == 1 ? 1 : B.dimension(1));
  auto       tau = tensor<out_value_type>{min(M, N)};
  auto       X   = K > 1 ? tensor<out_value_type>{N, K} : tensor<out_value_type>{N};

  // Q * R = A
  if (lapack::geqrf(A, tau) != 0) {
    return std::nullopt;
  }
  // R * x = Q^T * B
  if (lapack::ormqr(A, B, tau, ::lapack::Side::Left, ::lapack::Op::Trans) !=
      0) {
    return std::nullopt;
  }
  // Use back-substitution using the upper right part of A
  if (lapack::trtrs(A, B, ::lapack::Uplo::Upper, ::lapack::Op::NoTrans,
                    ::lapack::Diag::NonUnit) != 0) {
    return std::nullopt;
  }
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
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_mat MatA, static_vec VecB>
requires (tensor_dimension<MatA, 0> == tensor_dimension<VecB, 0>)
auto solve(MatA&& A, VecB&& b) {
  static auto constexpr M = tensor_dimension<MatA, 0>;
  static auto constexpr N = tensor_dimension<MatA, 1>;
  if constexpr (M == 2 && N == 2) {
    return solve_direct(std::forward<MatA>(A), std::forward<VecB>(b));
  } else if constexpr (M == 3 && N == 3) {
    return solve_cramer(std::forward<MatA>(A), std::forward<VecB>(b));
  } else if constexpr (M == N) {
    return solve_lu_lapack(std::forward<MatA>(A), std::forward<VecB>(b));
  } else if constexpr (M > N) {
    return solve_qr_lapack(std::forward<MatA>(A), std::forward<VecB>(b));
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
//==============================================================================
/// Computes the solution to a system of linear equations \(A X = B\), where A
/// is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
///
/// The diagonal pivoting method is used to factor A as \(A = U D U^T\) if uplo
/// = Upper, or \(A = L D L^T\) if uplo = Lower, where U (or L) is a product of
/// permutation and unit upper (lower) triangular matrices, and D is symmetric
/// and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The factored form
/// of A is then used to solve the system of equations \(A X = B\).
auto solve_symmetric_lapack(dynamic_tensor auto&& A_, dynamic_tensor auto&& B_,
                            lapack::Uplo uplo = lapack::Uplo::Lower)
    -> std::optional<tensor<common_type<tensor_value_type<decltype(A_)>,
                                        tensor_value_type<decltype(B_)>>>> {
  // assert A is quadratic matrix
  assert(A_.rank() == 2);
  assert(A_.dimension(0) == A_.dimension(1));

  // assert B is vector or matrix
  assert(B_.rank() == 1 || B_.rank() == 2);
  // assert B dimensions are correct
  assert(B_.dimension(0) == A_.dimension(0));
  if constexpr (same_as<tensor_value_type<decltype(A_)>,
                        tensor_value_type<decltype(B_)>>) {
    decltype(auto) A = copy_or_keep_if_rvalue_tensor_solve(A_);
    decltype(auto) B = copy_or_keep_if_rvalue_tensor_solve(B_);

    auto const info = lapack::sysv(A, B, uplo);
    if (info > 0) {
      std::cerr
          << "[lapack::sysv]\n  D(" << info << ", " << info
          << ") is exactly zero. The factorization has been completed, but the "
             "block diagonal matrix D is exactly singular, so the solution "
             "could "
             "not be computed.\n";
    } else if (info < 0) {
      std::cerr << "[lapack::sysv]\n  Parameter " << -info << " is wrong.\n";
    }
    if (info != 0) {
      return std::nullopt;
    }
    return std::forward<decltype(B)>(B);
  } else {
    using T = common_type<tensor_value_type<decltype(A_)>,
                          tensor_value_type<decltype(B_)>>;
    auto A = tensor<T>{A_};
    auto B = tensor<T>{B_};

    auto const info = lapack::sysv(A, B, uplo);
    if (info > 0) {
      std::cerr
          << "[lapack::sysv]\n  D(" << info << ", " << info
          << ") is exactly zero. The factorization has been completed, but the "
             "block diagonal matrix D is exactly singular, so the solution "
             "could "
             "not be computed.\n";
    } else if (info < 0) {
      std::cerr << "[lapack::sysv]\n  Parameter " << -info << " is wrong.\n";
    }
    if (info != 0) {
      return std::nullopt;
    }
    return B;
  }
}
//==============================================================================
/// Computes the solution to a system of linear equations \(A X = B\), where A
/// is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
///
/// Aasen's algorithm is used to factor A as \(A = U T U^T\) if uplo = Upper, or
/// \(A = L T L^T\) if uplo = Lower, where U (or L) is a product of permutation
/// and unit upper (lower) triangular matrices, and T is symmetric tridiagonal.
/// The factored form of A is then used to solve the system of equations \(A X =
/// B\).
auto solve_symmetric_lapack_aa(dynamic_tensor auto&& A_, dynamic_tensor auto&& B_,
                            lapack::Uplo uplo = lapack::Uplo::Lower)
    -> std::optional<tensor<common_type<tensor_value_type<decltype(A_)>,
                                        tensor_value_type<decltype(B_)>>>> {
  // assert A is quadratic matrix
  assert(A_.rank() == 2);
  assert(A_.dimension(0) == A_.dimension(1));

  // assert B is vector or matrix
  assert(B_.rank() == 1 || B_.rank() == 2);
  // assert B dimensions are correct
  assert(B_.dimension(0) == A_.dimension(0));
  if constexpr (same_as<tensor_value_type<decltype(A_)>,
                        tensor_value_type<decltype(B_)>>) {
    decltype(auto) A = copy_or_keep_if_rvalue_tensor_solve(A_);
    decltype(auto) B = copy_or_keep_if_rvalue_tensor_solve(B_);

    auto const info = lapack::sysv_aa(A, B, uplo);
    if (info > 0) {
      std::cerr
          << "[lapack::sysv_aa]\n  D(" << info << ", " << info
          << ") is exactly zero. The factorization has been completed, but the "
             "block diagonal matrix D is exactly singular, so the solution "
             "could "
             "not be computed.\n";
    } else if (info < 0) {
      std::cerr << "[lapack::sysv_aa]\n  Parameter " << -info << " is wrong.\n";
    }
    if (info != 0) {
      return std::nullopt;
    }
    return std::forward<decltype(B)>(B);
  } else {
    using T = common_type<tensor_value_type<decltype(A_)>,
                          tensor_value_type<decltype(B_)>>;
    auto A = tensor<T>{A_};
    auto B = tensor<T>{B_};

    auto const info = lapack::sysv_aa(A, B, uplo);
    if (info > 0) {
      std::cerr
          << "[lapack::sysv_aa]\n  D(" << info << ", " << info
          << ") is exactly zero. The factorization has been completed, but the "
             "block diagonal matrix D is exactly singular, so the solution "
             "could "
             "not be computed.\n";
    } else if (info < 0) {
      std::cerr << "[lapack::sysv_aa]\n  Parameter " << -info << " is wrong.\n";
    }
    if (info != 0) {
      return std::nullopt;
    }
    return B;
  }
}
//==============================================================================
/// Computes the solution to a system of linear equations.
///
///\[ A X = B, \]
///
/// where A is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
///
/// The bounded Bunch-Kaufman (rook) diagonal pivoting method is used to factor
/// A as \(A = P U D U^T P^T\) if uplo = Upper, or \(A = P L D L^T P^T\) if uplo
/// = Lower, where U (or L) is unit upper (or lower) triangular matrix, \(U^T\)
/// (or
///    \(L^T\)) is the transpose of U (or L), P is a permutation matrix, \(P^T\)
/// is the transpose of P, and D is symmetric and block diagonal with 1-by-1 and
/// 2-by-2 diagonal blocks.
///
/// lapack::sytrf_rk is called to compute the factorization of a symmetric
/// matrix. The factored form of A is then used to solve the system of equations
/// \(A X = B\) by calling lapack::sytrs_rk.
auto solve_symmetric_lapack_rk(dynamic_tensor auto&& A_,
                               dynamic_tensor auto&& B_,
                               lapack::Uplo          uplo = lapack::Uplo::Lower)
    -> std::optional<tensor<common_type<tensor_value_type<decltype(A_)>,
                                        tensor_value_type<decltype(B_)>>>> {
  // assert A is quadratic matrix
  assert(A_.rank() == 2);
  assert(A_.dimension(0) == A_.dimension(1));

  // assert B is vector or matrix
  assert(B_.rank() == 1 || B_.rank() == 2);
  // assert B dimensions are correct
  assert(B_.dimension(0) == A_.dimension(0));
  if constexpr (same_as<tensor_value_type<decltype(A_)>,
                        tensor_value_type<decltype(B_)>>) {
    decltype(auto) A = copy_or_keep_if_rvalue_tensor_solve(A_);
    decltype(auto) B = copy_or_keep_if_rvalue_tensor_solve(B_);

    auto const info = lapack::sysv_rk(A, B, uplo);
    if (info > 0) {
      std::cerr
          << "[lapack::sysv_rk]\n  D(" << info << ", " << info
          << ") is exactly zero. The factorization has been completed, but the "
             "block diagonal matrix D is exactly singular, so the solution "
             "could "
             "not be computed.\n";
    } else if (info < 0) {
      std::cerr << "[lapack::sysv_rk]\n  Parameter " << -info << " is wrong.\n";
    }
    if (info != 0) {
      return std::nullopt;
    }
    return std::forward<decltype(B)>(B);
  } else {
    using T = common_type<tensor_value_type<decltype(A_)>,
                          tensor_value_type<decltype(B_)>>;
    auto A = tensor<T>{A_};
    auto B = tensor<T>{B_};

    auto const info = lapack::sysv_rk(A, B, uplo);
    if (info > 0) {
      std::cerr
          << "[lapack::sysv_rk]\n  D(" << info << ", " << info
          << ") is exactly zero. The factorization has been completed, but the "
             "block diagonal matrix D is exactly singular, so the solution "
             "could "
             "not be computed.\n";
    } else if (info < 0) {
      std::cerr << "[lapack::sysv_rk]\n  Parameter " << -info << " is wrong.\n";
    }
    if (info != 0) {
      return std::nullopt;
    }
    return B;
  }
}
//------------------------------------------------------------------------------
auto solve_symmetric(dynamic_tensor auto&& A, dynamic_tensor auto&& B) {
  return solve_symmetric_lapack(std::forward<decltype(A)>(A),
                                std::forward<decltype(B)>(B));
}
//------------------------------------------------------------------------------
auto solve(dynamic_tensor auto && A, dynamic_tensor auto && B) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(B.dimension(0) == A.dimension(0));
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  if (M == N) {
    return solve_lu_lapack(std::forward<decltype(A)>(A),
                           std::forward<decltype(B)>(B));
  } else if (M > N) {
    return solve_qr_lapack(std::forward<decltype(A)>(A),
                           std::forward<decltype(B)>(B));
  } else {
    throw std::runtime_error{"System is under-determined."};
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
