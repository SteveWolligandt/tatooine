#ifndef TATOOINE_LAPACK_H
#define TATOOINE_LAPACK_H
//==============================================================================
#if TATOOINE_INCLUDE_MKL_LAPACKE
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
#include <tatooine/math.h>
#include <tatooine/tensor.h>
#include <tatooine/lapack_job.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
static constexpr auto including_mkl_lapacke() {
#if TATOOINE_INCLUDE_MKL_LAPACKE
  return true;
#else
  return false;
#endif
}
//==============================================================================
template <typename T, size_t M, size_t N>
auto getrf(tensor<T, M, N>&& A) {
  vec<int, tatooine::min(M, N)> p;
  if constexpr (is_same<double, T>) {
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (is_same<float, T>) {
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (is_same<std::complex<double>, T>) {
    LAPACKE_zgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (is_same<std::complex<float>, T>) {
    LAPACKE_cgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else {
    throw std::runtime_error{"[tatooine::lapack::getrf] - type not accepted"};
  }
  return A;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto getrf(tensor<T, M, N>& A) {
  vec<int, tatooine::min(M, N)> p;
  if constexpr (is_same<double, T>) {
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (is_same<float, T>) {
    LAPACKE_sgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (is_same<std::complex<double>, T>) {
    LAPACKE_zgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  } else if constexpr (is_same<std::complex<float>, T>) {
    LAPACKE_cgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  }
  return A;
}
//==============================================================================
template <typename T, size_t N>
auto gesv(tensor<T, N, N> A, tensor<T, N> b) {
  vec<int, N> ipiv;
  int         nrhs = 1;
  if constexpr (is_same<double, T>) {
    LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, A.data_ptr(), N, ipiv.data_ptr(),
                  b.data_ptr(), N);
  } else if constexpr (is_same<float, T>) {
    LAPACKE_sgesv(LAPACK_COL_MAJOR, N, nrhs, A.data_ptr(), N, ipiv.data(),
                  b.data_ptr(), N);
  } else {
    throw std::runtime_error{"[tatooine::lapack::gesv] - type not accepted"};
  }
  return b;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N, size_t NRHS>
auto gesv(tensor<T, N, N> A, tensor<T, N, NRHS> B) {
  std::array<int, N> ipiv;
  if constexpr (is_same<double, T>) {
    LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, A.data_ptr(), N, ipiv.data(),
                  B.data_ptr(), N);
  } else if constexpr (is_same<float, T>) {
    LAPACKE_sgesv(LAPACK_COL_MAJOR, N, NRHS, A.data_ptr(), N, ipiv.data_ptr(),
                  B.data_ptr(), N);
  } else {
    throw std::runtime_error{"[tatooine::lapack::gesv] - type not accepted"};
  }
  return B;
}
//==============================================================================
template <typename T, size_t M, size_t N>
auto lange(const tensor<T, M, N>& A, const char norm) {
  if constexpr (is_same<double, T>) {
    return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else if constexpr (is_same<float, T>) {
    return LAPACKE_slange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else if constexpr (is_same<std::complex<double>, T>) {
    return LAPACKE_zlange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else if constexpr (is_same<std::complex<float>, T>) {
    return LAPACKE_clange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
  } else {
    throw std::runtime_error{"[tatooine::lapack::lange] - type not accepted"};
  }
}
//==============================================================================
/// Estimates the reciprocal of the condition number of a general matrix A.
/// http://www.netlib.org/lapack/explore-html/d7/db5/lapacke__dgecon_8c_a7c007823b949b0b118acf7e0235a6fc5.html
/// https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga188b8d30443d14b1a3f7f8331d87ae60.html
template <typename T, size_t N>
auto gecon(tensor<T, N, N>&& A) {
  T              rcond = 0;
  constexpr char p     = '1';
  const auto     n     = lange(A, p);
  getrf(A);
  const auto info = [&] {
    if constexpr (is_same<double, T>) {
      return LAPACKE_dgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else if constexpr (is_same<float, T>) {
      return LAPACKE_sgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else if constexpr (is_same<std::complex<float>, T>) {
      return LAPACKE_cgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else if constexpr (is_same<std::complex<double>, T>) {
      return LAPACKE_zgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N, n, &rcond);
    } else {
      throw std::runtime_error{"[tatooine::lapack::gecon] - type not accepted"};
    }
  }();
  if (info < 0) {
    throw std::runtime_error{"[tatooine::lapack::gecon] - " +
                             std::to_string(-info) + "-th argument is invalid"};
  }
  return rcond;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Estimates the reciprocal of the condition number of a general matrix A.
/// http://www.netlib.org/lapack/explore-html/d7/db5/lapacke__dgecon_8c_a7c007823b949b0b118acf7e0235a6fc5.html
/// https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga188b8d30443d14b1a3f7f8331d87ae60.html
template <typename T, size_t N>
auto gecon(tensor<T, N, N>& A) {
  T              rcond = 0;
  constexpr char p     = 'I';
  getrf(A);
  const auto info = [&] {
    if constexpr (is_same<double, T>) {
      return LAPACKE_dgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else if constexpr (is_same<float, T>) {
      return LAPACKE_sgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else if constexpr (is_same<std::complex<float>, T>) {
      return LAPACKE_cgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else if constexpr (is_same<std::complex<double>, T>) {
      return LAPACKE_zgecon(LAPACK_COL_MAJOR, p, N, A.data_ptr(), N,
                            lange(A, p), &rcond);
    } else {
      throw std::runtime_error{"[tatooine::lapack::gecon] - type not accepted"};
    }
  }();
  if (info < 0) {
    throw std::runtime_error{"[tatooine::lapack::gecon] - " +
                             std::to_string(-info) + "-th argument is invalid"};
  }
  return rcond;
}
//==============================================================================
/// Estimates the reciprocal of the condition number of a general matrix A.
/// http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
/// http://www.netlib.org/lapack/explore-html/d0/dee/lapacke__dgesvd_8c_af31b3cb47f7cc3b9f6541303a2968c9f.html
template <typename T, size_t M, size_t N, typename JOBU, typename JOBVT>
auto gesvd(tensor<T, M, N>&& A, JOBU, JOBVT) {
  static_assert(
      !is_same<JOBU, job::O_t> || !is_same<JOBVT, job::O_t>,
      "either jobu or jobvt must not be O");
  vec<T, tatooine::min(M, N)> s;
  constexpr char              jobu  = job::to_char<JOBU>();
  constexpr char              jobvt = job::to_char<JOBVT>();
  auto                        U     = [] {
    if constexpr (is_same<job::A_t, JOBU>) {
      return mat<T, M, M>{};
    } else if constexpr (is_same<job::S_t, JOBU>) {
      return mat<T, M, tatooine::min(M, N)>{};
    } else {
      return nullptr;
    }
  }();

  auto VT = [] {
    if constexpr (is_same<job::A_t, JOBVT>) {
      return mat<T, N, N>{};
    } else if constexpr (is_same<job::S_t, JOBVT>) {
      return mat<T, tatooine::min(M, N), N>{};
    } else {
      return nullptr;
    }
  }();
  constexpr auto ldu = [&U] {
    if constexpr (is_same<job::A_t, JOBU> ||
                  is_same<job::S_t, JOBU>) {
      return U.dimension(0);
    } else {
      return 1;
    }
  }();
  constexpr auto ldvt = [&VT] {
    if constexpr (is_same<job::A_t, JOBVT> ||
                  is_same<job::S_t, JOBVT>) {
      return VT.dimension(0);
    } else {
      return 1;
    }
  }();
  T* U_ptr = [&U] {
    if constexpr (is_same<job::A_t, JOBU> ||
                  is_same<job::S_t, JOBU>) {
      return U.data_ptr();
    } else {
      return nullptr;
    }
  }();
  T* VT_ptr = [&VT] {
    if constexpr (is_same<job::A_t, JOBVT> ||
                  is_same<job::S_t, JOBVT>) {
      return VT.data_ptr();
    } else {
      return nullptr;
    }
  }();
  std::array<T, tatooine::min(M, N) - 1> superb;

  const auto info = [&] {
    if constexpr (is_same<double, T>) {
      return LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr, ldvt,
                            superb.data());
    } else if constexpr (is_same<float, T>) {
      return LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr, ldvt,
                            superb.data());
    } else if constexpr (is_same<std::complex<float>, T>) {
      return LAPACKE_cgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr, ldvt,
                            superb.data());
    } else if constexpr (is_same<std::complex<double>, T>) {
      return LAPACKE_zgesvd(LAPACK_COL_MAJOR, jobu, jobvt, M, N, A.data_ptr(),
                            M, s.data_ptr(), U_ptr, ldu, VT_ptr, ldvt,
                            superb.data());
    }
  }();
  if (info < 0) {
    throw std::runtime_error{"[tatooine::lapack::gesvd] - " +
                             std::to_string(-info) + "-th argument is invalid"};
  } else if (info > 0) {
    throw std::runtime_error{
        "[tatooine::lapack::gesvd] - DBDSQR did not converge. " +
        std::to_string(info) +
        " superdiagonals of an intermediate bidiagonal "
        "form B did not converge to zero."};
  }
  if constexpr (is_same<job::N_t, JOBU>) {
    if constexpr (is_same<job::N_t, JOBVT>) {
      return s;
    } else {
      return std::tuple{s, VT};
    }
  } else {
    if constexpr (is_same<job::N_t, JOBVT>) {
      return std::tuple{U, s};
    } else {
      return std::tuple{U, s, VT};
    }
  }
}
//------------------------------------------------------------------------------
/// Computes all eigenvalues and, optionally, eigenvectors of a real symmetric
/// matrix A.
template <typename Tensor, typename Real, size_t N, typename JOBZ,
          typename UPLO>
auto syev(base_tensor<Tensor, Real, N, N> const& A, JOBZ, UPLO) {
  static_assert(is_same<JOBZ, job::N_t> || is_same<JOBZ, job::V_t>);
  static_assert(is_same<UPLO, job::U_t> || is_same<UPLO, job::L_t>);
  constexpr char jobz = job::to_char<JOBZ>();
  constexpr char uplo = job::to_char<UPLO>();
  if constexpr (is_same<JOBZ, job::V_t>) {
    auto out = std::pair{mat<Real, N, N>{A}, vec<Real, N>{}};
    [[maybe_unused]] lapack_int info;
    if constexpr (is_same<Real, float>) {
      info = LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, uplo, N,
                           out.first.data_ptr(), N, out.second.data_ptr());
    } else if constexpr (is_same<Real, double>) {
      info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, N,
                           out.first.data_ptr(), N, out.second.data_ptr());
    }
    return out;
  } else if constexpr (is_same<JOBZ, job::N_t>) {
    auto                        out = vec<Real, N>{};
      mat<Real, N, N> C = A;
    [[maybe_unused]] lapack_int info;
    if constexpr (is_same<Real, float>) {
      info = LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, uplo, N, C.data_ptr(), N,
                           out.data_ptr());
    } else if constexpr (is_same<Real, double>) {
      info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, N, C.data_ptr(), N,
                           out.data_ptr());
    }
    return out;
  }
}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
