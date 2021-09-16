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
/// \defgroup lapack Lapack
/// \{
//==============================================================================
static auto constexpr including_mkl_lapacke() {
#if TATOOINE_INCLUDE_MKL_LAPACKE
  return true;
#else
  return false;
#endif
}
//==============================================================================
/// \defgroup lapack_getrf GETRF
/// \ingroup lapack
/// \{
//==============================================================================
///  DGETRF computes an LU factorization of a general M-by-N matrix A using
///  partial pivoting with row interchanges.
template <size_t M, size_t N>
auto getrf(tensor<double, M, N>& A) {
  vec<int, tatooine::min(M, N)> p;
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  return A;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///  DGETRF computes an LU factorization of a general M-by-N matrix A using
///  partial pivoting with row interchanges.
template <size_t M, size_t N>
auto getrf(tensor<float, M, N>& A) {
  vec<int, tatooine::min(M, N)> p;
  LAPACKE_sgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  return A;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///  DGETRF computes an LU factorization of a general M-by-N matrix A using
///  partial pivoting with row interchanges.
template <size_t M, size_t N>
auto getrf(tensor<std::complex<double>, M, N>& A) {
  vec<int, tatooine::min(M, N)> p;
  LAPACKE_zgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  return A;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///  DGETRF computes an LU factorization of a general M-by-N matrix A using
///  partial pivoting with row interchanges.
template <size_t M, size_t N>
auto getrf(tensor<std::complex<float>, M, N>& A) {
  vec<int, tatooine::min(M, N)> p;
  LAPACKE_cgetrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, p.data_ptr());
  return A;
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_gesv GESV
/// \ingroup lapack
/// \{
//==============================================================================
template <size_t N>
auto gesv(tensor<double, N, N> A, tensor<double, N> b) {
  vec<int, N>          ipiv;
  static constexpr int nrhs = 1;
  LAPACKE_dgesv(LAPACK_COL_MAJOR, N, nrhs, A.data_ptr(), N, ipiv.data_ptr(),
                b.data_ptr(), N);
  return b;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t N>
auto gesv(tensor<float, N, N> A, tensor<float, N> b) {
  vec<int, N>          ipiv;
  static constexpr int nrhs = 1;
  LAPACKE_sgesv(LAPACK_COL_MAJOR, N, nrhs, A.data_ptr(), N, ipiv.data(),
                b.data_ptr(), N);
  return b;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t N, size_t NRHS>
auto gesv(tensor<double, N, N> A, tensor<double, N, NRHS> B) {
  std::array<int, N> ipiv;
  LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, A.data_ptr(), N, ipiv.data(),
                B.data_ptr(), N);
  return B;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t N, size_t NRHS>
auto gesv(tensor<float, N, N> A, tensor<float, N, NRHS> B) {
  std::array<int, N> ipiv;
  LAPACKE_sgesv(LAPACK_COL_MAJOR, N, NRHS, A.data_ptr(), N, ipiv.data_ptr(),
                B.data_ptr(), N);
  return B;
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_geqrf GEQRF
/// \ingroup lapack
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html'>LAPACK
/// documentation</a>
/// - <a
/// href='https://www.netlib.org/lapack/explore-html/d3/dd8/lapacke__dgeqrf_8c_a60664318f275813a1ab1faf3d44fe17f.html'>LAPACKE
/// documentation</a>
/// \{
//==============================================================================
template <size_t M, size_t N>
auto geqrf(tensor<double, M, N>& A, vec<double, (M < N) ? M : N>& tau) {
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, M, N, A.data_ptr(), M, tau.data_ptr());
  return A;
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_ormqr ORMQR
/// \ingroup lapack
///
/// **ORMQR** overwrites the general real `M x N` matrix `C` with
/// <table>
/// <tr><th>              <th>side = `L` <th>side = `R` </tr>
/// <tr><th>trans = `N`:  <td>`Q * C`    <td>`C * Q`    </tr>
/// <tr><th>trans = `T`:  <td>`Q^T * C`  <td>`C * Q^T`  </tr>
///
/// </table>
/// where `Q` is a real orthogonal matrix defined as the product of `k`
/// elementary reflectors
///
/// `Q = H(1) H(2) . . . H(k)`
///
/// as returned by \ref lapack_geqrf. `Q` is of order `M` if side = `L` and of order `N`
/// if side = `R`
///
/// - <a
/// href='https://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga17b0765a8a0e6547bcf933979b38f0b0.html'>LAPACK
/// documentation</a>
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/da/d35/lapacke__dormqr_8c_a6703fd6ad7fee186c33025bd88533b30.html'>LAPACKE
/// documentation</a>
/// \{
//==============================================================================
template <size_t K, size_t M>
auto ormqr(tensor<double, M, K>& A, tensor<double, M>& c, vec<double, K>& tau,
           char const side, char const trans) {
  LAPACKE_dormqr(LAPACK_COL_MAJOR, side, trans, M, 1, K, A.data_ptr(), M,
                 tau.data_ptr(), c.data_ptr(), M);
  return A;
}
//==============================================================================
template <size_t K, size_t M, size_t N>
auto ormqr(tensor<double, M, K>& A, tensor<double, M, N>& C, vec<double, K>& tau,
           char const side, char const trans) {
  LAPACKE_dormqr(LAPACK_COL_MAJOR, side, trans, M, N, K, A.data_ptr(), M,
                 tau.data_ptr(), C.data_ptr(), M);
  return A;
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_trtrs TRTRS
/// \ingroup lapack
///
/// **DTRTRS** solves a triangular system of the form
///
/// `A * X = B`  or  `A^T * X = B`
///
/// where `A` is a triangular matrix of order `N`, and `B` is an `N x RHS`
/// matrix. A check is made to verify that `A` is nonsingular.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga7068947990361e55177155d044435a5c.html'>LAPACK
/// documentation</a>
/// - <a
/// href=`http://www.netlib.org/lapack/explore-html/d9/da3/lapacke__dtrtrs_8c_a635dbba8f58ec13c74e25fe4a7c47cd6.html#a635dbba8f58ec13c74e25fe4a7c47cd6'>LAPACKE
/// documentation</a>
/// \{
//==============================================================================
/// \param uplo A is lower or upper triangular matrix:
/// - 'U': A is upper triangular;
/// - 'L': A is lower triangular.
/// \param diag A is unit (1s on main diagonal) or non-unit
/// - 'N': A is non-unit triangular
/// - 'U': A is unit triangular
template <size_t M, size_t N, size_t NRHS>
auto trtrs(tensor<double, M, N>& A, tensor<double, M, NRHS>& B, char const uplo,
           char const diag = 'N') {
  return LAPACKE_dtrtrs(LAPACK_COL_MAJOR, uplo, 'N', diag, N, NRHS,
                        A.data_ptr(), M, B.data_ptr(), M);
}
//------------------------------------------------------------------------------
/// \param uplo A is lower or upper triangular matrix:
/// - 'U': A is upper triangular;
/// - 'L': A is lower triangular.
/// \param diag A is unit (1s on main diagonal) or non-unit
/// - 'N': A is non-unit triangular
/// - 'U': A is unit triangular
template <size_t M, size_t N>
auto trtrs(tensor<double, M, N>& A, tensor<double, M>& B, char const uplo,
           char const diag = 'N') {
  return LAPACKE_dtrtrs(LAPACK_COL_MAJOR, uplo, 'N', diag, N, 1, A.data_ptr(),
                        M, B.data_ptr(), M);
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_lange LANGE
/// \ingroup lapack
/// \{
//==============================================================================
/// DLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed:
///   - `M` / `m` for max-norm
///   - `1` / `O` / `o` for 1-norm
///   - `I` / `i` for infinity-norm 
///   - `F` / `f` / `E` / `e` for frobenius-norm.
template <size_t M, size_t N>
auto lange(tensor<double, M, N>& A, char const norm) {
  return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// DLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed:
///   - `M` / `m` for max-norm
///   - `1` / `O` / `o` for 1-norm
///   - `I` / `i` for infinity-norm 
///   - `F` / `f` / `E` / `e` for frobenius-norm.
template <size_t M, size_t N>
auto lange(tensor<float, M, N>& A, char const norm) {
  return LAPACKE_slange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// DLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed:
///   - `M` / `m` for max-norm
///   - `1` / `O` / `o` for 1-norm
///   - `I` / `i` for infinity-norm 
///   - `F` / `f` / `E` / `e` for frobenius-norm.
template <size_t M, size_t N>
auto lange(tensor<std::complex<double>, M, N>& A, char const norm) {
  return LAPACKE_zlange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// DLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed:
///   - `M` / `m` for max-norm
///   - `1` / `O` / `o` for 1-norm
///   - `I` / `i` for infinity-norm 
///   - `F` / `f` / `E` / `e` for frobenius-norm.
template <size_t M, size_t N>
auto lange(tensor<std::complex<float>, M, N>& A, char const norm) {
  return LAPACKE_clange(LAPACK_COL_MAJOR, norm, M, N, A.data_ptr(), M);
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_gecon GECON
/// \ingroup lapack
/// \{
//==============================================================================
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
/// \}
//==============================================================================
/// \defgroup lapack_gesvd GESVD
///
/// Computes the singular value decomposition (SVD) of a real M-by-N matrix A,
/// optionally computing the left and/or right singular vectors.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html'>LAPACK
/// documentation</a>
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d0/dee/lapacke__dgesvd_8c_af31b3cb47f7cc3b9f6541303a2968c9f.html'>LAPACKE
/// documentation</a>
///
/// \ingroup lapack
/// \{
//==============================================================================
/// Computes the singular value decomposition (SVD) of a real M-by-N matrix A,
/// optionally computing the left and/or right singular vectors.
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
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_syev SYEV
/// \ingroup lapack
/// \{
//==============================================================================
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
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
