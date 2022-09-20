#ifndef TATOOINE_LAPACK_BASE_H
#define TATOOINE_LAPACK_BASE_H
//==============================================================================
#include <tatooine/blas/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack Lapack
///
/// <table>
/// <tr><th>Name	<th>Description</tr>
/// <tr><td>BD <td>bidiagonal matrix</tr>
/// <tr><td>DI <td>diagonal matrix</tr>
/// <tr><td>GB <td>general band matrix</tr>
/// <tr><td>GE <td>general matrix (i.e., unsymmetric, in some cases
/// rectangular)</tr> <tr><td>GG <td>general matrices, generalized problem
/// (i.e., a pair of general matrices)</tr> <tr><td>GT <td>general tridiagonal
/// matrix</tr> <tr><td>HB <td>(complex) Hermitian band matrix</tr> <tr><td>HE
/// <td>(complex) Hermitian matrix</tr> <tr><td>HG <td>upper Hessenberg
/// matrix, generalized problem (i.e. a Hessenberg and a triangular
/// matrix)</tr> <tr><td>HP <td>(complex) Hermitian, packed storage
/// matrix</tr> <tr><td>HS <td>upper Hessenberg matrix</tr> <tr><td>OP
/// <td>(real) orthogonal matrix, packed storage matrix</tr> <tr><td>OR
/// <td>(real) orthogonal matrix</tr> <tr><td>PB <td>symmetric matrix or
/// Hermitian matrix positive definite band</tr> <tr><td>PO <td>symmetric
/// matrix or Hermitian matrix positive definite</tr> <tr><td>PP <td>symmetric
/// matrix or Hermitian matrix positive definite, packed storage matrix</tr>
/// <tr><td>PT <td>symmetric matrix or Hermitian matrix positive definite
/// tridiagonal matrix</tr> <tr><td>SB <td>(real) symmetric band matrix</tr>
/// <tr><td>SP <td>symmetric, packed storage matrix</tr>
/// <tr><td>ST <td>(real) symmetric matrix tridiagonal matrix</tr>
/// <tr><td>SY <td>symmetric matrix</tr>
/// <tr><td>TB <td>triangular band matrix</tr>
/// <tr><td>TG <td>triangular matrices, generalized problem (i.e., a pair of
/// triangular matrices)</tr> <tr><td>TP <td>triangular, packed storage
/// matrix</tr> <tr><td>TR <td>triangular matrix (or in some cases
/// quasi-triangular)</tr> <tr><td>TZ <td>trapezoidal matrix</tr> <tr><td>UN
/// <td>(complex) unitary matrix</tr> <tr><td>UP <td>(complex) unitary, packed
/// storage matrix</tr>
/// </table>
using blas::diag;
using blas::format;
using blas::op;
using blas::uplo;
using blas::side;

// like blas::side, but adds Both for trevc
enum class sides : char {
  left  = 'L',
  right = 'R',
  both  = 'B',
};
// Job for computing eigenvectors and singular vectors
// # needs custom map
enum class job : char {
  no_vec     = 'N',
  vec        = 'V',  // geev, syev, ...
  update_vec = 'U',  // gghrd#, hbtrd, hgeqz#, hseqr#, ... (many compq or compz)

  all_vec       = 'A',  // gesvd, gesdd, gejsv#
  some_vec      = 'S',  // gesvd, gesdd, gejsv#, gesvj#
  overwrite_vec = 'O',  // gesvd, gesdd

  compact_vec  = 'P',  // bdsdc
  some_vec_tol = 'C',  // gesvj
  vec_jacobi   = 'J',  // gejsv
  workspace    = 'W',  // gejsv
};

enum class norm : char {
  one = '1',  // or 'O'
  two = '2',
  inf = 'I',
  fro = 'F',  // or 'E'
  max = 'M',
};
// hseqr
enum class job_schur : char {
  eigenvalues = 'E',
  schur       = 'S',
};
// gees
// todo: generic yes/no
enum class sort {
  not_sorted = 'N',
  sorted     = 'S',
};
// syevx
enum class range {
  all   = 'A',
  value = 'V',
  index = 'I',
};
enum class vect {
  q    = 'Q',  // orgbr, ormbr
  p    = 'P',  // orgbr, ormbr
  none = 'N',  // orgbr, ormbr, gbbrd
  both = 'B',  // orgbr, ormbr, gbbrd
};
// larfb
enum class direction {
  forward  = 'F',
  backward = 'B',
};
// larfb
enum class store_v {
  column_wise = 'C',
  row_wise    = 'R',
};
// lascl, laset
enum class matrix_type {
  general    = 'G',
  lower      = 'L',
  upper      = 'U',
  hessenberg = 'H',
  lower_band = 'B',
  upper_band = 'Q',
  band       = 'Z',
};
// trevc
enum class how_many {
  all           = 'A',
  backtransform = 'B',
  select        = 'S',
};
// *svx, *rfsx
enum class equed {
  none = 'N',
  row  = 'R',
  col  = 'C',
  both = 'B',
  yes  = 'Y',  // porfsx
};
// *svx
// todo: what's good name for this?
enum class factored {
  factored    = 'F',
  notFactored = 'N',
  equilibrate = 'E',
};
// geesx, trsen
enum class sense {
  none        = 'N',
  eigenvalues = 'E',
  subspace    = 'V',
  both        = 'B',
};
// disna
enum class job_cond {
  eigen_vec          = 'E',
  left_singular_vec  = 'L',
  right_singular_vec = 'R',
};
// {ge,gg}{bak,bal}
enum class balance {
  none    = 'N',
  permute = 'P',
  scale   = 'S',
  both    = 'B',
};
// stebz, larrd, stein docs
enum class order {
  block  = 'B',
  entire = 'E',
};
// check_ortho (LAPACK testing zunt01)
enum class row_col {
  col = 'C',
  row = 'R',
};
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
