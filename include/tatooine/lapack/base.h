#ifndef TATOOINE_LAPACK_BASE_H
#define TATOOINE_LAPACK_BASE_H
//==============================================================================
#include <lapack.hh>
#include <tatooine/math.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
using ::lapack::Uplo;
using ::lapack::Side;
using ::lapack::Job;
using ::lapack::Diag;
using ::lapack::Norm;
using ::lapack::Op;
//==============================================================================
/// \defgroup lapack Lapack
/// 
/// <table>
/// <tr><th>Name	<th>Description</tr>
/// <tr><td>BD <td>bidiagonal matrix</tr>
/// <tr><td>DI <td>diagonal matrix</tr>
/// <tr><td>GB <td>general band matrix</tr>
/// <tr><td>GE <td>general matrix (i.e., unsymmetric, in some cases rectangular)</tr>
/// <tr><td>GG <td>general matrices, generalized problem (i.e., a pair of general matrices)</tr>
/// <tr><td>GT <td>general tridiagonal matrix</tr>
/// <tr><td>HB <td>(complex) Hermitian band matrix</tr>
/// <tr><td>HE <td>(complex) Hermitian matrix</tr>
/// <tr><td>HG <td>upper Hessenberg matrix, generalized problem (i.e. a Hessenberg and a triangular matrix)</tr>
/// <tr><td>HP <td>(complex) Hermitian, packed storage matrix</tr>
/// <tr><td>HS <td>upper Hessenberg matrix</tr>
/// <tr><td>OP <td>(real) orthogonal matrix, packed storage matrix</tr>
/// <tr><td>OR <td>(real) orthogonal matrix</tr>
/// <tr><td>PB <td>symmetric matrix or Hermitian matrix positive definite band</tr>
/// <tr><td>PO <td>symmetric matrix or Hermitian matrix positive definite</tr>
/// <tr><td>PP <td>symmetric matrix or Hermitian matrix positive definite, packed storage matrix</tr>
/// <tr><td>PT <td>symmetric matrix or Hermitian matrix positive definite tridiagonal matrix</tr>
/// <tr><td>SB <td>(real) symmetric band matrix</tr>
/// <tr><td>SP <td>symmetric, packed storage matrix</tr>
/// <tr><td>ST <td>(real) symmetric matrix tridiagonal matrix</tr>
/// <tr><td>SY <td>symmetric matrix</tr>
/// <tr><td>TB <td>triangular band matrix</tr>
/// <tr><td>TG <td>triangular matrices, generalized problem (i.e., a pair of triangular matrices)</tr>
/// <tr><td>TP <td>triangular, packed storage matrix</tr>
/// <tr><td>TR <td>triangular matrix (or in some cases quasi-triangular)</tr>
/// <tr><td>TZ <td>trapezoidal matrix</tr>
/// <tr><td>UN <td>(complex) unitary matrix</tr>
/// <tr><td>UP <td>(complex) unitary, packed storage matrix</tr>
/// </table>
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
