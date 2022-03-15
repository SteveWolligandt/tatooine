#ifndef TATOOINE_BLAS_BASE_H
#define TATOOINE_BLAS_BASE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <blas.hh>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex Real, std::size_t... N>
struct tensor;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::blas {
//==============================================================================
/// \defgroup blas BLAS
/// <table>
/// <tr><th>Name	<th>Description</tr>
/// <tr><td>BD <td>bidiagonal matrix</tr>
/// <tr><td>DI <td>diagonal matrix</tr>
/// <tr><td>GB <td>general band matrix</tr>
/// <tr><td>GE <td>general matrix (i.e., unsymmetric, in some cases
/// rectangular)</tr> <tr><td>GG <td>general matrices, generalized problem
/// (i.e., a pair of general matrices)</tr> <tr><td>GT <td>general tridiagonal
/// matrix</tr> <tr><td>HB <td>(complex) Hermitian band matrix</tr> <tr><td>HE
/// <td>(complex) Hermitian matrix</tr> <tr><td>HG <td>upper Hessenberg matrix,
/// generalized problem (i.e. a Hessenberg and a triangular matrix)</tr>
/// <tr><td>HP <td>(complex) Hermitian, packed storage matrix</tr>
/// <tr><td>HS <td>upper Hessenberg matrix</tr>
/// <tr><td>OP <td>(real) orthogonal matrix, packed storage matrix</tr>
/// <tr><td>OR <td>(real) orthogonal matrix</tr>
/// <tr><td>PB <td>symmetric matrix or Hermitian matrix positive definite
/// band</tr> <tr><td>PO <td>symmetric matrix or Hermitian matrix positive
/// definite</tr> <tr><td>PP <td>symmetric matrix or Hermitian matrix positive
/// definite, packed storage matrix</tr> <tr><td>PT <td>symmetric matrix or
/// Hermitian matrix positive definite tridiagonal matrix</tr> <tr><td>SB
/// <td>(real) symmetric band matrix</tr> <tr><td>SP <td>symmetric, packed
/// storage matrix</tr> <tr><td>ST <td>(real) symmetric matrix tridiagonal
/// matrix</tr> <tr><td>SY <td>symmetric matrix</tr> <tr><td>TB <td>triangular
/// band matrix</tr> <tr><td>TG <td>triangular matrices, generalized problem
/// (i.e., a pair of triangular matrices)</tr> <tr><td>TP <td>triangular, packed
/// storage matrix</tr> <tr><td>TR <td>triangular matrix (or in some cases
/// quasi-triangular)</tr> <tr><td>TZ <td>trapezoidal matrix</tr> <tr><td>UN
/// <td>(complex) unitary matrix</tr> <tr><td>UP <td>(complex) unitary, packed
/// storage matrix</tr>
/// </table>
using ::blas::Layout;
using ::blas::Op;

/// \defgroup blas1 BLAS Level 1
/// \brief Vector Operations
/// \ingroup blas


/// \defgroup blas2 BLAS Level 2
/// \brief Matrix-Vector Operations
/// \ingroup blas

/// \defgroup blas3 BLAS Level 3
/// \brief Matrix-Matrix Operations
/// \ingroup blas
//==============================================================================
}  // namespace tatooine::blas
//==============================================================================
#endif

