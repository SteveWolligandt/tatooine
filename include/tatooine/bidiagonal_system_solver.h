#ifndef __TATOOINE_BIDIAGONAL_SYSTEM_QR_SOLVER_H
#define __TATOOINE_BIDIAGONAL_SYSTEM_QR_SOLVER_H

#include <cassert>
#include <cmath>
#include <cstdlib>

//==============================================================================
namespace tatooine {
//==============================================================================

//-------------------------------------------------------------------------------
// General solvers
//-------------------------------------------------------------------------------

//! Solve bidiagonal linear system A*x=b with superdiagonal.
//! The _x x _n matrix A is bidiagonal with diagonal _d and \a superdiagonal
//!_du. \param _n dimension of A and number of rows of RHS _b \param _nrhs
//! number of RHS in _b \param _d main diagonal of A (array of size _n) \param
//!_du superdiagonal of A (array of size _n-1) \param[in,out] _b RHS on input,
//! solution x on output \param _ldb leading dimension of _b \return false if A
//! is singular
template <typename T>
bool bdsvu(int _n, int _nrhs, const T* __restrict__ _d,
           const T* __restrict__ _du, T* __restrict__ _b, int _ldb) {
  // assert(_n > 1);
  assert(_nrhs > 0);
  assert(_ldb >= _n || _nrhs == 1);

  if (_d[_n - 1] == T(0)) return false;  // shall we handle inf?

  for (int j = 0; j < _nrhs; ++j) _b[(_n - 1) + j * _ldb] /= _d[_n - 1];

  for (int i = _n - 2; i >= 0; --i) {
    if (_d[i] == T(0)) return false;
    for (int j = 0; j < _nrhs; ++j)
      _b[i + j * _ldb] =
          (_b[i + j * _ldb] - _du[i] * _b[(i + 1) + j * _ldb]) / _d[i];
  }
  return true;
}

//! Solve bidiagonal linear system A*x=b with subdiagonal.
//! The _x x _n matrix A is bidiagonal with diagonal _d and \a subdiagonal _dl.
//! \param _n dimension of A and number of rows of RHS _b
//! \param _nrhs number of RHS in _b
//! \param _dl subdiagonal of A (array of size _n-1)
//! \param _d main diagonal of A (array of size _n)
//! \param[in,out] _b RHS on input, solution x on output
//! \param _ldb leading dimension of _b
//! \return false if A is singular
template <typename T>
bool bdsvl(int _n, int _nrhs, const T* __restrict__ _dl,
           const T* __restrict__ _d, T* __restrict__ _b, int _ldb) {
  // assert(_n > 1);
  assert(_nrhs > 0);
  assert(_ldb >= _n || _nrhs == 1);

  if (_d[0] == T(0)) return false;  // shall we handle inf?

  for (int j = 0; j < _nrhs; ++j) _b[0 + j * _ldb] /= _d[0];

  for (int i = 1; i < _n; ++i) {
    if (_d[i] == T(0)) return false;
    for (int j = 0; j < _nrhs; ++j)
      _b[i + j * _ldb] =
          (_b[i + j * _ldb] - _dl[i - 1] * _b[(i - 1) + j * _ldb]) / _d[i];
  }
  return true;
}

//-------------------------------------------------------------------------------
// Special solvers tailored to problem
//-------------------------------------------------------------------------------

//! This function solves your problem A*x=b blockwise w/o LAPACK.
//! Assume quadratic matrix A=[A11,A12;ONES,1] with
//! bidiagonal matrix A11=A(1:end-1,,1:end-1) and
//! A12=[0,...0,a12]
//! \param _n dimension of matrix A
//! \param _d main diagonal of A11 (_n-1 entries)
//! \param _du superdiagonal of A11 (first _n-2 entries) and a12 as (at index
//!            _n-1) \param[in,out] _b right hand side on input, solution x on
//!            output
//!
//! \b Note: calls \c alloca!
template <typename T>
bool solve_blockwise(int _n, const T* __restrict__ _d,
                     const T* __restrict__ _du, T* __restrict__ _b) {
  // assert(_n > 1);

  int m = _n - 1;

  // get s
  T s(1);
  {
    if (_d[m - 1] == T(0)) return false;

    T x = _du[m - 1] / _d[m - 1];
    s -= x;
    for (int i = m - 2; i >= 0; --i) {
      x *= -_du[i] / _d[i];
      s -= x;
    }
  }
  VC_DBG_P(s);

  // get y
  assert(_n < 2048 && "limit  temporary buffer size using alloca");
  T* b = (T*)alloca(_n * sizeof(T));
  memcpy(b, _b, _n * sizeof(T));

  bool regular = bdsvl(m, 1, _d, _du, b, 1);
  assert(regular);

  T y(b[m]);
  for (int i = 0; i < m; ++i) y -= b[i];
  y /= s;

  VC_DBG_P(y);

  _b[m - 1] -= _du[m - 1] * y;

  regular = bdsvu(m, 1, _d, _du, _b, 1);
  assert(regular);

  _b[m] = y;

  return true;
}

//-------------------------------------------------------------------------------

//! Apply rotation _x=[_c,-_s;_s,_c]*_x.
//! \param _c cosine
//! \param _s sine
//! \param _x[in,out] 2-vector
//! \sa solve_qr()
template <typename T>
void _planerot(T _c, T _s, T* __restrict__ _x) {
  T x   = _x[0];
  T y   = _x[1];
  _x[0] = _c * x - _s * y;
  _x[1] = _s * x + _c * y;
}

//! \def _Q_IN_NULL Tweak solve_qr()
//!
//! Store cosine and sine entries of Givens plane rotations in
//! different places: cos go to a temporary array of length _n, sin go
//! to _null.  The default is to use a 2*_n temporary array for both.
//!
//! This option may provide a slight advantage for rather large _n,
//! because we are accessing less memory. -- However, the access pattern
//! is worse!

#ifdef DOXYGEN_SKIP
#ifndef _Q_IN_NULL
#define _Q_IN_NULL
#endif
#endif

//! Get least-norm solution xln and _null space by QR factorization.
//!
//! \arg Input is the bidiagonal _n x (_n+_1) matrix A with diagonal _d and
//!      superdiagonal  _du.
//! \arg We use the factorization A'=Q*R.
//! \arg Output is the least-norm xln solution to A*x=b in _b, the vector _null
//!      spanning the kernel of A, and the factor R in _d (diagonal) and _du
//!      (superdiagonal).
//!
//! \param _n number of rows in A
//! \param[in,out] _d on \b input _d diagonal of A (_n elements),
//!                   on \b output diagonal of R
//! \param[in,out] _du on \b input _du superdiagonal of A (_n elements, because
//!                    A is _n x (_n+1), on \b output superdiagonal of R
//! \param[in,out] _b on \b input right hand side vector of length _n, on \b
//!                   output least-norm solution xln of size _n+1 (reserve
//!                   <b>_n+1</b> entries)
//! \param[out] _null (_n+1)-vector spanning nullspace of A \return \c true on
//!                   success, \c false if rank(A)<n
//!
template <typename T>
bool solve_qr(int _n, T* __restrict__ _d, T* __restrict__ _du,
              T* __restrict__ _b, T* __restrict__ _null) {
  assert(_n < 2048 && "limit  temporary buffer size using alloca");
#ifndef _Q_IN_NULL
  T* q = (T*)alloca(2 * _n * sizeof(T));  // store givens rotations
#else
  T* qc = (T*)alloca(_n * sizeof(T));  // store cosines of givens rotations
#endif

  //
  // 1. Compute givens rotations and construct R.
  //

  for (int j = 0; j < _n; ++j) {
    T c, s;

    if (_du[j] == T(0)) {
      c = 1;
      s = 0;
    } else {
      if (fabs(_du[j]) > fabs(_d[j])) {  // Do it the super-stable way!
        T tau = _d[j] / _du[j];          // (avoid overflows)
        s     = 1 / sqrt(T(1) + tau * tau);
        c     = s * tau;
      } else {
        T tau = _du[j] / _d[j];
        c     = 1 / sqrt(T(1) + tau * tau);
        s     = c * tau;
      }
    }

    _d[j] = c * _d[j] + s * _du[j];

    if (j + 1 < _n) {
      T a       = _d[j + 1];
      _du[j]    = s * a;  // first component...
      _d[j + 1] = c * a;  // ... is zero
    }
#ifndef _Q_IN_NULL
    q[2 * j]     = c;
    q[2 * j + 1] = s;
#else
    qc[j]    = c;
    _null[j] = s;  // store sine in _null
#endif
  }
  _du[_n - 1] = T(0);

  // Now we have Q' as givens rotations in q and
  //  R in _d (diagonal), _du (superdiagonal).

  //
  // 2. Solve y=R'\_b
  //

  if (!bdsvl(_n, 1, _du, _d, _b, 1)) return false;
  _b[_n] = T(0);

  //
  // 3. Compute xln=Q*y and _null=Q(:,end).
  //

  _null[_n] = T(1);

  for (int j = _n - 1; j >= 0; --j) {
#ifndef _Q_IN_NULL
    T c = q[2 * j];
    T s = q[2 * j + 1];
#else
    T c      = qc[j];
    T s      = _null[j];
#endif

    _null[j] = T(0);
    _planerot(c, s, _null + j);
    _planerot(c, s, _b + j);
  }

  return true;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
