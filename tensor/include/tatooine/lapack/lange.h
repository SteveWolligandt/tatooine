#ifndef TATOOINE_LAPACK_LANGE_H
#define TATOOINE_LAPACK_LANGE_H
//==============================================================================
extern "C" {
//==============================================================================
auto dlange_(char* NORM, int* M, int* N, double* A, int* LDA, double* WORK)
    -> double;
auto slange_(char* NORM, int* M, int* N, float* A, int* LDA, float* WORK)
    -> float;
//==============================================================================
} // extern "C"
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_lange LANGE
/// \brief Calculating norms.
/// \ingroup lapack
/// \{
//==============================================================================
template <std::floating_point Float>
auto lange(norm n, int M, int N, Float* A, int LDA, Float* WORK) -> Float {
  auto ret = Float{};
  if constexpr (std::same_as<Float, double>) {
    ret = dlange_(reinterpret_cast<char*>(&n), &M, &N, A, &LDA, WORK);
  } else if constexpr (std::same_as<Float, float>) {
    ret = slange_(reinterpret_cast<char*>(&n), &M, &N, A, &LDA, WORK);
  }
  return ret;
}
//------------------------------------------------------------------------------
template <std::floating_point Float>
auto lange(norm n, int M, int N, Float* A, int LDA) -> Float {
  auto ret = Float{};
  auto WORK = std::unique_ptr<Float>{nullptr};
  if (n == norm::inf) {
    WORK = std::unique_ptr<Float>{new Float[M]};
  }
  if constexpr (std::same_as<Float, double>) {
    ret = dlange_(reinterpret_cast<char*>(&n), &M, &N, A, &LDA, WORK.get());
  } else if constexpr (std::same_as<Float, float>) {
    ret = slange_(reinterpret_cast<char*>(&n), &M, &N, A, &LDA, WORK.get());
  }
  return ret;
}
//------------------------------------------------------------------------------
/// Returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed
template <typename T, size_t M, size_t N>
auto lange(tensor<T, M, N>& A, norm n) {
  return lange(n, M, N, A.data(), M);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
