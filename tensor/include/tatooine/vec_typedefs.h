#ifndef TATOOINE_VEC_TYPEDEFS_H
#define TATOOINE_VEC_TYPEDEFS_H
//==============================================================================
#include <tatooine/vec.h>
#include <tatooine/tensor_concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::size_t N>
using Vec = vec<real_number, N>;
template <typename T>
using Vec2 = vec<T, 2>;
template <typename T>
using Vec3 = vec<T, 3>;
template <typename T>
using Vec4 = vec<T, 4>;
template <typename T>
using Vec5 = vec<T, 5>;
template <typename T>
using Vec6 = vec<T, 6>;
template <typename T>
using Vec7 = vec<T, 7>;
template <typename T>
using Vec8 = vec<T, 8>;
template <typename T>
using Vec9 = vec<T, 9>;

using vec2 = Vec<2>;
using vec3 = Vec<3>;
using vec4 = Vec<4>;
using vec5 = Vec<5>;
using vec6 = Vec<6>;
using vec7 = Vec<7>;
using vec8 = Vec<8>;
using vec9 = Vec<9>;

template <std::size_t N>
using VecF  = vec<float, N>;
using vec2f = VecF<2>;
using vec3f = VecF<3>;
using vec4f = VecF<4>;
using vec5f = VecF<5>;
using vec6f = VecF<6>;
using vec7f = VecF<7>;
using vec8f = VecF<8>;
using vec9f = VecF<9>;

template <std::size_t N>
using VecD  = vec<double, N>;
using vec2d = VecD<2>;
using vec3d = VecD<3>;
using vec4d = VecD<4>;
using vec5d = VecD<5>;
using vec6d = VecD<6>;
using vec7d = VecD<7>;
using vec8d = VecD<8>;
using vec9d = VecD<9>;

template <std::size_t N>
using VecI64  = vec<std::int64_t, N>;
using vec2i64 = VecI64<2>;
using vec3i64 = VecI64<3>;
using vec4i64 = VecI64<4>;
using vec5i64 = VecI64<5>;
using vec6i64 = VecI64<6>;
using vec7i64 = VecI64<7>;
using vec8i64 = VecI64<8>;
using vec9i64 = VecI64<9>;

template <typename T, std::size_t N>
using complex_vec = vec<std::complex<T>, N>;
template <std::size_t N>
using ComplexVec   = vec<std::complex<real_number>, N>;
template <typename T>
using ComplexVec2   = vec<std::complex<T>, 2>;
template <typename T>
using ComplexVec3   = vec<std::complex<T>, 3>;
template <typename T>
using ComplexVec4   = vec<std::complex<T>, 4>;
template <typename T>
using ComplexVec5   = vec<std::complex<T>, 5>;
template <typename T>
using ComplexVec6   = vec<std::complex<T>, 6>;
template <typename T>
using ComplexVec7   = vec<std::complex<T>, 7>;
template <typename T>
using ComplexVec8   = vec<std::complex<T>, 8>;
template <typename T>
using ComplexVec9   = vec<std::complex<T>, 9>;
using complex_vec2 = ComplexVec<2>;
using complex_vec3 = ComplexVec<3>;
using complex_vec4 = ComplexVec<4>;
using complex_vec5 = ComplexVec<5>;
using complex_vec6 = ComplexVec<6>;
using complex_vec7 = ComplexVec<7>;
using complex_vec8 = ComplexVec<8>;
using complex_vec9 = ComplexVec<9>;
template <std::size_t N>
using ComplexVecD   = vec<std::complex<double>, N>;
using complex_vec2d = ComplexVecD<2>;
using complex_vec3d = ComplexVecD<3>;
using complex_vec4d = ComplexVecD<4>;
using complex_vec5d = ComplexVecD<5>;
using complex_vec6d = ComplexVecD<6>;
using complex_vec7d = ComplexVecD<7>;
using complex_vec8d = ComplexVecD<8>;
using complex_vec9d = ComplexVecD<9>;
template <std::size_t N>
using ComplexVecF   = vec<std::complex<float>, N>;
using complex_vec2f = ComplexVecF<2>;
using complex_vec3f = ComplexVecF<3>;
using complex_vec4f = ComplexVecF<4>;
using complex_vec5f = ComplexVecF<5>;
using complex_vec6f = ComplexVecF<6>;
using complex_vec7f = ComplexVecF<7>;
using complex_vec8f = ComplexVecF<8>;
using complex_vec9f = ComplexVecF<9>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
