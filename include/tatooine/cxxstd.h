#ifndef TATOOINE_CXXSTD_H
#define TATOOINE_CXXSTD_H

//==============================================================================
namespace tatooine {
//==============================================================================
struct _cxx {};
static constexpr _cxx cxx;
struct _cxx98 {};
static constexpr _cxx98 cxx98;
struct _cxx11 {};
static constexpr _cxx11 cxx11;
struct _cxx14 {};
static constexpr _cxx14 cxx14;
struct _cxx17 {};
static constexpr _cxx17 cxx17;
struct _cxx20 {};
static constexpr _cxx20 cxx20;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr inline bool operator>=(_cxx, _cxx  ) { return true; }
constexpr inline bool operator>=(_cxx, _cxx98) { return false; }
constexpr inline bool operator>=(_cxx, _cxx11) { return false; }
constexpr inline bool operator>=(_cxx, _cxx14) { return false; }
constexpr inline bool operator>=(_cxx, _cxx17) { return false; }
constexpr inline bool operator>=(_cxx, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>=(_cxx98, _cxx  ) { return true; }
constexpr inline bool operator>=(_cxx98, _cxx98) { return true; }
constexpr inline bool operator>=(_cxx98, _cxx11) { return false; }
constexpr inline bool operator>=(_cxx98, _cxx14) { return false; }
constexpr inline bool operator>=(_cxx98, _cxx17) { return false; }
constexpr inline bool operator>=(_cxx98, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~

constexpr inline bool operator>=(_cxx11, _cxx  ) { return true; }
constexpr inline bool operator>=(_cxx11, _cxx98) { return true; }
constexpr inline bool operator>=(_cxx11, _cxx11) { return true; }
constexpr inline bool operator>=(_cxx11, _cxx14) { return false; }
constexpr inline bool operator>=(_cxx11, _cxx17) { return false; }
constexpr inline bool operator>=(_cxx11, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>=(_cxx14, _cxx  ) { return true; }
constexpr inline bool operator>=(_cxx14, _cxx98) { return true; }
constexpr inline bool operator>=(_cxx14, _cxx11) { return true; }
constexpr inline bool operator>=(_cxx14, _cxx14) { return true; }
constexpr inline bool operator>=(_cxx14, _cxx17) { return false; }
constexpr inline bool operator>=(_cxx14, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>=(_cxx17, _cxx  ) { return true; }
constexpr inline bool operator>=(_cxx17, _cxx98) { return true; }
constexpr inline bool operator>=(_cxx17, _cxx11) { return true; }
constexpr inline bool operator>=(_cxx17, _cxx14) { return true; }
constexpr inline bool operator>=(_cxx17, _cxx17) { return true; }
constexpr inline bool operator>=(_cxx17, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>=(_cxx20, _cxx  ) { return true; }
constexpr inline bool operator>=(_cxx20, _cxx98) { return true; }
constexpr inline bool operator>=(_cxx20, _cxx11) { return true; }
constexpr inline bool operator>=(_cxx20, _cxx14) { return true; }
constexpr inline bool operator>=(_cxx20, _cxx17) { return true; }
constexpr inline bool operator>=(_cxx20, _cxx20) { return true; }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr inline bool operator>(_cxx, _cxx  ) { return false; }
constexpr inline bool operator>(_cxx, _cxx98) { return false; }
constexpr inline bool operator>(_cxx, _cxx11) { return false; }
constexpr inline bool operator>(_cxx, _cxx14) { return false; }
constexpr inline bool operator>(_cxx, _cxx17) { return false; }
constexpr inline bool operator>(_cxx, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>(_cxx98, _cxx  ) { return true; }
constexpr inline bool operator>(_cxx98, _cxx98) { return false; }
constexpr inline bool operator>(_cxx98, _cxx11) { return false; }
constexpr inline bool operator>(_cxx98, _cxx14) { return false; }
constexpr inline bool operator>(_cxx98, _cxx17) { return false; }
constexpr inline bool operator>(_cxx98, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>(_cxx11, _cxx  ) { return true; }
constexpr inline bool operator>(_cxx11, _cxx98) { return true; }
constexpr inline bool operator>(_cxx11, _cxx11) { return false; }
constexpr inline bool operator>(_cxx11, _cxx14) { return false; }
constexpr inline bool operator>(_cxx11, _cxx17) { return false; }
constexpr inline bool operator>(_cxx11, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>(_cxx14, _cxx  ) { return true; }
constexpr inline bool operator>(_cxx14, _cxx98) { return true; }
constexpr inline bool operator>(_cxx14, _cxx11) { return true; }
constexpr inline bool operator>(_cxx14, _cxx14) { return false; }
constexpr inline bool operator>(_cxx14, _cxx17) { return false; }
constexpr inline bool operator>(_cxx14, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>(_cxx17, _cxx  ) { return true; }
constexpr inline bool operator>(_cxx17, _cxx98) { return true; }
constexpr inline bool operator>(_cxx17, _cxx11) { return true; }
constexpr inline bool operator>(_cxx17, _cxx14) { return true; }
constexpr inline bool operator>(_cxx17, _cxx17) { return false; }
constexpr inline bool operator>(_cxx17, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator>(_cxx20, _cxx  ) { return true; }
constexpr inline bool operator>(_cxx20, _cxx98) { return true; }
constexpr inline bool operator>(_cxx20, _cxx11) { return true; }
constexpr inline bool operator>(_cxx20, _cxx14) { return true; }
constexpr inline bool operator>(_cxx20, _cxx17) { return true; }
constexpr inline bool operator>(_cxx20, _cxx20) { return false; }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr inline bool operator<=(_cxx, _cxx  ) { return true; }
constexpr inline bool operator<=(_cxx, _cxx98) { return true; }
constexpr inline bool operator<=(_cxx, _cxx11) { return true; }
constexpr inline bool operator<=(_cxx, _cxx14) { return true; }
constexpr inline bool operator<=(_cxx, _cxx17) { return true; }
constexpr inline bool operator<=(_cxx, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<=(_cxx98, _cxx  ) { return false; }
constexpr inline bool operator<=(_cxx98, _cxx98) { return true; }
constexpr inline bool operator<=(_cxx98, _cxx11) { return true; }
constexpr inline bool operator<=(_cxx98, _cxx14) { return true; }
constexpr inline bool operator<=(_cxx98, _cxx17) { return true; }
constexpr inline bool operator<=(_cxx98, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<=(_cxx11, _cxx  ) { return false; }
constexpr inline bool operator<=(_cxx11, _cxx98) { return false; }
constexpr inline bool operator<=(_cxx11, _cxx11) { return true; }
constexpr inline bool operator<=(_cxx11, _cxx14) { return true; }
constexpr inline bool operator<=(_cxx11, _cxx17) { return true; }
constexpr inline bool operator<=(_cxx11, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<=(_cxx14, _cxx  ) { return false; }
constexpr inline bool operator<=(_cxx14, _cxx98) { return false; }
constexpr inline bool operator<=(_cxx14, _cxx11) { return false; }
constexpr inline bool operator<=(_cxx14, _cxx14) { return true; }
constexpr inline bool operator<=(_cxx14, _cxx17) { return true; }
constexpr inline bool operator<=(_cxx14, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<=(_cxx17, _cxx  ) { return false; }
constexpr inline bool operator<=(_cxx17, _cxx98) { return false; }
constexpr inline bool operator<=(_cxx17, _cxx11) { return false; }
constexpr inline bool operator<=(_cxx17, _cxx14) { return false; }
constexpr inline bool operator<=(_cxx17, _cxx17) { return true; }
constexpr inline bool operator<=(_cxx17, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<=(_cxx20, _cxx  ) { return false; }
constexpr inline bool operator<=(_cxx20, _cxx98) { return false; }
constexpr inline bool operator<=(_cxx20, _cxx11) { return false; }
constexpr inline bool operator<=(_cxx20, _cxx14) { return false; }
constexpr inline bool operator<=(_cxx20, _cxx17) { return false; }
constexpr inline bool operator<=(_cxx20, _cxx20) { return true; }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr inline bool operator<(_cxx, _cxx  ) { return false; }
constexpr inline bool operator<(_cxx, _cxx98) { return true; }
constexpr inline bool operator<(_cxx, _cxx11) { return true; }
constexpr inline bool operator<(_cxx, _cxx14) { return true; }
constexpr inline bool operator<(_cxx, _cxx17) { return true; }
constexpr inline bool operator<(_cxx, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<(_cxx98, _cxx  ) { return false; }
constexpr inline bool operator<(_cxx98, _cxx98) { return false; }
constexpr inline bool operator<(_cxx98, _cxx11) { return true; }
constexpr inline bool operator<(_cxx98, _cxx14) { return true; }
constexpr inline bool operator<(_cxx98, _cxx17) { return true; }
constexpr inline bool operator<(_cxx98, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<(_cxx11, _cxx  ) { return false; }
constexpr inline bool operator<(_cxx11, _cxx98) { return false; }
constexpr inline bool operator<(_cxx11, _cxx11) { return false; }
constexpr inline bool operator<(_cxx11, _cxx14) { return true; }
constexpr inline bool operator<(_cxx11, _cxx17) { return true; }
constexpr inline bool operator<(_cxx11, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<(_cxx14, _cxx  ) { return false; }
constexpr inline bool operator<(_cxx14, _cxx98) { return false; }
constexpr inline bool operator<(_cxx14, _cxx11) { return false; }
constexpr inline bool operator<(_cxx14, _cxx14) { return false; }
constexpr inline bool operator<(_cxx14, _cxx17) { return true; }
constexpr inline bool operator<(_cxx14, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<(_cxx17, _cxx  ) { return false; }
constexpr inline bool operator<(_cxx17, _cxx98) { return false; }
constexpr inline bool operator<(_cxx17, _cxx11) { return false; }
constexpr inline bool operator<(_cxx17, _cxx14) { return false; }
constexpr inline bool operator<(_cxx17, _cxx17) { return false; }
constexpr inline bool operator<(_cxx17, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator<(_cxx20, _cxx  ) { return false; }
constexpr inline bool operator<(_cxx20, _cxx98) { return false; }
constexpr inline bool operator<(_cxx20, _cxx11) { return false; }
constexpr inline bool operator<(_cxx20, _cxx14) { return false; }
constexpr inline bool operator<(_cxx20, _cxx17) { return false; }
constexpr inline bool operator<(_cxx20, _cxx20) { return false; }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr inline bool operator==(_cxx, _cxx  ) { return true; }
constexpr inline bool operator==(_cxx, _cxx98) { return false; }
constexpr inline bool operator==(_cxx, _cxx11) { return false; }
constexpr inline bool operator==(_cxx, _cxx14) { return false; }
constexpr inline bool operator==(_cxx, _cxx17) { return false; }
constexpr inline bool operator==(_cxx, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator==(_cxx98, _cxx  ) { return false; }
constexpr inline bool operator==(_cxx98, _cxx98) { return true; }
constexpr inline bool operator==(_cxx98, _cxx11) { return false; }
constexpr inline bool operator==(_cxx98, _cxx14) { return false; }
constexpr inline bool operator==(_cxx98, _cxx17) { return false; }
constexpr inline bool operator==(_cxx98, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator==(_cxx11, _cxx  ) { return false; }
constexpr inline bool operator==(_cxx11, _cxx98) { return false; }
constexpr inline bool operator==(_cxx11, _cxx11) { return true; }
constexpr inline bool operator==(_cxx11, _cxx14) { return false; }
constexpr inline bool operator==(_cxx11, _cxx17) { return false; }
constexpr inline bool operator==(_cxx11, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator==(_cxx14, _cxx  ) { return false; }
constexpr inline bool operator==(_cxx14, _cxx98) { return false; }
constexpr inline bool operator==(_cxx14, _cxx11) { return false; }
constexpr inline bool operator==(_cxx14, _cxx14) { return true; }
constexpr inline bool operator==(_cxx14, _cxx17) { return false; }
constexpr inline bool operator==(_cxx14, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator==(_cxx17, _cxx  ) { return false; }
constexpr inline bool operator==(_cxx17, _cxx98) { return false; }
constexpr inline bool operator==(_cxx17, _cxx11) { return false; }
constexpr inline bool operator==(_cxx17, _cxx14) { return false; }
constexpr inline bool operator==(_cxx17, _cxx17) { return true; }
constexpr inline bool operator==(_cxx17, _cxx20) { return false; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator==(_cxx20, _cxx  ) { return false; }
constexpr inline bool operator==(_cxx20, _cxx98) { return false; }
constexpr inline bool operator==(_cxx20, _cxx11) { return false; }
constexpr inline bool operator==(_cxx20, _cxx14) { return false; }
constexpr inline bool operator==(_cxx20, _cxx17) { return false; }
constexpr inline bool operator==(_cxx20, _cxx20) { return true; }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
constexpr inline bool operator!=(_cxx, _cxx  ) { return false; }
constexpr inline bool operator!=(_cxx, _cxx98) { return true; }
constexpr inline bool operator!=(_cxx, _cxx11) { return true; }
constexpr inline bool operator!=(_cxx, _cxx14) { return true; }
constexpr inline bool operator!=(_cxx, _cxx17) { return true; }
constexpr inline bool operator!=(_cxx, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator!=(_cxx98, _cxx  ) { return true; }
constexpr inline bool operator!=(_cxx98, _cxx98) { return false; }
constexpr inline bool operator!=(_cxx98, _cxx11) { return true; }
constexpr inline bool operator!=(_cxx98, _cxx14) { return true; }
constexpr inline bool operator!=(_cxx98, _cxx17) { return true; }
constexpr inline bool operator!=(_cxx98, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator!=(_cxx11, _cxx  ) { return true; }
constexpr inline bool operator!=(_cxx11, _cxx98) { return true; }
constexpr inline bool operator!=(_cxx11, _cxx11) { return false; }
constexpr inline bool operator!=(_cxx11, _cxx14) { return true; }
constexpr inline bool operator!=(_cxx11, _cxx17) { return true; }
constexpr inline bool operator!=(_cxx11, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator!=(_cxx14, _cxx  ) { return true; }
constexpr inline bool operator!=(_cxx14, _cxx98) { return true; }
constexpr inline bool operator!=(_cxx14, _cxx11) { return true; }
constexpr inline bool operator!=(_cxx14, _cxx14) { return false; }
constexpr inline bool operator!=(_cxx14, _cxx17) { return true; }
constexpr inline bool operator!=(_cxx14, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator!=(_cxx17, _cxx  ) { return true; }
constexpr inline bool operator!=(_cxx17, _cxx98) { return true; }
constexpr inline bool operator!=(_cxx17, _cxx11) { return true; }
constexpr inline bool operator!=(_cxx17, _cxx14) { return true; }
constexpr inline bool operator!=(_cxx17, _cxx17) { return false; }
constexpr inline bool operator!=(_cxx17, _cxx20) { return true; }
//  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
constexpr inline bool operator!=(_cxx20, _cxx  ) { return true; }
constexpr inline bool operator!=(_cxx20, _cxx98) { return true; }
constexpr inline bool operator!=(_cxx20, _cxx11) { return true; }
constexpr inline bool operator!=(_cxx20, _cxx14) { return true; }
constexpr inline bool operator!=(_cxx20, _cxx17) { return true; }
constexpr inline bool operator!=(_cxx20, _cxx20) { return false; }

//==============================================================================
constexpr inline auto cxx_standard() {
#if __cplusplus == 201703L
  return cxx17;
#elif __cplusplus == 201402L
  return cxx14;
#elif __cplusplus == 201103L
  return cxx11;
#elif __cplusplus == 199711L
  return cxx98;
#else
  return cxx;
#endif
}
//==============================================================================
#ifndef TATOOINE_CXXSTD
  #define TATOOINE_CXX20 5
  #define TATOOINE_CXX17 4
  #define TATOOINE_CXX14 3
  #define TATOOINE_CXX11 2
  #define TATOOINE_CXX98 1
  #define TATOOINE_CXX   0

  #if __cplusplus == 201703L           
    #define TATOOINE_CXXSTD TATOOINE_CXX17
  #elif __cplusplus == 201402L         
    #define TATOOINE_CXXSTD TATOOINE_CXX14
  #elif __cplusplus == 201103L         
    #define TATOOINE_CXXSTD TATOOINE_CXX11
  #elif __cplusplus == 199711L         
    #define TATOOINE_CXXSTD TATOOINE_CXX98
  #else                                
    #define TATOOINE_CXXSTD TATOOINE_CXX  
  #endif

  #define is_cxx20() TATOOINE_CXXSTD == TATOOINE_CXX20
  #define is_cxx17() TATOOINE_CXXSTD == TATOOINE_CXX17
  #define is_cxx14() TATOOINE_CXXSTD == TATOOINE_CXX14
  #define is_cxx11() TATOOINE_CXXSTD == TATOOINE_CXX11
  #define is_cxx98() TATOOINE_CXXSTD == TATOOINE_CXX98

  #define has_cxx20_support() TATOOINE_CXXSTD >= TATOOINE_CXX20
  #define has_cxx17_support() TATOOINE_CXXSTD >= TATOOINE_CXX17
  #define has_cxx14_support() TATOOINE_CXXSTD >= TATOOINE_CXX14
  #define has_cxx11_support() TATOOINE_CXXSTD >= TATOOINE_CXX11
  #define has_cxx98_support() TATOOINE_CXXSTD >= TATOOINE_CXX98
#endif

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
