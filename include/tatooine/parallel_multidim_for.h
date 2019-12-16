#ifndef TATOOINE_PARALLEL_MULTIDIM_FOR_H
#define TATOOINE_PARALLEL_MULTIDIM_FOR_H
//=============================================================================
#include "type_traits.h"
#include "utility.h"
#if defined(_OPENMP)
#  include <omp.h>
#endif
//=============================================================================
namespace tatooine {
//=============================================================================
namespace detail {
//=============================================================================
template <size_t N, size_t D, size_t NThreads>
struct Loop {
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //============================================================================
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {
    static_assert(0 < N && 0 < D && D <= N, "count D as dimension+1");
  }
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
    for (m_is[D - 1] = 0; m_is[D - 1] < m_ends[D - 1]; ++m_is[D - 1]) {
      Loop<N, D - 1, NThreads> nested(m_is, m_ends);
      if (!nested(f)) return false;
    }
    return true;
  }
};
//==============================================================================
#if defined(_OPENMP)
template <size_t N, size_t D>
struct Loop<N, D, D> {
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //============================================================================
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {
    static_assert(0 < N && 0 < D && D <= N, "count D as dimension+1");
  }
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
    // std::cerr << "parallel for: D=" << D << std::endl;
#pragma omp parallel for
    for (size_t i = 0; i < m_ends[D - 1]; ++i) {
      auto jj   = m_is;
      jj[D - 1] = i;
      Loop<N, D - 1, D> nested(jj, m_ends);
      bool                      cont = nested(f);
      assert(cont && "cannot break out of parallel loop");
    }
    for (size_t i = D - 1; i < D; ++i) m_is[i] = m_ends[i];

    return true;
  }
};
#endif

//==============================================================================
template <size_t N, size_t NThreads>
struct Loop<N, 1, NThreads> {
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //============================================================================
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {}
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
    for (m_is[0] = 0; m_is[0] < m_ends[0]; ++m_is[0]) {
      if (!f(m_is)) return false;
    }
    return true;
  }
};

//==============================================================================
#if defined(_OPENMP)
template <size_t N>
struct Loop<N, 1, 1> {
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //============================================================================
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {}
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
    // std::cerr << "parallel for: D=1\n";
#pragma omp parallel for
    for (size_t i = 0; i < m_ends[0]; ++i) {
      auto jj   = m_is;
      jj[0]     = i;
      bool cont = f(jj);
      assert(cont && "cannot break out of parallel loop");
    }
    m_is[0] = m_ends[0];
    return true;
  }
};
#endif

//==============================================================================
}  // namespace detail
//==============================================================================

template <size_t NThreads, typename... Ends, typename F,
          enable_if_integral<Ends...> = true>
bool parallel_for(Ends... ends, F&& f) {
  auto is = make_array<size_t, sizeof...(Ends)>(0);
  std::array<size_t, sizeof...(Ends)> ends_arr{static_cast<size_t>(ends)...};
  return detail::Loop<sizeof...(Ends), sizeof...(Ends), NThreads>{is, ends_arr}(std::forward<F>(f));
}
}  // namespace tatooine
#endif
