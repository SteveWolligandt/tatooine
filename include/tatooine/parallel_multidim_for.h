#ifndef TATOOINE_PARALLEL_MULTIDIM_FOR_H
#define TATOOINE_PARALLEL_MULTIDIM_FOR_H
//=============================================================================
#include <cassert>

#include "functional.h"
#include "type_traits.h"
#include "utility.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
//=============================================================================
namespace tatooine {
//=============================================================================
namespace detail {
//=============================================================================
template <size_t N, size_t D, size_t OMP>
struct Loop {
  static_assert(0 < N && 0 < D && D <= N, "count D as dimension+1");
  //----------------------------------------------------------------------------
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //----------------------------------------------------------------------------
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {}
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
    for (m_is[D - 1] = 0; m_is[D - 1] < m_ends[D - 1]; ++m_is[D - 1]) {
      Loop<N, D - 1, OMP> nested{m_is, m_ends};
      if (!nested(std::forward<F>(f))) { return false; }
    }
    return true;
  }
};
//==============================================================================
template <size_t N, size_t OMP>
struct Loop<N, 1, OMP> {
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //----------------------------------------------------------------------------
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {}
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
    return (*this)(std::forward<F>(f), std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  template <typename F, size_t... Is>
  bool operator()(F&& f, std::index_sequence<Is...>) {
    for (m_is.front() = 0; m_is.front() < m_ends.front(); ++m_is.front()) {
      if constexpr (std::is_same_v<std::invoke_result_t<F, decltype(Is)...>,
                                   void>) {
        invoke_unpacked(std::forward<F>(f), unpack(m_is));
      } else {
        if (!invoke_unpacked(std::forward<F>(f), unpack(m_is))) {
          return false;
        }
      }
    }
    return true;
  }
};
//==============================================================================
#if defined(_OPENMP)
template <size_t N, size_t D>
struct Loop<N, D, D> {
  static_assert(0 < N && 0 < D && D <= N, "count D as dimension+1");
  //----------------------------------------------------------------------------
  std::array<size_t, N>&       m_is;
  const std::array<size_t, N>& m_ends;
  //----------------------------------------------------------------------------
  Loop(std::array<size_t, N>& is, const std::array<size_t, N>& ends)
      : m_is{is}, m_ends{ends} {}
  //----------------------------------------------------------------------------
  template <typename F>
  bool operator()(F&& f) {
#pragma omp parallel for
    for (size_t i = 0; i < m_ends[D - 1]; ++i) {
      auto js   = m_is;
      js[D - 1] = i;
      Loop<N, D - 1, D> nested{js, m_ends};
      const auto        cont = nested(std::forward<F>(f));
      assert(cont && "cannot break out of parallel loop");
    }
    for (size_t i = D - 1; i < D; ++i) { m_is[i] = m_ends[i]; }
    return true;
  }
};
//==============================================================================
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
    return (*this)(std::forward<F>(f), std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  template <typename F, size_t... Is>
  bool operator()(F&& f, std::index_sequence<Is...>) {
#pragma omp parallel for
    for (size_t i = 0; i < m_ends.front(); ++i) {
      auto js = m_is;
      js[0]=i;
      if constexpr (std::is_same_v<std::invoke_result_t<F, decltype(Is)...>,
                                   void>) {
        invoke_unpacked(std::forward<F>(f), unpack(js));
      } else {
        const auto cont = invoke_unpacked(std::forward<F>(f), unpack(js));
        assert(cont && "cannot break out of parallel loop");
      }
    }
    m_is[0] = m_ends[0];
    return true;
  }
};
#endif

//==============================================================================
}  // namespace detail
//==============================================================================

template <size_t OMP = 0, typename... Ends, typename F,
          enable_if_invocable<F, decltype((std::declval<Ends>(),
                                           size_t{0}))...> = true,
          enable_if_integral<Ends...>                      = true>
bool parallel_for(F&& f, Ends... ends) {
  constexpr auto        N  = sizeof...(Ends);
  auto                  is = make_array<size_t, N>(0);
  std::array<size_t, N> ends_arr{static_cast<size_t>(ends)...};
  return detail::Loop<N, N, OMP + 1>{is, ends_arr}(std::forward<F>(f));
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
