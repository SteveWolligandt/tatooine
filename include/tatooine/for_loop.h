#ifndef TATOOINE_FOR_LOOP_H
#define TATOOINE_FOR_LOOP_H
//==============================================================================
#include <array>
#include <tatooine/packages.h>
#include <tatooine/concepts.h>
#include "type_traits.h"
#include "utility.h"
#if TATOOINE_OPENMP_AVAILABLE
#include <omp.h>
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail {
//==============================================================================
/// \tparam Int integer type for counting
/// \tparam N number of nestings
/// \tparam I current nesting number counting backwards from N to 1
/// \tparam ParallelIndex If I and ParallelIndex are the same then the current
///         nested loop will be executed in parallel
template <typename Int, std::size_t N, std::size_t I, std::size_t ParallelIndex>
struct for_loop_impl {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<Int, N>&       m_status;
  const std::array<Int, N>& m_ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  /// recursively creates loops
#ifdef __cpp_concepts
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
#else
  template <std::size_t... Is, typename Iteration,
            enable_if<is_invocable<Iteration,
                                   decltype(((void)Is, Int{}))...> > = true>
#endif
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

    for (; m_status[I - 1] < m_ends[I - 1]; ++m_status[I - 1]) {
      if constexpr (returns_void) {
        // if if returns nothing just create another nested loop
        for_loop_impl<Int, N, I - 1, ParallelIndex>{
            m_status, m_ends}(std::forward<Iteration>(iteration));
      } else {
        // if iteration returns bool and the current nested iteration returns
        // false stop the whole nested for loop by recursively returning false
        if (!for_loop_impl<Int, N, I - 1, ParallelIndex>{
                m_status, m_ends}(std::forward<Iteration>(iteration))) {
          return false;
        }
      }
      // reset nested status
      m_status[I - 2] = 0;
    }
    if constexpr (returns_bool) {
      // return true if iteration never returned false
      return true;
    }
  }

 public:
  template <typename Iteration>
  constexpr auto operator()(Iteration&& iteration) const {
    return loop(std::forward<Iteration>(iteration),
                std::make_index_sequence<N>{});
  }
};
//------------------------------------------------------------------------------
/// \brief Last nesting of nested for loops, I is 1.
/// \tparam Int integer type for counting
/// \tparam N number of nestings
template <typename Int, std::size_t N, std::size_t ParallelIndex>
struct for_loop_impl<Int, N, 1, ParallelIndex> {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<Int, N>&       m_status;
  const std::array<Int, N>& m_ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
#ifdef __cpp_concepts
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
#else
  template <
      std::size_t... Is, typename Iteration,
      enable_if<is_invocable<Iteration, decltype(((void)Is, Int{}))...>> = true>
#endif
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

    for (; m_status[0] < m_ends[0]; ++m_status[0]) {
      if constexpr (returns_void) {
        // if if returns nothing just call it
        iteration(m_status[Is]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        if (!iteration(m_status[Is]...)) { return false; }
      }
    }
    if constexpr (returns_bool) {
      // return true if iteration never returned false
      return true;
    }
  }

 public:
  template <typename Iteration>
  constexpr auto operator()(Iteration&& iteration) const {
    return loop(std::forward<Iteration>(iteration),
                std::make_index_sequence<N>{});
  }
};
#if TATOOINE_OPENMP_AVAILABLE
////==============================================================================
/// nesting reached parallel state
/// \tparam Int integer type for counting
/// \tparam N number of nestings
/// \tparam I current nesting number counting backwards from N to 1
template <typename Int, std::size_t N, std::size_t I>
struct for_loop_impl<Int, N, I, I> {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<Int, N>&       m_status;
  const std::array<Int, N>& m_ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  /// recursively creates loops
#ifdef __cpp_concepts
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
#else
  template <
      std::size_t... Is, typename Iteration,
      enable_if<is_invocable<Iteration, decltype(((void)Is, Int{}))...>> = true>
#endif
  auto loop(Iteration&& iteration,
            std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = 0; i < m_ends[I - 1]; ++i) {
      auto status_copy   = m_status;
      status_copy[I - 1] = i;
      if constexpr (returns_void) {
        // if if returns nothing just create another nested loop
        for_loop_impl<Int, N, I - 1, I>{
            status_copy, m_ends}(std::forward<Iteration>(iteration));
      } else {
        // if iteration returns bool and the current nested iteration returns
        // false stop the whole nested for loop by recursively returning false
        const auto cont = for_loop_impl<Int, N, I - 1, I>{
            status_copy, m_ends}(std::forward<Iteration>(iteration));
        assert(cont && "cannot break in parallel loop");
      }
    }
    if constexpr (returns_bool) {
      // return true if iteration never returned false
      return true;
    }
  }

 public:
  template <typename Iteration>
  auto operator()(Iteration&& iteration) const {
    return loop(std::forward<Iteration>(iteration),
                std::make_index_sequence<N>{});
  }
};
//------------------------------------------------------------------------------
/// \brief Last nesting of nested for loops, I is 1 and also reached parallel
///        state.
/// \tparam Int integer type for counting
/// \tparam N number of nestings
template <typename Int, std::size_t N>
struct for_loop_impl<Int, N, 1, 1> {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<Int, N>&       m_status;
  const std::array<Int, N>& m_ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
#ifdef __cpp_concepts
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
#else
  template <
      std::size_t... Is, typename Iteration,
      enable_if<is_invocable<Iteration, decltype(((void)Is, Int{}))...>> = true>
#endif
  auto loop(Iteration&& iteration,
            std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = 0; i < m_ends[0]; ++i) {
      auto status_copy = m_status;
      status_copy[0]   = i;
      if constexpr (returns_void) {
        // if if returns nothing just call it
        iteration(status_copy[Is]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        const auto cont = iteration(status_copy[Is]...);
        assert(cont && "cannot break in parallel loop");
      }
    }
    if constexpr (returns_bool) {
      // return true if iteration never returned false
      return true;
    }
  }

 public:
  template <typename Iteration>
  auto operator()(Iteration&& iteration) const {
    return loop(std::forward<Iteration>(iteration),
                std::make_index_sequence<N>{});
  }
};
#endif  // TATOOINE_OPENMP_AVAILABLE
//==============================================================================
#ifdef __cpp_concepts
template <std::size_t ParallelIndex, typename Int, Int... Is, integral... Ends,
          invocable<decltype(((void)Is, Int{}))...> Iteration>
#else
template <
    std::size_t ParallelIndex, typename Int, Int... Is, typename Iteration,
    typename... Ends, enable_if<is_integral<Ends...>> = true,
    enable_if<is_invocable<Iteration, decltype(((void)Is, Int{}))...>> = true>
#endif
constexpr auto for_loop(Iteration&& iteration,
                        std::integer_sequence<Int, Is...>,
                        Ends const... ends) {
  // check if Iteration either returns bool or nothing
  using return_type =
      std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
  constexpr bool returns_void = std::is_same_v<return_type, void>;
  constexpr bool returns_bool = std::is_same_v<return_type, bool>;
  static_assert(returns_void || returns_bool);

  std::array zeros{((void)Is, Int(0))...};
  std::array ends_arr{static_cast<Int>(ends)...};
  return for_loop_impl<Int, sizeof...(ends), sizeof...(ends),
                       ParallelIndex + 1>{
      zeros, ends_arr}(std::forward<Iteration>(iteration));
}
//==============================================================================
}  // namespace detail
//==============================================================================
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
#ifdef __cpp_concepts
template <typename Int = std::size_t, typename Iteration, integral... Ends>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ends,
          enable_if<is_integral<Ends...>> = true>
#endif
constexpr void for_loop(Iteration&& iteration, Ends const... ends) {
  detail::for_loop<sizeof...(ends) + 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ends)>{},
      ends...);
}
//------------------------------------------------------------------------------
/// \brief Use this function for creating a parallel nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
#ifdef __cpp_concepts
template <typename Int = std::size_t, typename Iteration, integral... Ends>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ends,
          enable_if<is_integral<Ends...>> = true>
#endif
constexpr void parallel_for_loop(Iteration&& iteration,
                                 Ends const... ends) {
#ifdef _OPENMP
  return detail::for_loop<sizeof...(ends) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ends)>{}, ends...);
#else
#pragma message \
    "Not able to execute nested for loop in parallel because OpenMP is not available."
  return for_loop(std::forward<Iteration>(iteration), ends...);
#endif  // _OPENMP
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif  // NESTED_FOR_LOOP_H
