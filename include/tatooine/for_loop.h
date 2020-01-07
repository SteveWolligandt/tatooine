#ifndef TATOOINE_FOR_LOOP_H
#define TATOOINE_FOR_LOOP_H
//==============================================================================
#include <array>
#include "type_traits.h"
#include "utility.h"
#ifdef _OPENMP
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
  std::array<Int, N>&       status;
  const std::array<Int, N>& ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  /// recursively creates loops
  template <typename Iteration, std::size_t... Is,
            enable_if_invocable<Iteration,
                                     decltype(((void)Is, Int{}))...> = true>
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

    for (; status[I - 1] < ends[I - 1]; ++status[I - 1]) {
      if constexpr (returns_void) {
        // if if returns nothing just create another nested loop
        for_loop_impl<Int, N, I - 1, ParallelIndex>{
            status, ends}(std::forward<Iteration>(iteration));
      } else {
        // if iteration returns bool and the current nested iteration returns
        // false stop the whole nested for loop by recursively returning false
        if (!for_loop_impl<Int, N, I - 1, ParallelIndex>{
                status, ends}(std::forward<Iteration>(iteration))) {
          return false;
        }
      }
      // reset nested status
      status[I - 2] = 0;
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
  std::array<Int, N>&       status;
  const std::array<Int, N>& ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  template <typename Iteration, std::size_t... Is,
            enable_if_invocable<Iteration,
                                     decltype(((void)Is, Int{}))...> = true>
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

    for (; status[0] < ends[0]; ++status[0]) {
      if constexpr (returns_void) {
        // if if returns nothing just call it
        iteration(status[Is]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        if (!iteration(status[Is]...)) { return false; }
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
#ifdef _OPENMP
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
  std::array<Int, N>&       status;
  const std::array<Int, N>& ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  /// recursively creates loops
  template <typename Iteration, std::size_t... Is,
            enable_if_invocable<Iteration,
                                     decltype(((void)Is, Int{}))...> = true>
  auto loop(Iteration&& iteration,
            std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = 0; i < ends[I - 1]; ++i) {
      auto status_copy   = status;
      status_copy[I - 1] = i;
      if constexpr (returns_void) {
        // if if returns nothing just create another nested loop
        for_loop_impl<Int, N, I - 1, I>{
            status_copy, ends}(std::forward<Iteration>(iteration));
      } else {
        // if iteration returns bool and the current nested iteration returns
        // false stop the whole nested for loop by recursively returning false
        const auto cont = for_loop_impl<Int, N, I - 1, I>{
            status_copy, ends}(std::forward<Iteration>(iteration));
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
  std::array<Int, N>&       status;
  const std::array<Int, N>& ends;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  template <typename Iteration, std::size_t... Is,
            enable_if_invocable<Iteration,
                                     decltype(((void)Is, Int{}))...> = true>
  auto loop(Iteration&& iteration,
            std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = std::is_same_v<return_type, void>;
    constexpr bool returns_bool = std::is_same_v<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = 0; i < ends[0]; ++i) {
      auto status_copy = status;
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
#endif  // _OPENMP
//==============================================================================
template <
    std::size_t ParallelIndex, typename Int, typename Iteration,
    typename... Ends, Int... Is,
    enable_if_integral<std::decay_t<Ends>...> = true,
    enable_if_invocable<Iteration, decltype(((void)Is, Int{}))...> = true>
constexpr auto for_loop(Iteration&& iteration,
                          std::integer_sequence<Int, Is...>, Ends&&... ends) {
  // check if Iteration either returns bool or nothing
  using return_type =
      std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
  constexpr bool returns_void = std::is_same_v<return_type, void>;
  constexpr bool returns_bool = std::is_same_v<return_type, bool>;
  static_assert(returns_void || returns_bool);

  std::array zeros{((void)Is, Int(0))...};
  return for_loop_impl<Int, sizeof...(Ends), sizeof...(Ends),
                         ParallelIndex + 1>{
      zeros, std::array{static_cast<Int>(ends)...}}(
      std::forward<Iteration>(iteration));
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
template <typename Int = std::size_t, typename Iteration, typename... Ends,
          enable_if_integral<std::decay_t<Ends>...> = true>
constexpr void for_loop(Iteration&& iteration, Ends&&... ends) {
  detail::for_loop<sizeof...(Ends) + 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(Ends)>{},
      std::forward<Ends>(ends)...);
}
//------------------------------------------------------------------------------
/// \brief Use this function for creating a parallel nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, typename... Ends,
          enable_if_integral<std::decay_t<Ends>...> = true>
constexpr void parallel_for_loop(Iteration&& iteration, Ends&&... ends) {
#ifdef _OPENMP
  return detail::for_loop<sizeof...(Ends) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(Ends)>{},
      std::forward<Ends>(ends)...);
#else
#pragma message \
    "Not able to execute nested for loop in parallel because OpenMP is not available."
  return for_loop(std::forward<Iteration>(iteration),
                    std::forward<Ends>(ends)...);
#endif  // _OPENMP
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif  // NESTED_FOR_LOOP_H
