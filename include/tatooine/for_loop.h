#ifndef TATOOINE_FOR_LOOP_H
#define TATOOINE_FOR_LOOP_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/packages.h>
#include <tatooine/tags.h>

#include <array>

#include <tatooine/type_traits.h>
#include <tatooine/utility.h>
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
/// \tparam ParallelIndex If I and ParallelIndex are the same and OpenMP is
///                       available then the current nested loop will be
///                       executed in parallel.
template <typename Int, std::size_t N, std::size_t I, std::size_t ParallelIndex>
struct for_loop_impl {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<Int, N>&       m_status;
  std::array<Int, N> const& m_begins;
  std::array<Int, N> const& m_ends;

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
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr bool returns_void = is_same<return_type, void>;
    constexpr bool returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);
    m_status[I - 1] = m_begins[I - 1];
    for (; m_status[I - 1] < m_ends[I - 1]; ++m_status[I - 1]) {
      if constexpr (returns_void) {
        // if if returns nothing just create another nested loop
        for_loop_impl<Int, N, I - 1, ParallelIndex>{
            m_status, m_begins, m_ends}(std::forward<Iteration>(iteration));
      } else {
        // if iteration returns bool and the current nested iteration returns
        // false stop the whole nested for loop by recursively returning false
        if (!for_loop_impl<Int, N, I - 1, ParallelIndex>{
                m_status, m_begins,
                m_ends}(std::forward<Iteration>(iteration))) {
          return false;
        }
      }
      // reset nested status
      m_status[I - 2] = m_begins[I - 2];
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
  std::array<Int, N> const& m_begins;
  std::array<Int, N> const& m_ends;

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
    constexpr bool returns_void = is_same<return_type, void>;
    constexpr bool returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);

    m_status[0] = m_begins[0];
    for (; m_status[0] < m_ends[0]; ++m_status[0]) {
      if constexpr (returns_void) {
        // if returns nothing just call it
        iteration(m_status[Is]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        if (!iteration(m_status[Is]...)) {
          return false;
        }
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
  std::array<Int, N> const& m_begins;
  std::array<Int, N> const& m_ends;

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
    constexpr bool returns_void = is_same<return_type, void>;
    constexpr bool returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = m_begins[I - 1]; i < m_ends[I - 1]; ++i) {
      auto status_copy   = m_status;
      status_copy[I - 1] = i;
      if constexpr (returns_void) {
        // if if returns nothing just create another nested loop
        for_loop_impl<Int, N, I - 1, I>{
            status_copy, m_begins, m_ends}(std::forward<Iteration>(iteration));
      } else {
        // if iteration returns bool and the current nested iteration returns
        // false stop the whole nested for loop by recursively returning false
        auto const cont = for_loop_impl<Int, N, I - 1, I>{
            status_copy, m_begins, m_ends}(std::forward<Iteration>(iteration));
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
  std::array<Int, N> const& m_begins;
  std::array<Int, N> const& m_ends;

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
    constexpr bool returns_void = is_same<return_type, void>;
    constexpr bool returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = m_begins[0]; i < m_ends[0]; ++i) {
      auto status_copy = m_status;
      status_copy[0]   = i;
      if constexpr (returns_void) {
        // if if returns nothing just call it
        iteration(status_copy[Is]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        auto const cont = iteration(status_copy[Is]...);
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
template <std::size_t ParallelIndex, typename Int, Int... Is,
          integral... Ranges,
          invocable<decltype(((void)Is, Int{}))...> Iteration>
#else
template <
    std::size_t ParallelIndex, typename Int, Int... Is, typename Iteration,
    typename... Ranges, enable_if_integral<Ranges...> = true,
    enable_if<is_invocable<Iteration, decltype(((void)Is, Int{}))...>> = true>
#endif
constexpr auto for_loop(Iteration&& iteration,
                        std::integer_sequence<Int, Is...>,
                        std::pair<Ranges, Ranges> const&... ranges) {
  // check if Iteration either returns bool or nothing
  using return_type =
      std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
  constexpr bool returns_void = is_same<return_type, void>;
  constexpr bool returns_bool = is_same<return_type, bool>;
  static_assert(returns_void || returns_bool);

  auto       status = std::array{((void)Is, static_cast<Int>(ranges.first))...};
  auto const begins = std::array{((void)Is, static_cast<Int>(ranges.first))...};
  auto const ends   = std::array{static_cast<Int>(ranges.second)...};
  return for_loop_impl<Int, sizeof...(ranges), sizeof...(ranges),
                       ParallelIndex + 1>{
      status, begins, ends}(std::forward<Iteration>(iteration));
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ranges,
          enable_if_integral<Ranges...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        Ranges(&&... ranges)[2]) -> void {
  detail::for_loop<sizeof...(ranges) + 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ranges)>{},
      std::pair{ranges[0], ranges[1]}...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
#ifdef __cpp_concepts
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ranges,
          enable_if_integral<Ranges...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        std::pair<Ranges, Ranges> const&... ranges) -> void {
  detail::for_loop<sizeof...(ranges) + 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ranges)>{}, ranges...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
          enable_if_integral<Ends...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        Ends const... ends) -> void {
  detail::for_loop<sizeof...(ends) + 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ends)>{},
      std::pair{Ends(0), ends}...);
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ranges,
          enable_if_integral<Ranges...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, execution_policy::parallel_t,
                        Ranges(&&... ranges)[2]) -> void {
#ifdef _OPENMP
  return detail::for_loop<sizeof...(ranges) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ranges)>{},
      std::pair{ranges[0], ranges[1]}...);
#else
  //#pragma message "Not able to execute nested for loop in parallel because
  // OpenMP is not available."
  return for_loop(std::forward<Iteration>(iteration), ranges...);
#endif  // _OPENMP
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Use this function for creating a parallel nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
#ifdef __cpp_concepts
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ranges,
          enable_if_integral<Ranges...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, execution_policy::parallel_t,
                        std::pair<Ranges, Ranges> const&... ranges) -> void {
#ifdef _OPENMP
  return detail::for_loop<sizeof...(ranges) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ranges)>{}, ranges...);
#else
  //#pragma message "Not able to execute nested for loop in parallel because
  // OpenMP is not available."
  return for_loop(std::forward<Iteration>(iteration), ranges...);
#endif  // _OPENMP
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
          enable_if_integral<Ends...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, execution_policy::parallel_t,
                        Ends const... ends) -> void {
#ifdef _OPENMP
  return detail::for_loop<sizeof...(ends) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ends)>{},
      std::pair{Ends(0), ends}...);
#else
  //#pragma message "Not able to execute nested for loop in parallel because
  // OpenMP is not available."
  return for_loop(std::forward<Iteration>(iteration),
                  std::pair{Ends(0), ends}...);
#endif  // _OPENMP
}
//==============================================================================
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
#ifdef __cpp_concepts
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ranges,
          enable_if_integral<Ranges...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, Ranges(&&... ranges)[2])
    -> void {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential,
           std::pair{ranges[0], ranges[1]}...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
#ifdef __cpp_concepts
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
#else
template <typename Int = std::size_t, typename Iteration, typename... Ranges,
          enable_if_integral<Ranges...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration,
                        std::pair<Ranges, Ranges> const&... ranges) -> void {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential, ranges...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
          enable_if_integral<Ends...> = true>
#endif
constexpr auto for_loop(Iteration&& iteration, Ends const... ends) -> void {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential, ends...);
}
//==============================================================================
/// dynamically-sized for loop
template <typename Iteration>
auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
              std::vector<std::pair<size_t, size_t>> const& ranges) {
  auto cur_indices = std::vector<size_t>(size(ranges));
  std::transform(begin(ranges), end(ranges), begin(cur_indices),
                 [](auto const& range) { return range.first; });
  bool finished = false;
  while (!finished) {
    iteration(cur_indices);
    ++cur_indices.front();
    for (size_t i = 0; i < size(ranges) - 1; ++i) {
      if (cur_indices[i] == ranges[i].second) {
        cur_indices[i] = ranges[i].first;
        ++cur_indices[i + 1];
        if (i == size(ranges) - 2 &&
            cur_indices[i + 1] == ranges[i + 1].second) {
          finished = true;
        }
      } else {
        break;
      }
    }
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// dynamically-sized for loop
template <typename Iteration>
auto for_loop(Iteration&&                                   iteration,
              std::vector<std::pair<size_t, size_t>> const& ranges) {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential, ranges);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
