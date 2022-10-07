#ifndef TATOOINE_FOR_LOOP_H
#define TATOOINE_FOR_LOOP_H
//==============================================================================
#include <tatooine/available_libraries.h>
#include <tatooine/cache_alignment.h>
#include <tatooine/concepts.h>
#include <tatooine/tags.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>

#include <array>
#include <boost/range/algorithm/transform.hpp>
#include <vector>
#if TATOOINE_OPENMP_AVAILABLE
#include <omp.h>
#define TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE 1
#else
#define TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE 0
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
static constexpr bool parallel_for_loop_support = true;
#else
static constexpr bool parallel_for_loop_support = false;
#endif

#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
template <typename T>
auto create_aligned_data_for_parallel() {
  auto num_threads = std::size_t{};
#pragma omp parallel
  {
    if (omp_get_thread_num()) {
      num_threads = static_cast<std::size_t>(omp_get_num_threads());
    }
  }
  return std::vector<aligned<T>>(num_threads);
}
#endif

//==============================================================================
namespace detail::for_loop {
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
  template <std::size_t... IndexSequence,
            invocable<decltype(((void)IndexSequence, Int{}))...> Iteration>
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<IndexSequence...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration,
                             decltype(((void)IndexSequence, Int{}))...>;
    constexpr bool returns_void = is_same<return_type, void>;
    constexpr bool returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);
    m_status[I - 1] = m_begins[I - 1];
    for (; m_status[I - 1] < m_ends[I - 1]; ++m_status[I - 1]) {
      if constexpr (returns_void) {
        // if returns nothing just create another nested loop
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
  template <std::size_t... IndexSequence,
            invocable<decltype(((void)IndexSequence, Int{}))...> Iteration>
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<IndexSequence...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration,
                             decltype(((void)IndexSequence, Int{}))...>;
    constexpr auto returns_void = is_same<return_type, void>;
    constexpr auto returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);

    m_status[0] = m_begins[0];
    for (; m_status[0] < m_ends[0]; ++m_status[0]) {
      if constexpr (returns_void) {
        // if returns nothing just call it
        iteration(m_status[IndexSequence]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        if (!iteration(m_status[IndexSequence]...)) {
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
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
//==============================================================================
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
  template <std::size_t... IndexSequence,
            invocable<decltype(((void)IndexSequence, Int{}))...> Iteration>
  auto loop(Iteration&& iteration,
            std::index_sequence<IndexSequence...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration,
                             decltype(((void)IndexSequence, Int{}))...>;
    static constexpr auto returns_void = is_same<return_type, void>;
    static constexpr auto returns_bool = is_same<return_type, bool>;
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
  template <std::size_t... IndexSequence,
            invocable<decltype(((void)IndexSequence, Int{}))...> Iteration>
  auto loop(Iteration&& iteration,
            std::index_sequence<IndexSequence...> /*seq*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration,
                             decltype(((void)IndexSequence, Int{}))...>;
    constexpr bool returns_void = is_same<return_type, void>;
    constexpr bool returns_bool = is_same<return_type, bool>;
    static_assert(returns_void || returns_bool);

#pragma omp parallel for
    for (Int i = m_begins[0]; i < m_ends[0]; ++i) {
      auto status_copy = m_status;
      status_copy[0]   = i;
      if constexpr (returns_void) {
        // if if returns nothing just call it
        iteration(status_copy[IndexSequence]...);
      } else {
        // if iteration returns bool and the current iteration returns false
        // stop the whole nested for loop by recursively returning false
        auto const cont = iteration(status_copy[IndexSequence]...);
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
#endif  // TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
//==============================================================================
template <std::size_t ParallelIndex, typename Int, Int... IndexSequence,
          integral... Ranges,
          invocable<decltype(((void)IndexSequence, Int{}))...> Iteration>
constexpr auto for_loop(Iteration&& iteration,
                        std::integer_sequence<Int, IndexSequence...>,
                        std::pair<Ranges, Ranges> const&... ranges) {
  // check if Iteration either returns bool or nothing
  using return_type =
      std::invoke_result_t<Iteration,
                           decltype(((void)IndexSequence, Int{}))...>;
  constexpr bool returns_void = is_same<return_type, void>;
  constexpr bool returns_bool = is_same<return_type, bool>;
  static_assert(returns_void || returns_bool);

  auto status =
      std::array{((void)IndexSequence, static_cast<Int>(ranges.first))...};
  auto const begins =
      std::array{((void)IndexSequence, static_cast<Int>(ranges.first))...};
  auto const ends = std::array{static_cast<Int>(ranges.second)...};
  return for_loop_impl<Int, sizeof...(ranges), sizeof...(ranges),
                       ParallelIndex + 1>{
      status, begins, ends}(std::forward<Iteration>(iteration));
}
//==============================================================================
}  // namespace detail::for_loop
//==============================================================================
template <typename Iteration, typename Range>
concept for_loop_range_iteration =
    std::invocable<Iteration, std::ranges::range_value_t<Range>>
    && either_of<std::invoke_result_t<Iteration, std::ranges::range_value_t<Range>>,
                 void, bool>;
//------------------------------------------------------------------------------
template <typename Iteration, typename IntRange>
concept for_loop_nested_index_iteration =
    std::invocable<Iteration, IntRange> &&
    either_of<std::invoke_result_t<Iteration, IntRange>,
             void, bool>;
//------------------------------------------------------------------------------
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        Ranges (&&... ranges)[2]) -> void {
  detail::for_loop::for_loop<sizeof...(ranges) + 1, Int>(
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        std::pair<Ranges, Ranges> const&... ranges) -> void {
  detail::for_loop::for_loop<sizeof...(ranges) + 1, Int>(
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
template <typename Int = std::size_t, typename Iteration, integral... Ends>
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        Ends const... ends) -> void {
  detail::for_loop::for_loop<sizeof...(ends) + 1, Int>(
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop([[maybe_unused]] Iteration&& iteration,
                        execution_policy::parallel_t,
                        [[maybe_unused]] Ranges (&&... ranges)[2])
    -> void requires parallel_for_loop_support {
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
  return detail::for_loop::for_loop<sizeof...(ranges) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ranges)>{},
      std::pair{ranges[0], ranges[1]}...);
#endif
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Use this function for creating a parallel nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop([[maybe_unused]] Iteration&& iteration,
                        execution_policy::parallel_t /*policy*/,
                        [[maybe_unused]] std::pair<Ranges, Ranges> const&... ranges)
    -> void requires parallel_for_loop_support {
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
  return detail::for_loop::for_loop<sizeof...(ranges) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ranges)>{}, ranges...);
#endif
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Use this function for creating a parallel nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, integral... Ends>
constexpr auto for_loop([[maybe_unused]] Iteration&& iteration,
                        execution_policy::parallel_t /*policy*/,
                        [[maybe_unused]] Ends const... ends)
    -> void requires parallel_for_loop_support {
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
  return detail::for_loop::for_loop<sizeof...(ends) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ends)>{},
      std::pair{Ends(0), ends}...);
#endif
}
//==============================================================================
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop(Iteration&& iteration, Ranges (&&... ranges)[2])
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop(Iteration&& iteration,
                        std::pair<Ranges, Ranges> const&... ranges) -> void {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential,
           ranges...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, integral... Ends>
constexpr auto for_loop(Iteration&& iteration, Ends const... ends) -> void {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential,
           ends...);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Iteration, integral Int, std::size_t N>
auto for_loop_unpacked(Iteration&& iteration, execution_policy_tag auto policy,
                       std::array<Int, N> const& sizes) {
  invoke_unpacked(
      [&](auto const... is) {
        for_loop(std::forward<Iteration>(iteration), policy, is...);
      },
      unpack(sizes));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Iteration, integral Int, std::size_t N>
auto for_loop_unpacked(Iteration&& iteration, std::array<Int, N> const& sizes) {
  for_loop_unpacked(std::forward<Iteration>(iteration),
                    execution_policy::sequential, sizes);
}
//==============================================================================
template <typename Int = std::size_t, integral... Ends>
constexpr auto chunked_for_loop(
    invocable<decltype(((void)std::declval<Ends>(),
                        std::declval<Int>()))...> auto&& iteration,
    execution_policy_tag auto policy, integral auto const chunk_size,
    Ends const... ends) -> void {
  for_loop(
      [&](auto const... chunk_is) {
        for_loop(
            [&](auto const... inner_is) {
              iteration((chunk_is * chunk_size + inner_is)...);
            },
            policy,
            std::min<Int>(chunk_size, chunk_is * chunk_size - ends)...);
      },
      ends / chunk_size...);
}
//------------------------------------------------------------------------------
template <typename Int = std::size_t, typename Iteration, integral... Ends>
requires invocable<Iteration,
                   decltype((std::declval<Ends>(), std::declval<Int>()))...>
constexpr auto chunked_for_loop(Iteration&&         iteration,
                                integral auto const chunk_size,
                                Ends const... ends) -> void {
  chunked_for_loop(std::forward<Iteration>(iteration),
                   execution_policy::sequential, chunk_size, ends...);
}
//==============================================================================
/// Sequential nested loop over index ranges. Pairs describe begin and ends of
/// single ranges. Second element of pair is excluded.
template <integral_pair_range IntPairRange,
          for_loop_nested_index_iteration<std::vector<common_type<
              typename std::ranges::range_value_t<IntPairRange>::first_type,
              typename std::ranges::range_value_t<IntPairRange>::second_type>>>
              Iteration>
auto for_loop(Iteration&& iteration, IntPairRange const& ranges,
              execution_policy::sequential_t) {
  using integral_pair_t = std::ranges::range_value_t<IntPairRange>;
  using int_t           = common_type<typename integral_pair_t::first_type,
                                      typename integral_pair_t::second_type>;
  using iteration_invoke_result_type =
      std::invoke_result_t<Iteration, std::vector<int_t>>;
  auto cur_indices = std::vector<int_t>(size(ranges));
  std::transform(
      begin(ranges), end(ranges), begin(cur_indices),
      [](auto const& range) { return static_cast<int_t>(range.first); });
  auto finished = false;
  while (!finished) {
    if constexpr (same_as<bool, iteration_invoke_result_type>) {
      auto const can_continue = iteration(cur_indices);
      if (!can_continue) {
        finished = true;
      }
    } else {
      iteration(cur_indices);
    }
    ++cur_indices.front();
    for (std::size_t i = 0; i < size(ranges) - 1; ++i) {
      if (cur_indices[i] == static_cast<int_t>(ranges[i].second)) {
        cur_indices[i] = static_cast<int_t>(ranges[i].first);
        ++cur_indices[i + 1];
        if (i == size(ranges) - 2 &&
            cur_indices[i + 1] == static_cast<int_t>(ranges[i + 1].second)) {
          finished = true;
        }
      } else {
        break;
      }
    }
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Either sequential or parallel nested loop over index ranges. Pairs describe
/// begin and ends of single ranges. Second element of pair is excluded.
template <integral_pair_range IntPairRange,
          for_loop_nested_index_iteration<std::vector<common_type<
              typename std::ranges::range_value_t<IntPairRange>::first_type,
              typename std::ranges::range_value_t<IntPairRange>::second_type>>>
              Iteration>
auto for_loop(Iteration&& iteration, IntPairRange const& ranges,
              execution_policy_tag auto policy) {
  for_loop(std::forward<Iteration>(iteration), ranges, policy);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Sequential nested loop over index ranges. Pairs describe begin and ends of
/// single ranges. Second element of pair is excluded.
template <integral_pair_range IntPairRange,
          for_loop_nested_index_iteration<std::vector<common_type<
              typename std::ranges::range_value_t<IntPairRange>::first_type,
              typename std::ranges::range_value_t<IntPairRange>::second_type>>>
              Iteration>
auto for_loop(Iteration&& iteration, IntPairRange const& ranges) {
  for_loop(std::forward<Iteration>(iteration), ranges,
           execution_policy::sequential);
}
//------------------------------------------------------------------------------
/// Sequential nested loop over index ranges from [begins] to [ends], excluding
/// indices that are equal to elements of ends.
template <integral_range                            IntRange,
          for_loop_nested_index_iteration<IntRange> Iteration>
auto for_loop(Iteration&& iteration, IntRange const& begin,
              IntRange const& ends, execution_policy::sequential_t)
/*-> std::invoke_result_t<Iteration, decltype(status)>*/ {
  assert(size(begin) == size(ends));
  auto const nesting_depth = size(ends);
  using int_t              = std::ranges::range_value_t<IntRange>;
  using iteration_invoke_result_type =
      std::invoke_result_t<Iteration, std::vector<int_t>>;
  auto cur_indices = begin;
  auto finished    = false;
  while (!finished) {
    if constexpr (same_as<bool, iteration_invoke_result_type>) {
      auto const can_continue = iteration(cur_indices);
      if (!can_continue) {
        finished = true;
      }
    } else {
      iteration(cur_indices);
    }
    ++cur_indices.front();
    for (std::size_t i = 0; i < nesting_depth - 1; ++i) {
      if (cur_indices[i] == ends[i]) {
        cur_indices[i] = begin[i];
        ++cur_indices[i + 1];
        if (i == nesting_depth - 2 && cur_indices[i + 1] == ends[i + 1]) {
          finished = true;
        }
      } else {
        break;
      }
    }
  }
}
//------------------------------------------------------------------------------
/// Sequential nested loop over index ranges from [begins] to [ends], excluding
/// indices that are equal to elements of ends.
template <integral_range                            IntRange,
          for_loop_nested_index_iteration<IntRange> Iteration>
auto for_loop(Iteration&& iteration, IntRange const& begin,
              IntRange const& ends) {
  return for_loop(std::forward<Iteration>(iteration), begin, ends,
                  execution_policy::sequential);
}
//------------------------------------------------------------------------------
/// Either sequential or parallel nested loop over index ranges from [0,..,0] to
/// [ends], excluding indices that are equal to elements of ends.
template <integral_range                            IntRange,
          for_loop_nested_index_iteration<IntRange> Iteration>
auto for_loop(Iteration&& iteration, IntRange const& ends,
              execution_policy_tag auto policy) {
  using int_t = std::ranges::range_value_t<IntRange>;
  for_loop<int_t>(std::forward<Iteration>(iteration),
                  std::vector<int_t>(size(ends), 0), ends, policy);
}
//------------------------------------------------------------------------------
/// Sequential nested loop over index ranges from [0,..,0]  to
/// [ends], excluding indices that are equal to elements of ends.
template <integral_range                            IntRange,
          for_loop_nested_index_iteration<IntRange> Iteration>
auto for_loop(Iteration&& iteration, IntRange const& ends) {
  using int_t = std::ranges::range_value_t<IntRange>;
  for_loop(std::forward<Iteration>(iteration),
           std::vector<int_t>(size(ends), 0), ends,
           execution_policy::sequential);
}
//------------------------------------------------------------------------------
/// Sequential nested loop over a generic range.
template <range Range, for_loop_range_iteration<Range> Iteration>
requires (!integral<std::ranges::range_value_t<Range>>) &&
         (!integral_pair<std::ranges::range_value_t<Range>>)
auto for_loop([[maybe_unused]] Iteration&& iteration,
              [[maybe_unused]] Range const& r,
              execution_policy::parallel_t /*seq*/)
requires parallel_for_loop_support {
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
#pragma omp parallel for
  for (auto const& elem : r) {
    iteration(elem);
  }
#endif
}
//------------------------------------------------------------------------------
/// Sequential nested loop over a generic range.
template <range Range, for_loop_range_iteration<Range> Iteration>
requires (!integral<std::ranges::range_value_t<Range>>) &&
         (!integral_pair<std::ranges::range_value_t<Range>>)
auto for_loop(Iteration&& iteration, Range const& r,
              execution_policy::sequential_t /*seq*/) {
  using iteration_invoke_result_type =
      std::invoke_result_t<Iteration, std::ranges::range_value_t<Range>>;
  for (auto const& elem : r) {
    if constexpr (same_as<bool, iteration_invoke_result_type>) {
      auto const can_continue = iteration(elem);
      if (!can_continue) {
        break;
      }
    } else {
      iteration(elem);
    }
  }
}
//------------------------------------------------------------------------------
/// Sequential loop over a generic range.
template <range Range, for_loop_range_iteration<Range> Iteration>
requires (!integral<std::ranges::range_value_t<Range>>) &&
         (!integral_pair<std::ranges::range_value_t<Range>>)
auto for_loop(Iteration && iteration, Range const& r) {
  for_loop(std::forward<Iteration>(iteration), r, execution_policy::sequential);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
