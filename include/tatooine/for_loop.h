#ifndef TATOOINE_FOR_LOOP_H
#define TATOOINE_FOR_LOOP_H
//==============================================================================
#include <tatooine/cache_alignment.h>
#include <vector>
#include <tatooine/concepts.h>
#include <tatooine/available_libraries.h>
#include <tatooine/tags.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>

#include <array>
#include <boost/range/algorithm/transform.hpp>
#if TATOOINE_OPENMP_AVAILABLE
#include <omp.h>
#endif
//==============================================================================
namespace tatooine {
template <typename T>
auto create_aligned_data_for_parallel() {
  auto num_threads = std::size_t{};
#pragma omp parallel
  {
    if (omp_get_thread_num()) {
      num_threads = omp_get_num_threads();
    }
  }
  return std::vector<aligned<T>>(num_threads);
}
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
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
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
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
  constexpr auto loop(Iteration&& iteration,
                      std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr auto returns_void = is_same<return_type, void>;
    constexpr auto returns_bool = is_same<return_type, bool>;
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
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
  auto loop(Iteration&& iteration,
            std::index_sequence<Is...> /*unused*/) const {
    // check if Iteration either returns bool or nothing
    using return_type =
        std::invoke_result_t<Iteration, decltype(((void)Is, Int{}))...>;
    constexpr auto returns_void = is_same<return_type, void>;
    constexpr auto returns_bool = is_same<return_type, bool>;
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
  template <std::size_t... Is,
            invocable<decltype(((void)Is, Int{}))...> Iteration>
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
template <std::size_t ParallelIndex, typename Int, Int... Is,
          integral... Ranges,
          invocable<decltype(((void)Is, Int{}))...> Iteration>
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
}  // namespace detail::for_loop
//==============================================================================
/// \brief Use this function for creating a sequential nested loop.
///
/// First Index grows fastest, then the second and so on.
///
/// iteration must either return bool or nothing. If iteration returns false in
/// any state the whole nested iteration will stop. iteration must return true
/// to continue.
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop(Iteration&& iteration, execution_policy::sequential_t,
                        Ranges(&&... ranges)[2]) -> void {
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
constexpr auto for_loop(Iteration&& iteration, execution_policy::parallel_t,
                        Ranges(&&... ranges)[2]) -> void {
#ifdef _OPENMP
  return detail::for_loop::for_loop<sizeof...(ranges) - 1, Int>(
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
constexpr auto for_loop(Iteration&& iteration,
                        execution_policy::parallel_t /*policy*/,
                        std::pair<Ranges, Ranges> const&... ranges) -> void {
#ifdef _OPENMP
  return detail::for_loop::for_loop<sizeof...(ranges) - 1, Int>(
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
template <typename Int = std::size_t, typename Iteration, integral... Ends>
constexpr auto for_loop(Iteration&& iteration,
                        execution_policy::parallel_t /*policy*/,
                        Ends const... ends) -> void
// requires invocable<Iteration, decltype((Ends, Int{}))...>
{
#ifdef _OPENMP
  return detail::for_loop::for_loop<sizeof...(ends) - 1, Int>(
      std::forward<Iteration>(iteration),
      std::make_integer_sequence<Int, sizeof...(ends)>{},
      std::pair{Ends(0), ends}...);
#else
  static_assert(false,
                "OpenMP is not available. Cannot execute parallel for loop");
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
template <typename Int = std::size_t, typename Iteration, integral... Ranges>
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
template <typename Iteration, typename ExecutionPolicy, integral Int,
          std::size_t N>
auto for_loop_unpacked(Iteration&& iteration, ExecutionPolicy pol,
                       std::array<Int, N> const& sizes) {
  invoke_unpacked(
      [&](auto const... is) {
        for_loop(std::forward<Iteration>(iteration), pol, is...);
      },
      unpack(sizes));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Iteration, integral Int, std::size_t N>
auto for_loop_unpacked(Iteration&& iteration, std::array<Int, N> const& sizes) {
  for_loop_unpacked(std::forward<Iteration>(iteration),
                    execution_policy::sequential, sizes);
}
//------------------------------------------------------------------------------
template <typename Iteration>
auto        for_loop(Iteration&&       iteration, execution_policy::parallel_t,
                     range auto const& r) {
#pragma omp parallel for
  for (auto it = begin(r); it < end(r); it++) {
    iteration(*it);
  }
}
//------------------------------------------------------------------------------
template <typename Iteration>
auto for_loop(Iteration&&       iteration, execution_policy::sequential_t,
              range auto const& r) {
  for (auto const& s : r) {
    iteration(s);
  }
}
//------------------------------------------------------------------------------
template <typename Iteration>
auto for_loop(Iteration&& iteration, range auto const& r) {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential, r);
}
//==============================================================================
template <typename Int = std::size_t, integral... Ends>
constexpr auto chunked_for_loop(
    invocable<decltype(((void)std::declval<Ends>(),
                        std::declval<Int>()))...> auto&& iteration,
    auto exec_policy, integral auto const chunk_size, Ends const... ends)
    -> void requires(
        is_same<execution_policy::parallel_t, decltype(exec_policy)> ||
        is_same<execution_policy::sequential_t, decltype(exec_policy)>) {
  for_loop(
      [&](auto const... chunk_is) {
        for_loop(
            [&](auto const... inner_is) {
              iteration((chunk_is * chunk_size + inner_is)...);
            },
            exec_policy,
            std::min<Int>(chunk_size, chunk_is * chunk_size - ends)...);
      },
      ends / chunk_size...);
}
//------------------------------------------------------------------------------
template <typename Int = std::size_t, integral... Ends>
constexpr auto chunked_for_loop(
    invocable<decltype((std::declval<Ends>(), std::declval<Int>()))...> auto&&
                        iteration,
    integral auto const chunk_size, Ends const... ends) -> void {
  chunked_for_loop(std::forward<decltype(iteration)>(iteration),
                   execution_policy::sequential, chunk_size, ends...);
}
//==============================================================================
/// dynamically-sized for loop
template <typename Iteration>
auto for_loop(
    Iteration&& iteration, execution_policy::sequential_t,
    range_of<std::pair<std::size_t, std::size_t>> auto const& ranges) {
  auto cur_indices = std::vector<std::size_t>(size(ranges));
  std::transform(begin(ranges), end(ranges), begin(cur_indices),
                 [](auto const& range) { return range.first; });
  bool finished = false;
  while (!finished) {
    iteration(cur_indices);
    ++cur_indices.front();
    for (std::size_t i = 0; i < size(ranges) - 1; ++i) {
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
auto for_loop(
    Iteration&&                                               iteration,
    range_of<std::pair<std::size_t, std::size_t>> auto const& ranges) {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential,
           ranges);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// dynamically-sized for loop
template <typename Iteration, typename ExecutionPolicy>
auto for_loop(Iteration&& iteration, ExecutionPolicy pol,
              std::vector<std::size_t> const& sizes) {
  auto ranges = std::vector<std::pair<std::size_t, std::size_t>>(sizes.size());
  boost::transform(sizes, begin(ranges), [](auto const s) {
    return std::pair{std::size_t(0), s};
  });
  for_loop(std::forward<Iteration>(iteration), pol, ranges);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// dynamically-sized for loop
template <typename Iteration>
auto for_loop(Iteration&& iteration, std::vector<std::size_t> const& sizes) {
  for_loop(std::forward<Iteration>(iteration), execution_policy::sequential,
           sizes);
}

//------------------------------------------------------------------------------
template <std::integral Int>
auto for_loop(std::invocable<std::vector<Int>> auto&& iteration,
              std::vector<Int> const& begin, std::vector<Int> const& end,
              std::vector<Int>& status, Int const dim)
    -> std::invoke_result_t<decltype(iteration), decltype(status)>
requires
  std::same_as<std::invoke_result_t<
    decltype(iteration), std::vector<Int>>, void> ||
  std::same_as<std::invoke_result_t<
    decltype(iteration), std::vector<Int>>, bool>
{
  if (static_cast<std::size_t>(dim) == size(begin)) {
    return iteration(status);
  } else {
    for (; status[dim] < end[dim]; ++status[dim]) {
      if constexpr (std::same_as<std::invoke_result_t<decltype(iteration),
                                                      std::vector<int>>,
                                 bool>) {
        auto const can_continue =
            for_loop(iteration, begin, end, status, dim + 1);
        if (!can_continue) {
          return false;
        }
      } else {
        for_loop(std::forward<decltype(iteration)>(iteration), begin, end,
                 status, dim + 1);
      }
    }
    status[dim] = begin[dim];
  }
  if constexpr (std::same_as<
                    std::invoke_result_t<decltype(iteration), std::vector<int>>,
                    bool>) {
    return true;
  }
}
//------------------------------------------------------------------------------
template <std::integral Int>
auto for_loop(std::invocable<std::vector<Int>> auto&& iteration,
              std::vector<Int> const& begin, std::vector<Int> const& end)
requires
  std::same_as<std::invoke_result_t<
    decltype(iteration), std::vector<Int>>, void> ||
  std::same_as<std::invoke_result_t<
    decltype(iteration), std::vector<Int>>, bool>
{
  auto status = std::vector{begin};
  for_loop<Int>(std::forward<decltype(iteration)>(iteration), begin, end,
                status, 0);
}
//------------------------------------------------------------------------------
template <std::integral Int>
auto for_loop(std::invocable<std::vector<Int>> auto&& iteration,
              std::vector<Int> const& end)
requires
  std::same_as<std::invoke_result_t<
    decltype(iteration), std::vector<Int>>, void> ||
  std::same_as<std::invoke_result_t<
    decltype(iteration), std::vector<Int>>, bool>
{
  auto status = std::vector<Int>(size(end), 0);
  for_loop<Int>(std::forward<decltype(iteration)>(iteration),
                std::vector<Int>(size(end), 0), end, status, 0);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
