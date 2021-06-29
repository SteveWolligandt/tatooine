#ifndef TATOOINE_RANDOM_H
#define TATOOINE_RANDOM_H
//==============================================================================
#include <tatooine/real.h>
#include <tatooine/type_traits.h>

#include <boost/range/algorithm/generate.hpp>
#include <random>
//==============================================================================
namespace tatooine::random {
//==============================================================================
template <typename T, typename Engine = std::mt19937_64>
struct uniform {
  static_assert(is_arithmetic<T>);
  //============================================================================
  using engine_t = Engine;
  using real_t   = T;
  using distribution_t =
      std::conditional_t<is_floating_point<T>,
                         std::uniform_real_distribution<T>,
                         std::uniform_int_distribution<T>>;
  //============================================================================
 private:
  Engine         engine;
  distribution_t distribution;
  //============================================================================
 public:
  uniform() : engine{std::random_device{}()}, distribution{T(0), T(1)} {}

  uniform(const uniform&)     = default;
  uniform(uniform&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const uniform&) noexcept ->uniform& = default;
  auto operator=(uniform&&) noexcept -> uniform& = default;
  //----------------------------------------------------------------------------
  ~uniform() = default;
  //----------------------------------------------------------------------------
  template <typename... Args>
  uniform(T const min, T const max, Args&&... args)
      : engine{std::forward<Args>(args)...}, distribution{min, max} {}
  //============================================================================
  auto get() { return distribution(engine); }
  auto operator()() { return get(); }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
uniform()->uniform<real_t, std::mt19937_64>;
template <typename T>
uniform(T const min, T const max) -> uniform<T, std::mt19937_64>;
template <typename T, typename... Args>
uniform(T const min, T const max, Args&&...)
    -> uniform<T, std::mt19937_64>;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T, typename Engine = std::mt19937_64>
auto uniform_vector(size_t n, T a = T(0), T b = T(1),
                           Engine&& engine = Engine{std::random_device{}()}) {
  uniform rand(a, b, std::forward<Engine>(engine));

  std::vector<T> rand_data(n);
  boost::generate(rand_data, [&] { return rand(); });
  return rand_data;
}
//==============================================================================
template <typename T, typename Engine = std::mt19937_64>
struct normal {
  using engine_t       = Engine;
  using real_t         = T;
  using distribution_t = std::normal_distribution<T>;

  //============================================================================
  normal() : engine{std::random_device{}()}, distribution{0, 1} {}
  normal(const normal&)     = default;
  normal(normal&&) noexcept = default;
  //============================================================================
  auto operator=(const normal&) -> normal& = default;
  auto operator=(normal&&) noexcept -> normal& = default;
  //----------------------------------------------------------------------------
  ~normal() = default;

  //----------------------------------------------------------------------------
  explicit normal(const Engine& _engine)
      : engine{_engine}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit normal(Engine&& _engine)
      : engine{std::move(_engine)}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(T mean, T stddev)
      : engine{std::random_device{}()}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(const Engine& _engine, T mean, T stddev)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(Engine&& _engine, T mean, T stddev)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(T mean, T stddev, const Engine& _engine)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(T mean, T stddev, Engine&& _engine)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}

  //============================================================================
 private:
  Engine         engine;
  distribution_t distribution;

  //============================================================================
 public:
  auto get() { return distribution(engine); }
  auto operator()() { return get(); }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
normal()->normal<double, std::mt19937_64>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
normal(Engine &&)->normal<double, Engine>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
normal(T mean, T stddev)->normal<T, std::mt19937_64>;
//==============================================================================
template <typename Iterator, typename RandomEngine>
auto random_elem(Iterator begin, Iterator end, RandomEngine& eng) {
  if (begin == end) { return end; }
  const auto size = static_cast<size_t>(distance(begin, end));
  std::uniform_int_distribution<size_t> rand{0, size - 1};
  return next(begin, rand(eng));
}
//------------------------------------------------------------------------------
template <typename Range, typename RandomEngine>
auto random_elem(Range&& range, RandomEngine& eng) {
  return random_elem(begin(range), end(range), eng);
}
//==============================================================================
enum coin { HEADS, TAILS };
template <typename RandomEngine>
auto flip_coin(RandomEngine&& eng) {
  std::uniform_int_distribution<size_t> coin{0, 1};
  return coin(eng) == 0 ? HEADS : TAILS;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
