#ifndef TATOOINE_RANDOM_H
#define TATOOINE_RANDOM_H
//==============================================================================
#include <boost/range/algorithm/generate.hpp>
#include <random>

#include "type_traits.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Engine = std::mt19937_64>
struct random_uniform {
  static_assert(is_arithmetic<T>);
  //============================================================================
  using engine_t = Engine;
  using real_t   = T;
  using distribution_t =
      std::conditional_t<std::is_floating_point<T>::value,
                         std::uniform_real_distribution<T>,
                         std::uniform_int_distribution<T>>;
  //============================================================================
 private:
  Engine         engine;
  distribution_t distribution;
  //============================================================================
 public:
  random_uniform()
      : engine{Engine{std::random_device{}()}}, distribution{T(0), T(1)} {}

  random_uniform(const random_uniform&)     = default;
  random_uniform(random_uniform&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const random_uniform&) noexcept ->random_uniform& = default;
  auto operator=(random_uniform&&) noexcept -> random_uniform& = default;
  //----------------------------------------------------------------------------
  ~random_uniform() = default;
  //----------------------------------------------------------------------------
  explicit random_uniform(Engine _engine)
      : engine{_engine}, distribution{T(0), T(1)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(T min, T max, Engine _engine = Engine{std::random_device{}()})
      : engine{_engine}, distribution{min, max} {}
  //============================================================================
  auto get() { return distribution(engine); }
  auto operator()() { return get(); }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
random_uniform()->random_uniform<double, std::mt19937_64>;

// copy when having rvalue
template <typename Engine>
random_uniform(Engine &&)->random_uniform<double, Engine>;

// keep reference when having lvalue
template <typename Engine>
random_uniform(const Engine&)->random_uniform<double, const Engine&>;

// copy when having rvalue
template <typename T, typename Engine>
random_uniform(T min, T max, Engine &&)->random_uniform<T, Engine>;

// keep reference when having lvalue
template <typename T, typename Engine>
random_uniform(T min, T max, const Engine&)->random_uniform<T, const Engine&>;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T, typename Engine = std::mt19937_64>
auto random_uniform_vector(size_t n, T a = T(0), T b = T(1),
                           Engine&& engine = Engine{std::random_device{}()}) {
  random_uniform rand(a, b, std::forward<Engine>(engine));

  std::vector<T> rand_data(n);
  boost::generate(rand_data, [&] { return rand(); });
  return rand_data;
}
//==============================================================================
template <typename T, typename Engine = std::mt19937_64>
struct random_normal {
  using engine_t       = Engine;
  using real_t         = T;
  using distribution_t = std::normal_distribution<T>;

  //============================================================================
  random_normal() : engine{std::random_device{}()}, distribution{0, 1} {}
  random_normal(const random_normal&)     = default;
  random_normal(random_normal&&) noexcept = default;
  //============================================================================
  auto operator=(const random_normal&) -> random_normal& = default;
  auto operator=(random_normal&&) noexcept -> random_normal& = default;
  //----------------------------------------------------------------------------
  ~random_normal() = default;

  //----------------------------------------------------------------------------
  explicit random_normal(const Engine& _engine)
      : engine{_engine}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  explicit random_normal(Engine&& _engine)
      : engine{std::move(_engine)}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(T mean, T stddev)
      : engine{std::random_device{}()}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(const Engine& _engine, T mean, T stddev)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Engine&& _engine, T mean, T stddev)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(T mean, T stddev, const Engine& _engine)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(T mean, T stddev, Engine&& _engine)
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
random_normal()->random_normal<double, std::mt19937_64>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
random_normal(Engine &&)->random_normal<double, Engine>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
random_normal(T mean, T stddev)->random_normal<T, std::mt19937_64>;
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
