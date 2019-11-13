#ifndef TATOOINE_RANDOM_H
#define TATOOINE_RANDOM_H

#include <random>
#include "type_traits.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename T, typename Engine = std::mt19937>
struct random_uniform {
  using engine_t = Engine;
  using real_t   = T;
  using distribution_t =
      std::conditional_t<std::is_floating_point<T>::value,
                         std::uniform_real_distribution<T>,
                         std::uniform_int_distribution<T>>;

  //============================================================================
  random_uniform() : engine{std::random_device{}()}, distribution{0, 1} {}
  random_uniform(const random_uniform&)     = default;
  random_uniform(random_uniform&&) noexcept = default;
  random_uniform& operator=(const random_uniform&) = default;
  random_uniform& operator=(random_uniform&&) noexcept = default;
  //----------------------------------------------------------------------------
  random_uniform(const Engine& _engine) : engine{_engine}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <enable_if_not_arithmetic<Engine> = true>
  random_uniform(Engine&& _engine)
      : engine{std::move(_engine)}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <enable_if_arithmetic<T> = true>
  random_uniform(T min, T max)
      : engine{std::random_device{}()}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <enable_if_arithmetic<T> = true,
            enable_if_not_arithmetic<Engine> = true>
  random_uniform(const Engine& _engine, T min, T max)
      : engine{_engine}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <enable_if_arithmetic<T> = true,
            enable_if_not_arithmetic<Engine> = true>
  random_uniform(Engine&& _engine, T min, T max)
      : engine{std::move(_engine)}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <enable_if_arithmetic<T> = true,
            enable_if_not_arithmetic<Engine> = true>
  random_uniform(T min, T max, const Engine& _engine)
      : engine{_engine}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <enable_if_arithmetic<T> = true,
            enable_if_not_arithmetic<Engine> = true>
  random_uniform(T min, T max, Engine&& _engine)
      : engine{std::move(_engine)}, distribution{min, max} {}

  //============================================================================
 private:
  Engine         engine;
  distribution_t distribution;

  //============================================================================
 public:
  auto get() { return distribution(engine); }
  auto operator()() { return get(); }

  template <typename OtherEngine>
  auto get(OtherEngine& e) {
    return distribution(e);
  }
  template <typename OtherEngine>
  auto operator()(OtherEngine& e) {
    return get(e);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if has_cxx17_support()
random_uniform()->random_uniform<double, std::mt19937_64>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
random_uniform(Engine &&)->random_uniform<double, Engine>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
random_uniform(T min, T max)->random_uniform<T, std::mt19937_64>;
#endif

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
  random_normal& operator=(const random_normal&) = default;
  random_normal& operator=(random_normal&&) noexcept = default;

  //----------------------------------------------------------------------------
  random_normal(const Engine& _engine) : engine{_engine}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Engine&& _engine)
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
#if has_cxx17_support()
random_normal()->random_normal<double, std::mt19937_64>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
random_normal(Engine &&)->random_normal<double, Engine>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
random_normal(T mean, T stddev)->random_normal<T, std::mt19937_64>;
#endif
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
