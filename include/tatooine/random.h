#ifndef TATOOINE_RANDOM_H
#define TATOOINE_RANDOM_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/real.h>
#include <tatooine/type_traits.h>

#include <boost/range/algorithm/generate.hpp>
#include <random>
#include <string>
//==============================================================================
namespace tatooine::random {
//==============================================================================
template <typename ValueType, typename Engine = std::mt19937_64>
struct uniform {
  static_assert(is_arithmetic<ValueType>);
  //============================================================================
  using engine_type = Engine;
  using real_type   = ValueType;
  using distribution_type =
      std::conditional_t<is_floating_point<ValueType>,
                         std::uniform_real_distribution<ValueType>,
                         std::uniform_int_distribution<ValueType>>;
  //============================================================================
 private:
  engine_type       m_engine;
  distribution_type m_distribution;
  //============================================================================
 public:
  uniform()
      : m_engine{std::random_device{}()},
        m_distribution{ValueType(0), ValueType(1)} {}
  uniform(const uniform&)     = default;
  uniform(uniform&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const uniform&) noexcept -> uniform& = default;
  auto operator=(uniform&&) noexcept -> uniform& = default;
  //----------------------------------------------------------------------------
  ~uniform() = default;
  //----------------------------------------------------------------------------
  uniform(ValueType const min, ValueType const max)
      : m_engine{std::random_device{}()}, m_distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <convertible_to<Engine> EngineArg>
  uniform(ValueType const min, ValueType const max, EngineArg&& eng)
      : m_engine{std::forward<EngineArg>(eng)}, m_distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <convertible_to<Engine> EngineArg>
  uniform(EngineArg&& eng)
      : m_engine{std::forward<EngineArg>(eng)},
        m_distribution{ValueType(0), ValueType(1)} {}
  //============================================================================
  auto operator()() { return m_distribution(m_engine); }
  //----------------------------------------------------------------------------
  auto engine() const -> auto const& { return m_engine; }
  auto engine() -> auto& { return m_engine; }
  //----------------------------------------------------------------------------
  auto distribution() const -> auto const& { return m_distribution; }
  auto distribution() -> auto& { return m_distribution; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
uniform()->uniform<real_number, std::mt19937_64>;

template <typename ValueType>
uniform(ValueType const min, ValueType const max)
    -> uniform<ValueType, std::mt19937_64>;

template <typename ValueType, typename Engine>
uniform(ValueType const min, ValueType const max, Engine&&)
    -> uniform<ValueType, std::decay_t<Engine>>;

template <typename ValueType, typename Engine>
uniform(Engine&) -> uniform<ValueType, std::decay_t<Engine>&>;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename ValueType, typename Engine = std::mt19937_64>
auto uniform_vector(std::size_t n, ValueType a = ValueType(0),
                    ValueType b      = ValueType(1),
                    Engine&&  engine = Engine{std::random_device{}()}) {
  uniform rand(a, b, std::forward<Engine>(engine));

  std::vector<ValueType> rand_data(n);
  boost::generate(rand_data, [&] { return rand(); });
  return rand_data;
}
//==============================================================================
inline auto alpha_numeric_string(std::size_t const size) {
  static constexpr auto char_set = std::string_view{
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"};
  auto index = uniform<std::size_t>{0, char_set.size() - 1};
  auto str   = std::string(size, ' ');
  for (auto& c : str) {
    c = char_set[index()];
  }
  return str;
}
//==============================================================================
template <typename ValueType, typename Engine = std::mt19937_64>
struct normal {
  using engine_type       = Engine;
  using real_type         = ValueType;
  using distribution_type = std::normal_distribution<ValueType>;

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
  normal(ValueType mean, ValueType stddev)
      : engine{std::random_device{}()}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(const Engine& _engine, ValueType mean, ValueType stddev)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(Engine&& _engine, ValueType mean, ValueType stddev)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(ValueType mean, ValueType stddev, const Engine& _engine)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  normal(ValueType mean, ValueType stddev, Engine&& _engine)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}

  //============================================================================
 private:
  Engine            engine;
  distribution_type distribution;

  //============================================================================
 public:
  auto get() { return distribution(engine); }
  auto operator()() { return get(); }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
normal()->normal<double, std::mt19937_64>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
normal(Engine&&) -> normal<double, Engine>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ValueType>
normal(ValueType mean, ValueType stddev) -> normal<ValueType, std::mt19937_64>;
//==============================================================================
template <typename Iterator, typename RandomEngine>
auto random_elem(Iterator begin, Iterator end, RandomEngine& eng) {
  if (begin == end) {
    return end;
  }
  const auto size = static_cast<std::size_t>(distance(begin, end));
  std::uniform_int_distribution<std::size_t> rand{0, size - 1};
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
  std::uniform_int_distribution<std::size_t> coin{0, 1};
  return coin(eng) == 0 ? HEADS : TAILS;
}
//==============================================================================
}  // namespace tatooine::random
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename ValueType>
struct is_random_number_generator_impl : std::false_type {};
template <typename ValueType, typename Engine>
struct is_random_number_generator_impl<random::uniform<ValueType, Engine>>
    : std::true_type {};
template <typename ValueType, typename Engine>
struct is_random_number_generator_impl<random::normal<ValueType, Engine>>
    : std::true_type {};
template <typename ValueType>
static auto constexpr is_random_number_generator =
    is_random_number_generator_impl<ValueType>::value;
template <typename ValueType>
concept random_number_generator =
    is_random_number_generator<std::decay_t<ValueType>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
