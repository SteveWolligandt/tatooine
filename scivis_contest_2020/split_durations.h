#ifndef TATOOINE_SCIVIS_CONTEST_2020_SPLIT_DURATIONS_H
#define TATOOINE_SCIVIS_CONTEST_2020_SPLIT_DURATIONS_H
//==============================================================================
#include <chrono>
#include <optional>
#include <tuple>
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename Duration, typename Leftover>
struct split_duration {
  Duration d;
  Leftover leftover;

  template <typename T>
  split_duration(T t)
      : d(std::chrono::duration_cast<Duration>(t)),
        leftover(t - std::chrono::duration_cast<Leftover>(d)) {}
};
//==============================================================================
template <typename... Durations, typename Rep, typename Period>
std::tuple<Durations...> durations(std::chrono::duration<Rep, Period> dur) {
  std::tuple<std::optional<
      split_duration<Durations, std::chrono::duration<Rep, Period>>>...>
      tmp;
  ((void)((void)std::get<std::optional<
              split_duration<Durations, std::chrono::duration<Rep, Period>>>>(
              tmp)
              .emplace(dur),
          dur = std::get<std::optional<split_duration<
                    Durations, std::chrono::duration<Rep, Period>>>>(tmp)
                    ->leftover),
   ...);
  return std::tuple{std::get<std::optional<
      split_duration<Durations, std::chrono::duration<Rep, Period>>>>(tmp)
                        ->d...};
}
//==============================================================================
template <typename T>
struct tag_t {};
//==============================================================================
template <typename T>
constexpr tag_t<T> tag = {};

constexpr inline std::string_view duration_name(tag_t<std::chrono::milliseconds>) {
  return "ms";
}
constexpr inline std::string_view duration_name(tag_t<std::chrono::seconds>) {
  return "s";
}
constexpr inline std::string_view duration_name(tag_t<std::chrono::minutes>) {
  return "min";
}
constexpr inline std::string_view duration_name(tag_t<std::chrono::hours>) {
  return "h";
}
constexpr inline std::string_view duration_name(tag_t<std::chrono::days>) {
  return "d";
}
//------------------------------------------------------------------------------
template <typename Duration>
void format_duration_(Duration const& d, std::ostream& out, bool& written) {
  if (d.count() > 0) {
    if (written) {
      out << ":";
    }
    out << d.count() << duration_name(tag<Duration>);
    written = true;
  } else {
    written = false;
  }
}
//------------------------------------------------------------------------------
template <typename... Durations, typename Rep,
          typename Period>
void format_duration(std::chrono::duration<Rep, Period> const& t,
                     std::ostream&                             out) {
  auto const split   = durations<Durations...>(t);
  bool       written = false;
  (format_duration_(std::get<Durations>(split), out, written), ...);
}
  //==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
