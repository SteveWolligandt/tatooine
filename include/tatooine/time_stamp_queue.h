#ifndef TATOOINE_TIME_STAMP_QUEUE_H
#define TATOOINE_TIME_STAMP_QUEUE_H
//==============================================================================
#include <chrono>
#include <queue>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Timer>
class time_stamp_queue_components_type {
 public:
  auto operator()(std::pair<T, std::chrono::time_point<Timer>> const& lhs,
                  std::pair<T, std::chrono::time_point<Timer>> const& rhs) const
      -> bool {
    return lhs.second < rhs.second;
  }
};
//==============================================================================
template <typename T, typename Timer = std::chrono::high_resolution_clock>
class time_stamp_queue
    : public std::priority_queue<
          std::pair<T, std::chrono::time_point<Timer>>,
          std::vector<std::pair<T, std::chrono::time_point<Timer>>>,
          time_stamp_queue_components_type<T, Timer>> {
  using parent_t = std::priority_queue<
      std::pair<T, std::chrono::time_point<Timer>>,
      std::vector<std::pair<T, std::chrono::time_point<Timer>>>,
      time_stamp_queue_components_type<T, Timer>>;
  //============================================================================
 public:
  using parent_t::parent_t;
  template <typename... Args>
  //----------------------------------------------------------------------------
  auto emplace(Args&&... args) {
    auto t = T{std::forward<Args>(args)...};
    std::erase_if(this->c, [&](auto const& elem) { return elem.first == t; });
    parent_t::emplace(std::pair{std::move(t), Timer::now()});
  }
  //----------------------------------------------------------------------------
  auto push(T const& t) {
    std::erase_if(this->c, [&](auto const& elem) { return elem.first == t; });
    parent_t::emplace(std::pair{t, Timer::now()});
  }
  //----------------------------------------------------------------------------
  auto push(T&& t) {
    std::erase_if(this->c, [&](auto const& elem) { return elem.first == t; });
    parent_t::emplace(std::pair{std::move(t), Timer::now()});
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
