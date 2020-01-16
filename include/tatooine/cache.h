#ifndef TATOOINE_CACHE_H
#define TATOOINE_CACHE_H

#include <map>
#include <list>
#include <optional>
#include <algorithm>
#include "memory_usage.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Key, typename Value>
class cache {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using container_t    = std::map<Key, Value>;
  using const_iterator = typename container_t::const_iterator;
  using usage_t        = std::list<const_iterator>;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  container_t m_data;
  mutable usage_t m_usage;
  uint64_t m_max_elements;
  uint64_t m_max_memory_usage;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  cache(uint64_t max_elements = std::numeric_limits<uint64_t>::max(),
        uint64_t max_memory_usage = std::numeric_limits<uint64_t>::max())
      : m_max_elements{max_elements}, m_max_memory_usage{max_memory_usage} {}
  cache(const cache& other)
      : m_data{other.m_data},
        m_max_elements{other.m_max_elements},
        m_max_memory_usage{other.m_max_memory_usage} {
    for (auto it : other.m_usage) {
      m_usage.push_back(next(begin(m_data), distance(begin(other.m_data), it)));
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  cache(cache&& other) = default;
  auto& operator=(const cache& other) {
    m_data = other.m_data;
    m_max_elements = other.m_max_elements;
    m_max_memory_usage = other.m_max_memory_usage;
    for (auto it : other.m_usage) {
      m_usage.push_back(next(begin(m_data), distance(begin(other.m_data), it)));
    }
    return *this;
  }
  cache& operator=(cache&& other) = default;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  void capacity_check() {
    while (m_data.size() > m_max_elements ||
           (memory_usage().first / 1024.0 > m_max_memory_usage &&
            !m_data.empty())) {
      m_data.erase(m_usage.back());
      m_usage.pop_back();
    }
  }
  //----------------------------------------------------------------------------
  void enqueue(const_iterator it, bool ins) {
    if (ins) {
      m_usage.push_front(it);
      capacity_check();
    }
  }
  //----------------------------------------------------------------------------
  void refresh_usage(const_iterator it) {
    m_usage.erase(std::find(begin(m_usage), end(m_usage), it));
    m_usage.push_front(it);
  } 

 public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto insert(const Key& key, const Value& value) {
    auto insertion = m_data.insert(std::pair{key, value});
    enqueue(insertion.first, insertion.second);
    return insertion;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto insert(Key&& key, const Value& value) {
    auto insertion = m_data.insert(std::pair{std::move(key), value});
    enqueue(insertion.first, insertion.second);
    return insertion;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto insert(const Key& key, Value&& value) {
    auto insertion = m_data.insert(std::pair{key, std::move(value)});
    enqueue(insertion.first, insertion.second);
    return insertion;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto insert(Key&& key, Value&& value) {
    auto insertion = m_data.insert(std::pair{std::move(key), std::move(value)});
    enqueue(insertion.first, insertion.second);
    return insertion;
  }
  //----------------------------------------------------------------------------
  template <typename... Args>
  auto emplace(const Key& key, Args&&... args) {
    return insert(key, Value{std::forward<Args>(args)...});
  }
  //----------------------------------------------------------------------------
  const auto& operator[](const Key& key) const { return at(key); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& operator[](const Key& key) { return at(key); }
  //----------------------------------------------------------------------------
  const auto& at(const Key& key) const {
    auto it = m_data.find(key);
    refresh_usage(it);
    return *it;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& at(const Key& key) {
    auto it = m_data.find(key);
    refresh_usage(it);
    return *it;
  }
  //----------------------------------------------------------------------------
  std::optional<const_iterator> contains(const Key& key) const {
    if (auto it = m_data.find(key); it != end(m_data)) {
      return it;
    }
    return {};
  }
  //----------------------------------------------------------------------------
  auto size() const { return m_data.size(); }
  //----------------------------------------------------------------------------
  void clear() {
    m_data.clear();
    m_usage.clear();
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
