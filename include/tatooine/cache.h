#ifndef TATOOINE_CACHE_H
#define TATOOINE_CACHE_H

#include <functional>
#include <iostream>
#include <map>
#include <list>
#include <boost/range/algorithm.hpp>
#include "memory_usage.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename key_t, typename value_t>
struct cache : std::map<key_t, value_t> {
  using this_t         = cache<key_t, value_t>;
  using parent_t       = std::map<key_t, value_t>;
  using value_type     = typename parent_t::value_type;
  using key_type       = typename parent_t::key_type;
  using size_type      = typename parent_t::size_type;
  using iterator       = typename parent_t::iterator;
  using const_iterator = typename parent_t::const_iterator;

  using parent_t::at;

  //============================================================================
  uint64_t                  m_max_cache_size;
  uint64_t                  m_max_memory_usage;
  std::list<const_iterator> m_queue;

  //----------------------------------------------------------------------------
  cache(uint64_t max_cache_size   = std::numeric_limits<uint64_t>::max(),
        uint64_t max_memory_usage = std::numeric_limits<uint64_t>::max())
      : m_max_cache_size{max_cache_size},
        m_max_memory_usage{max_memory_usage} {}

  //----------------------------------------------------------------------------
  cache(const cache& other) = delete;
  cache(cache&& other)      = delete;

  auto max_cache_size() const { return m_max_cache_size; }
  auto max_memory_usage() const { return m_max_memory_usage; }
  void set_max_cache_size(uint64_t s) { m_max_cache_size = s; }
  void set_max_memory_usage(uint64_t s) { m_max_memory_usage = s; }

  //============================================================================
  /// \defgroup at_overwrites at overwrites
  /// \{
  //----------------------------------------------------------------------------
  value_t& at(const key_t& key) {
    auto [it, suc] = try_emplace(key);
    return it->second;
  }

  //----------------------------------------------------------------------------
  const value_t& at(const key_t& key) const {
    auto [it, suc] = try_emplace(key);
    return it->second;
  }
  /// \}

  //============================================================================
  /// \defgroup insert_overwrites insert overwrites
  /// \{
  std::pair<iterator, bool> insert(const value_type& value) {
    auto insert_result = parent_t::insert(value);
    enqueue(insert_result);
    return insert_result;
  }

  //----------------------------------------------------------------------------
  auto insert(value_type&& value) {
    auto insert_result = parent_t::insert(std::move(value));
    enqueue(insert_result);
    return insert_result;
  }

  //----------------------------------------------------------------------------
  template <typename P>
  auto insert(P&& value) {
    auto insert_result = parent_t::template insert<P>(std::forward<P>(value));
    enqueue(insert_result);
    return insert_result;
  }

  //----------------------------------------------------------------------------
  auto insert(const_iterator hint, const value_type& value) {
    auto insert_result = parent_t::insert(hint, value);
    enqueue(insert_result);
    return insert_result;
  }

  //----------------------------------------------------------------------------
  template <class P>
  auto insert(const_iterator hint, P&& value) {
    auto insert_result = parent_t::insert(hint, std::forward<P>(value));
    enqueue(insert_result);
    return insert_result;
  }

  //----------------------------------------------------------------------------
  template <class InputIt>
  void insert(InputIt first, InputIt last) = delete;

  //----------------------------------------------------------------------------
  void insert(std::initializer_list<value_type> ilist) {
    auto insert_result = parent_t::insert(ilist);
    enqueue(insert_result);
  }
  /// \}

  //============================================================================
  /// \defgroup erase_overwrites erase overwrites
  /// \{
  auto erase(const_iterator position) {
    remove_from_queue(position);
    auto it = parent_t::erase(position);
    return it;
  }

  //----------------------------------------------------------------------------
  auto erase(const_iterator first, const_iterator last) {
    for (auto i = first; i != last; ++i) remove_from_queue(i);
    auto it = parent_t::erase(first, last);
    return it;
  }

  //----------------------------------------------------------------------------
  auto erase(const key_type& key) {
    remove_from_queue(find(key));
    return parent_t::erase(key);
  }
  /// \}

  //============================================================================
  /// \defgroup emplace_overwrites emplace overwrites
  /// \{
  template <class... Args>
  auto try_emplace(const key_type& key, Args&&... args) {
    auto res = parent_t::try_emplace(key, std::forward<Args>(args)...);
    auto& [it, new_key] = res;
    if (new_key) enqueue(it);
    // else         refresh(it);
    return res;
  }

  //----------------------------------------------------------------------------
  template <class... Args>
  auto try_emplace(key_type&& key, Args&&... args) {
    auto res =
        parent_t::try_emplace(std::move(key), std::forward<Args>(args)...);
    auto& [it, new_key] = res;
    if (new_key) enqueue(it);
    // else         refresh(it);
    return res;
  }

  //----------------------------------------------------------------------------
  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) = delete;
  /// \}

  std::optional<const_iterator> has_key(const key_type& key) {
    if (auto iter = find(key); iter != this->end()) return iter;
    return {};
  }

 private:
  //============================================================================
  void enqueue(const_iterator insert_result) {
    m_queue.push_back(insert_result);
    capacity_check();
  }

  //----------------------------------------------------------------------------
  void refresh(const_iterator insert_result) {
    auto found_it = boost::find(m_queue, insert_result);
    if (found_it != end(m_queue)) {
      m_queue.push_back(insert_result);
      m_queue.erase(found_it);
    }
  }

  //----------------------------------------------------------------------------
  void remove_from_queue(const_iterator del_it) {
    for (auto queue_it = m_queue.begin(); queue_it != m_queue.end(); ++queue_it)
      if (*queue_it == del_it) {
        --queue_it;
        m_queue.erase(std::next(queue_it));
      }
  }

  //----------------------------------------------------------------------------
  void enqueue(std::pair<iterator, bool>& insert_result) {
    if (insert_result.second) {
      m_queue.push_back(insert_result.first);
      capacity_check();
    }
  }

  //----------------------------------------------------------------------------
  void capacity_check() {
    if (m_queue.size() > m_max_cache_size) {
      parent_t::erase(m_queue.front());
      m_queue.pop_front();
    }
    auto vm = memory_usage().first / 1024.0;
    if (vm >= m_max_memory_usage) {
      while (vm > m_max_memory_usage && m_queue.size() > 1) {
        parent_t::erase(m_queue.front());
        m_queue.pop_front();
        vm = memory_usage().first / 1024.0;
      }
    }
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
