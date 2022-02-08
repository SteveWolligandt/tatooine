#ifndef TATOOINE_STEADIFICATION_GPU_MEM_CACHE_H
#define TATOOINE_STEADIFICATION_GPU_MEM_CACHE_H

#include <tatooine/memory_usage.h>
#include <functional>
#include <iostream>
#include <unordered_map>

//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename key_t, typename value_t>
class GPUMemCache : public std::map<key_t, value_t> {
 public:
  //============================================================================

  using this_type         = GPUMemCache<key_t, value_t>;
  using parent_t       = std::map<key_t, value_t>;
  using value_type     = typename parent_t::value_type;
  using key_type       = typename parent_t::key_type;
  using size_type      = typename parent_t::size_type;
  using iterator       = typename parent_t::iterator;
  using const_iterator = typename parent_t::const_iterator;

  using parent_t::at;

  //----------------------------------------------------------------------------

  GPUMemCache(
      uint64_t max_cache_size       = std::numeric_limits<uint64_t>::max(),
      uint64_t max_memory_usage     = std::numeric_limits<uint64_t>::max(),
      uint64_t max_gpu_memory_usage = std::numeric_limits<uint64_t>::max())
      : m_max_cache_size{max_cache_size},
        m_max_memory_usage{max_memory_usage},
        m_max_gpu_memory_usage{max_gpu_memory_usage} {}

  //----------------------------------------------------------------------------

  GPUMemCache(const GPUMemCache& other) = delete;
  GPUMemCache(GPUMemCache&& other)      = delete;

  //============================================================================
  //! @defgroup insert_overwrites insert overwrites
  //! @{
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

  //! @}

  //============================================================================
  //! @defgroup insert_overwrites insert overwrites
  //! @{
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
    std::pair<iterator, bool> insert_result =
        parent_t::template insert<P>(std::forward<P>(value));
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
  //! @}

  //============================================================================
  //! @defgroup erase_overwrites erase overwrites
  //! @{
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
  //! @}

  //============================================================================
  //! @defgroup emplace_overwrites emplace overwrites
  //! @{
  template <class... Args>
  auto try_emplace(const key_type& key, Args&&... args) {
    auto res = parent_t::try_emplace(key, std::forward<Args>(args)...);
    if (res.second) enqueue(res.first);
    return res;
  }

  //----------------------------------------------------------------------------

  template <class... Args>
  auto try_emplace(key_type&& key, Args&&... args) {
    auto res =
        parent_t::try_emplace(std::move(key), std::forward<Args>(args)...);
    if (res.second) enqueue(res.first);
    return res;
  }

  //----------------------------------------------------------------------------

  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) = delete;
  //! @}

  std::optional<const_iterator> has_key(const key_type& key) {
    if (auto iter = find(key); iter != this->end()) return iter;
    return {};
  }

  //============================================================================
 private:
  //============================================================================
  void enqueue(const_iterator insert_result) {
    m_queue.push_back(insert_result);
    capacity_check();
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
    if (m_queue.size() >= m_max_cache_size) {
      parent_t::erase(m_queue.front());
      m_queue.pop_front();
    }
    auto vm = tatooine::memory_usage().first / 1024.0;
    if (vm >= m_max_memory_usage) {
      // std::cout << "========================= mem exceeded2: "
      //           << vm / 1024.0 / 1024.0
      //           << " >= " << m_max_memory_usage / 1024.0 / 1024.0 << '\n';
      while (vm > m_max_memory_usage && m_queue.size() > 1) {
        parent_t::erase(m_queue.front());
        m_queue.pop_front();
        vm = tatooine::memory_usage().first / 1024.0;
        // std::cout << "========================= delete: "
        //           << vm / 1024.0 / 1024.0 << '\n';
        // std::cout << "m_queue.size(): " << m_queue.size() << '\n';
      }
    }

    size_t usage = gl::get_total_available_memory() -
                   gl::get_current_available_memory();
    // std::cout << "gpu mem: " << usage / 1024.0 / 1024.0 << " / "
    //           << gl::get_total_available_memory() / 1024.0 / 1024.0 <<
    //           '\n';
    if (usage >= m_max_gpu_memory_usage) {
      // std::cout << "========================= gpumem exceeded: "
      //           << usage / 1024.0 / 1024.0
      //           << " >= " << m_max_gpu_memory_usage / 1024.0 / 1024.0 << '\n';
      while (usage > m_max_gpu_memory_usage && m_queue.size() > 1) {
        parent_t::erase(m_queue.front());
        m_queue.pop_front();
        usage = gl::get_total_available_memory() -
                gl::get_current_available_memory();
        // std::cout << "========================= gpu delete: "
        //           << usage / 1024.0 / 1024.0 << '\n';
        // std::cout << "m_queue.size(): " << m_queue.size() << '\n';
      }
    }
  }

  //============================================================================

  uint64_t                  m_max_cache_size;
  uint64_t                  m_max_memory_usage;
  uint64_t                  m_max_gpu_memory_usage;
  std::list<const_iterator> m_queue;
};
template <typename real_type, typename key_t, typename value_t>
using TimedGPUMemCache = GPUMemCache<real_type, GPUMemCache<key_t, value_t>>;

//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
