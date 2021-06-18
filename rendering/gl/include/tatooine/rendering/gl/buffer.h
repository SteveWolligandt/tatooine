#ifndef YAVIN_BUFFER
#define YAVIN_BUFFER
//==============================================================================
#include <mutex>
#include <vector>
#include "glfunctions.h"
#include "texsettings.h"
#include "mutexhandler.h"
#include "idholder.h"
//==============================================================================
namespace yavin {
//==============================================================================
enum usage_t {
  STREAM_DRAW  = GL_STREAM_DRAW,
  STREAM_READ  = GL_STREAM_READ,
  STREAM_COPY  = GL_STREAM_COPY,
  STATIC_DRAW  = GL_STATIC_DRAW,
  STATIC_READ  = GL_STATIC_READ,
  STATIC_COPY  = GL_STATIC_COPY,
  DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
  DYNAMIC_READ = GL_DYNAMIC_READ,
  DYNAMIC_COPY = GL_DYNAMIC_COPY
};
//==============================================================================
template <GLsizei array_type, typename T>
class buffer;
//==============================================================================
template <GLsizei ArrayType, typename T, GLbitfield Access>
class buffer_map {
 public:
  static constexpr auto access     = Access;
  static constexpr auto array_type = ArrayType;
  using buffer_t                   = buffer<array_type, T>;
  static constexpr auto data_size  = buffer_t::data_size;

 private:
  const buffer_t* m_buffer;
  size_t          m_offset;
  size_t          m_length;
  T*              m_gpu_mapping;
  bool            m_unmapped = false;

 public:
  /// constructor gets a mapping to gpu_buffer
  buffer_map(const buffer_t* buffer, size_t offset, size_t length)
      : m_buffer(buffer), m_offset(offset), m_length(length) {
    m_gpu_mapping = (T*)gl::map_named_buffer_range(
        m_buffer->id(), data_size * offset, data_size * m_length,
        access);
    detail::mutex::gl_call.lock();
  }
  buffer_map(const buffer_map&) = delete;
  buffer_map(buffer_map&&)      = delete;
  auto operator=(const buffer_map&) -> buffer_map& = delete;
  auto operator=(buffer_map &&)     -> buffer_map& = delete;

  auto operator=(const std::vector<T>& data) ->buffer_map& {
    assert(size(data) == m_buffer->size());
    for (size_t i = 0; i < size(data); ++i) { at(i) = data[i]; }
    return &this;
  }

  /// destructor unmaps the buffer
  ~buffer_map() { unmap(); }

  void unmap() {
    detail::mutex::gl_call.unlock();
    if (!m_unmapped) {
      gl::unmap_named_buffer(m_buffer->id());
      m_unmapped = true;
    }
  }

  auto at(size_t i) -> auto& {
    return m_gpu_mapping[i];
  }
  auto at(size_t i) const -> const auto& {
    return m_gpu_mapping[i];
  }

  auto front() -> auto& {
    return at(0);
  }
  auto front() const -> const auto& {
    return at(0);
  }

  auto back() -> auto& {
    return at(m_length - 1);
  }
  auto back() const -> const auto& {
    return at(m_length - 1);
  }

  auto operator[](size_t i) -> auto& {
    return at(i);
  }
  auto operator[](size_t i) const -> const auto& {
    return at(i);
  }

  auto begin() {
    return m_gpu_mapping;
  }
  auto begin() const {
    return m_gpu_mapping;
  }

  auto end() {
    return m_gpu_mapping + m_length;
  }
  auto end() const {
    return m_gpu_mapping + m_length;
  }

  auto offset() const {
    return m_offset;
  }
  auto length() const {
    return m_length;
  }
};
//------------------------------------------------------------------------------
// buffer_map free functions
//------------------------------------------------------------------------------
template <GLsizei ArrayType, typename T, GLbitfield Access>
auto begin(buffer_map<ArrayType, T, Access>& map) {
  return map.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <GLsizei ArrayType, typename T, GLbitfield Access>
auto begin(buffer_map<ArrayType, T, Access> const& map) {
  return map.begin();
}
//------------------------------------------------------------------------------
template <GLsizei ArrayType, typename T, GLbitfield Access>
auto end(buffer_map<ArrayType, T, Access>& map) {
  return map.end();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <GLsizei ArrayType, typename T, GLbitfield Access>
auto end(buffer_map<ArrayType, T, Access> const& map) {
  return map.end();
}
//------------------------------------------------------------------------------
// buffer_map free typedefs
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
using rbuffer_map = buffer_map<array_type, T, GL_MAP_READ_BIT>;

template <GLsizei array_type, typename T>
using wbuffer_map = buffer_map<array_type, T, GL_MAP_WRITE_BIT>;

template <GLsizei array_type, typename T>
using rwbuffer_map =
    buffer_map<array_type, T, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT>;

//==============================================================================
/// Returned by buffer::operator[] const for reading single elements
template <GLsizei array_type, typename T>
class rbuffer_map_element {
 public:
  using buffer_t = buffer<array_type, T>;
  using rmap_t   = rbuffer_map<array_type, T>;

 private:
  const buffer_t* m_buffer;
  size_t          m_idx;

 public:
  rbuffer_map_element(const buffer_t* buffer, size_t idx)
      : m_buffer{buffer}, m_idx{idx} {}
  rbuffer_map_element(const rbuffer_map_element& other)     = default;
  rbuffer_map_element(rbuffer_map_element&& other) noexcept = default;

  auto operator=(const rbuffer_map_element& other)
    -> rbuffer_map_element& = default;
  auto operator=(rbuffer_map_element&& other) noexcept
    -> rbuffer_map_element& = default;

  ~rbuffer_map_element() = default;

  /// for accessing single gpu data element.
  explicit operator T() const { return download(); }

  [[nodiscard]] auto download() const {
    rmap_t map(m_buffer, m_idx, 1);
    return map.front();
  }
  auto operator==(const T& t) const -> bool { return download() == t; }
  auto operator!=(const T& t) const -> bool { return !operator==(t); }
  auto operator>(const T& t) const -> bool { return download() > t; }
  auto operator>=(const T& t) const -> bool { return download() >= t; }
  auto operator<(const T& t) const -> bool { return download() < t; }
  auto operator<=(const T& t) const -> bool { return download() <= t; }
};

template <GLsizei array_type, typename T>
inline auto operator<<(std::ostream& out,
                       rbuffer_map_element<array_type, T>& data) -> auto& {
  out << data.download();
  return out;
}

//==============================================================================
/// Returned by buffer::operator[] for reading and writing single elements
/// Returned by buffer::operator[] for reading and writing single elements
template <GLsizei array_type, typename T>
class rwbuffer_map_element  {
 public:
  using buffer_t = buffer<array_type, T>;
  using rmap_t   = rbuffer_map<array_type, T>;

 private:
  const buffer_t* m_buffer;
  size_t          m_idx;

 public:
  rwbuffer_map_element(const buffer_t* buffer, size_t idx)
      : m_buffer{buffer}, m_idx{idx} {}
  rwbuffer_map_element(const rwbuffer_map_element& other)     = default;
  rwbuffer_map_element(rwbuffer_map_element&& other) noexcept = default;

  auto operator=(const rwbuffer_map_element& other)
    -> rwbuffer_map_element& = default;
  auto operator=(rwbuffer_map_element&& other) noexcept
    -> rwbuffer_map_element& = default;

  ~rwbuffer_map_element() = default;
  /// for assigning single gpu data element.
  auto operator=(T&& data) -> auto& {
    gl::named_buffer_sub_data(m_buffer->id(),
                              m_idx * buffer_t::data_size,
                              buffer_t::data_size, &data);
    return *this;
  }

  /// for accessing single gpu data element.
  explicit operator T() const { return download(); }

  [[nodiscard]] auto download() const {
    rmap_t map(m_buffer, m_idx, 1);
    return map.front();
  }
  auto operator==(const T& t) const -> bool { return download() == t; }
  auto operator!=(const T& t) const -> bool { return !operator==(t); }
  auto operator>(const T& t) const -> bool { return download() > t; }
  auto operator>=(const T& t) const -> bool { return download() >= t; }
  auto operator<(const T& t) const -> bool { return download() < t; }
  auto operator<=(const T& t) const -> bool { return download() <= t; }
};
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
inline auto operator<<(std::ostream& out,
                       rwbuffer_map_element<array_type, T>& data) -> auto& {
  out << data.download();
  return out;
}

//==============================================================================
/// non-const buffer iterator
template <GLsizei array_type, typename T>
class buffer_iterator {
 public:
  using buffer_t = buffer<array_type, T>;
  //----------------------------------------------------------------------------
  // iterator typedefs
  using value_type        = T;
  using reference         = T&;
  using pointer           = T*;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;
  //----------------------------------------------------------------------------
  buffer_iterator(buffer_t* buffer, size_t idx) : m_buffer(buffer), m_idx(idx) {}
  //----------------------------------------------------------------------------
  buffer_iterator(const buffer_iterator& other) = default;
  buffer_iterator(buffer_iterator&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const buffer_iterator& other) -> auto& = default;
  auto operator=(buffer_iterator&& other) noexcept -> auto& = default;
  //----------------------------------------------------------------------------
  ~buffer_iterator() = default;
  //----------------------------------------------------------------------------
  /// get the buffer element the iterator refers to
  auto operator*() const -> T {
    return rbuffer_map_element(m_buffer, m_idx);
  }
  //----------------------------------------------------------------------------
  /// are two iterators equal?
  auto operator==(const buffer_iterator& other) const {
    return m_idx == other.m_idx;
  }
  //----------------------------------------------------------------------------
  /// are two iterators different?
  auto operator!=(const buffer_iterator& other) const {
    return !operator==(other);
  }
  //----------------------------------------------------------------------------
  /// pre-increment iterator
  auto operator++() ->auto& {
    ++m_idx;
    return *this;
  }
  //----------------------------------------------------------------------------
  /// post-increment iterator
  auto operator++(int) {
    buffer_iterator vi{*this};
    ++(*this);
    return vi;
  }
  //----------------------------------------------------------------------------
  /// pre-decrement iterator
  auto operator--() -> auto& {
    --m_idx;
    return *this;
  }
  //----------------------------------------------------------------------------
  /// post-decrement iterator
  auto operator--(int) {
    buffer_iterator vi(*this);
    --(*this);
    return vi;
  }

 private:
  buffer_t* m_buffer;
  size_t    m_idx;
};

//==============================================================================
/// const buffer iterator
template <GLsizei array_type, typename T>
class cbuffer_iterator {
 public:
  using buffer_t = buffer<array_type, T>;

  // iterator typedefs
  using value_type        = T;
  using reference         = T&;
  using pointer           = T*;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;
  //----------------------------------------------------------------------------
  cbuffer_iterator(const buffer_t* buffer, size_t idx)
      : m_buffer(buffer), m_idx(idx) {}
  //----------------------------------------------------------------------------
  cbuffer_iterator(const cbuffer_iterator& other)     = default;
  cbuffer_iterator(cbuffer_iterator&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const cbuffer_iterator& other) -> auto& = default;
  auto operator=(cbuffer_iterator&& other) noexcept -> auto& = default;
  //----------------------------------------------------------------------------
  ~cbuffer_iterator() = default;
  //----------------------------------------------------------------------------
  /// get the buffer element the iterator refers to
  auto operator*() const -> T {
    return rbuffer_map_element(m_buffer, m_idx);
  }
  //----------------------------------------------------------------------------
  /// are two iterators equal?
  auto operator==(const cbuffer_iterator& other) const {
    return (m_idx == other.m_idx);
  }
  //----------------------------------------------------------------------------
  /// are two iterators different?
  auto operator!=(const cbuffer_iterator& other) const {
    return !operator==(other);
  }
  //----------------------------------------------------------------------------
  /// pre-increment iterator
  auto operator++() -> auto& {
    ++m_idx;
    return *this;
  }
  //----------------------------------------------------------------------------
  /// post-increment iterator
  auto operator++(int) {
    cbuffer_iterator vi(*this);
    ++(*this);
    return vi;
  }
  //----------------------------------------------------------------------------
  /// pre-decrement iterator
  auto operator--() -> auto& {
    --m_idx;
    return *this;
  }
  //----------------------------------------------------------------------------
  /// post-decrement iterator
  auto operator--(int) {
    cbuffer_iterator vi(*this);
    --(*this);
    return vi;
  }

 private:
  const buffer_t* m_buffer;
  size_t          m_idx;
};


//==============================================================================
/// buffer base class for each of the OpenGL buffer types
template <GLsizei _array_type, typename T>
class buffer : public id_holder<GLuint> {
 public:
  using parent_t = id_holder<GLuint>;
  using parent_t::id;
  friend class rbuffer_map_element<_array_type, T>;
  friend class rwbuffer_map_element<_array_type, T>;

  constexpr static GLsizei array_type = _array_type;
  constexpr static size_t  data_size  = sizeof(T);

  using this_t = buffer<array_type, T>;
  using data_t = T;

  using relement_t  = rbuffer_map_element<array_type, T>;
  using rwelement_t = rwbuffer_map_element<array_type, T>;

  using iterator_t       = buffer_iterator<array_type, T>;
  using const_iterator_t = cbuffer_iterator<array_type, T>;

  using rmap_t  = rbuffer_map<array_type, T>;
  using wmap_t  = wbuffer_map<array_type, T>;
  using rwmap_t = rwbuffer_map<array_type, T>;

 private:
  size_t  m_size     = 0;
  size_t  m_capacity = 0;
  usage_t m_usage;

 public:
  explicit buffer(usage_t usage);
  buffer(const buffer& other);
  buffer(buffer&& other) noexcept;
  auto operator=(const buffer& other) -> buffer&;
  auto operator=(buffer&& other) noexcept -> buffer&;
  auto operator=(const std::vector<T>& data) -> buffer&;

  buffer(size_t n, usage_t usage);
  buffer(size_t n, const T& initial, usage_t usage);
  buffer(const std::vector<T>& data, usage_t usage);
  ~buffer();

  void create_handle();
  void destroy_handle();

  void           upload_data(const T& data);
  void           upload_data(const std::vector<T>& data);
  [[nodiscard]] auto download_data() const -> std::vector<T>;

  void        bind() const;
  static void unbind();

  void copy(const this_t& other);

  [[nodiscard]] auto empty() const -> bool { return m_size == 0; }
  [[nodiscard]] auto size() const { return m_size; }
  [[nodiscard]] auto capacity() const { return m_capacity; }

  void reserve(size_t size);
  void resize(size_t size);
  void clear() { m_size = 0; }

  void gpu_malloc(size_t n);
  void gpu_malloc(size_t n, const T& initial);
  void set_usage (usage_t);

  void push_back(T&&);
  void pop_back();

  template <typename... Ts>
  void emplace_back(Ts&&...);

  [[nodiscard]] auto r_at(size_t idx) const { return relement_t(this, idx); }
  [[nodiscard]] auto rw_at(size_t idx) { return rwelement_t(this, idx); }

  [[nodiscard]] auto at(size_t idx) { return rw_at(idx); }
  [[nodiscard]] auto at(size_t idx) const { return r_at(idx); }

  [[nodiscard]] auto operator[](size_t idx) { return at(idx); }
  [[nodiscard]] auto operator[](size_t idx) const { return at(idx); }

  [[nodiscard]] auto front() { return at(0); }
  [[nodiscard]] auto front() const { return at(0); }

  [[nodiscard]] auto back() { return at(m_size - 1); }
  [[nodiscard]] auto back() const { return at(m_size - 1); }

  [[nodiscard]] auto begin() { return iterator_t(this, 0); }
  [[nodiscard]] auto end() { return iterator_t(this, m_size); }

  [[nodiscard]] auto begin() const { return const_iterator_t(this, 0); }
  [[nodiscard]] auto end() const { return const_iterator_t(this, m_size); }

  [[nodiscard]] auto rmap() const { return rmap_t(this, 0, m_size); }
  [[nodiscard]] auto wmap() { return wmap_t(this, 0, m_size); }
  [[nodiscard]] auto rwmap() { return rwmap_t(this, 0, m_size); }

  [[nodiscard]] auto map() { return rwmap_t(this, 0, m_size); }
  [[nodiscard]] auto map() const { return rmap_t(this, 0, m_size); }

  [[nodiscard]] auto map(size_t offset, size_t length) {
    return rwmap_t(this, offset, length);
  }
  [[nodiscard]] auto map(size_t offset, size_t length) const {
    return rmap_t(this, offset, length);
  }
};
//==============================================================================
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(usage_t usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(const buffer& other) : buffer(other.m_usage) {
  m_usage = other.m_usage;
  copy(other);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(buffer&& other) noexcept
    : parent_t{std::move(other)},
      m_size(std::exchange(other.m_size, 0)),
      m_capacity(std::exchange(other.m_capacity, 0)),
      m_usage(other.m_usage) {}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::operator=(const buffer& other)
    -> buffer& {
  m_usage = other.m_usage;
  copy(other);
  return *this;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::operator=(buffer&& other) noexcept -> buffer& {
  parent_t::operator=(std::move(other));
  std::swap(m_size, other.m_size);
  std::swap(m_capacity, other.m_capacity);
  std::swap(m_usage, other.m_usage);
  return *this;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::operator=(const std::vector<T>& data) -> buffer& {
  auto mapped = map();
  mapped = data;
  mapped.unmap();
  return *this;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(size_t n, usage_t usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
  gpu_malloc(n);
  m_size = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(size_t n, const T& initial, usage_t usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
  gpu_malloc(n, initial);
  m_size = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(const std::vector<T>& data, usage_t usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
  upload_data(data);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::~buffer() {
  destroy_handle();
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::create_handle() {
  gl::create_buffers(1, &id_ref());
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::destroy_handle() {
  if (id() != 0) { gl::delete_buffers(1, &id_ref()); }
  set_id(0);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::upload_data(const T& data) {
  if constexpr (std::is_arithmetic_v<T>) {
    using s = tex::settings<T, R>;
    gl::clear_named_buffer_data(id(), s::internal_format, s::format, s::type,
                                &data);
  } else {
    std::vector<T> data(m_capacity, data);
    gl::named_buffer_data(this->id(), data_size * m_capacity, data.data(),
                          m_usage);
  }
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::upload_data(const std::vector<T>& data) {
  if (capacity() < data.size()) {
    // reallocate new memory
    gl::named_buffer_data(id(), data_size * data.size(), data.data(),
                          m_usage);
    m_size = m_capacity = data.size();
  } else {
    // just update buffer
    gl::named_buffer_data(id(), data_size * data.size(), data.data(), m_usage);
    m_size = data.size();
  }
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::reserve(size_t size) {
  if (capacity() < size) {
    this_t tmp(*this);
    gpu_malloc(size);
    copy(tmp);
  }
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::resize(size_t size) {
  if (capacity() < size) { reserve(size); }
  m_size = size;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::download_data() const -> std::vector<T> {
  rmap_t         map(this, 0, size());
  std::vector<T> data(size());
  std::copy(map.begin(), map.end(), data.begin());
  return data;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::gpu_malloc(size_t n) {
  gl::named_buffer_data<void>(this->id(), data_size * n, nullptr, m_usage);
  m_capacity = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::gpu_malloc(size_t n, const T& initial) {
  if constexpr (std::is_arithmetic_v<T>) {
    gl::named_buffer_data<void>(this->id(), data_size * n, nullptr, m_usage);
    upload_data(initial);
  } else {
    std::vector<T> data(n, initial);
    gl::named_buffer_data(this->id(), data_size * n, data.data(), m_usage);
  }
  m_capacity = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::set_usage(usage_t u) {
  m_usage = u;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::bind() const {
  gl::bind_buffer(array_type, id());
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::unbind() {
  gl::bind_buffer(array_type, 0);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::copy(const this_t& other) {
  if (capacity() < other.size()) { gpu_malloc(other.size()); }
  gl::copy_named_buffer_sub_data(other.id(), id(), 0, 0,
                                 data_size * other.size());
  m_size = other.size();
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::push_back(T&& t) {
  if (m_capacity < m_size + 1) { reserve(std::max<size_t>(m_size * 2, 1)); }
  ++m_size;
  at(m_size - 1) = std::forward<T>(t);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
void buffer<array_type, T>::pop_back() {
  --m_size;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
template <typename... Ts>
void buffer<array_type, T>::emplace_back(Ts&&... ts) {
  if (m_capacity < m_size + 1) {
    // reallocate
    this_t tmp(*this);
    gpu_malloc(m_size * 2.0);
    copy(tmp);
  }
  at(size()) = T(std::forward<Ts>(ts)...);
  ++m_size;
}
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
