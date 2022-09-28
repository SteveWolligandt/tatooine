#ifndef TATOOINE_GL_BUFFER
#define TATOOINE_GL_BUFFER
//==============================================================================
#include <tatooine/gl/buffer_usage.h>
#include <tatooine/gl/glfunctions.h>
#include <tatooine/gl/idholder.h>
#include <tatooine/gl/mutexhandler.h>
#include <tatooine/gl/texsettings.h>

#include <mutex>
#include <vector>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
template <GLsizei array_type, typename T>
class buffer;
//==============================================================================
template <GLsizei ArrayType, typename T, GLbitfield Access>
class buffer_map {
 public:
  using value_type                 = T;
  static constexpr auto access     = Access;
  static constexpr auto array_type = ArrayType;
  using buffer_type                = buffer<array_type, T>;
  static constexpr auto data_size  = buffer_type::data_size;

 private:
  const buffer_type* m_buffer;
  std::size_t        m_offset;
  std::size_t        m_length;
  T*                 m_gpu_mapping;
  bool               m_unmapped = false;

 public:
  /// constructor gets a mapping to gpu_buffer
  buffer_map(const buffer_type* buffer, std::size_t offset, std::size_t length)
      : m_buffer(buffer), m_offset(offset), m_length(length) {
    m_gpu_mapping = (T*)gl::map_named_buffer_range(
        m_buffer->id(), data_size * offset,
        static_cast<GLsizei>(data_size * m_length), access);
    detail::mutex::gl_call.lock();
  }
  //============================================================================
  buffer_map(const buffer_map&) = delete;
  buffer_map(buffer_map&&)      = delete;
  //============================================================================
  auto operator=(const buffer_map&) -> buffer_map& = delete;
  auto operator=(buffer_map&&) -> buffer_map&      = delete;
  //============================================================================
  auto operator=(const std::vector<T>& data) -> buffer_map& {
    assert(size(data) == m_buffer->size());
    for (std::size_t i = 0; i < size(data); ++i) {
      at(i) = data[i];
    }
    return &this;
  }
  //============================================================================
  /// destructor unmaps the buffer
  ~buffer_map() { unmap(); }
  //============================================================================
  auto unmap() {
    detail::mutex::gl_call.unlock();
    if (!m_unmapped) {
      gl::unmap_named_buffer(m_buffer->id());
      m_unmapped = true;
    }
  }
  //============================================================================
  auto at(std::size_t i) -> auto& { return m_gpu_mapping[i]; }
  auto at(std::size_t i) const -> const auto& { return m_gpu_mapping[i]; }
  //============================================================================
  auto front() -> auto& { return at(0); }
  auto front() const -> const auto& { return at(0); }
  //============================================================================
  auto back() -> auto& { return at(m_length - 1); }
  auto back() const -> const auto& { return at(m_length - 1); }
  //============================================================================
  auto operator[](std::size_t i) -> auto& { return at(i); }
  auto operator[](std::size_t i) const -> const auto& { return at(i); }
  //============================================================================
  auto begin() { return m_gpu_mapping; }
  auto begin() const { return m_gpu_mapping; }
  //============================================================================
  auto end() { return m_gpu_mapping + m_length; }
  auto end() const { return m_gpu_mapping + m_length; }
  //============================================================================
  auto offset() const { return m_offset; }
  auto length() const { return m_length; }
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
  using buffer_type   = buffer<array_type, T>;
  using read_map_type = rbuffer_map<array_type, T>;

 private:
  const buffer_type* m_buffer;
  std::size_t        m_idx;

 public:
  rbuffer_map_element(const buffer_type* buffer, std::size_t idx)
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
    read_map_type map(m_buffer, m_idx, 1);
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
inline auto operator<<(std::ostream&                       out,
                       rbuffer_map_element<array_type, T>& data) -> auto& {
  out << data.download();
  return out;
}

//==============================================================================
/// Returned by buffer::operator[] for reading and writing single elements
template <GLsizei array_type, typename T>
class rwbuffer_map_element {
 public:
  using buffer_type   = buffer<array_type, T>;
  using read_map_type = rbuffer_map<array_type, T>;

 private:
  const buffer_type* m_buffer;
  std::size_t        m_idx;

 public:
  rwbuffer_map_element(const buffer_type* buffer, std::size_t idx)
      : m_buffer{buffer}, m_idx{idx} {}
  rwbuffer_map_element(const rwbuffer_map_element& other)     = default;
  rwbuffer_map_element(rwbuffer_map_element&& other) noexcept = default;

  auto operator=(const rwbuffer_map_element& other)
      -> rwbuffer_map_element& = default;
  auto operator=(rwbuffer_map_element&& other) noexcept
      -> rwbuffer_map_element& = default;

  ~rwbuffer_map_element() = default;
  /// for assigning single gpu data element.
  auto operator=(T const& data) -> auto& {
    if (m_idx * buffer_type::data_size >=
        m_buffer->size() * buffer_type::data_size) {
      std::cout << "attention!\n";
    }
    gl::named_buffer_sub_data(m_buffer->id(), m_idx * buffer_type::data_size,
                              buffer_type::data_size, &data);
    return *this;
  }

  /// for accessing single gpu data element.
  explicit operator T() const { return download(); }

  [[nodiscard]] auto download() const {
    read_map_type map(m_buffer, m_idx, 1);
    return map.front();
  }
  auto operator==(const T& t) const -> bool { return download() == t; }
  auto operator!=(const T& t) const -> bool { return !operator==(t); }
  auto operator>(const T& t) const -> bool { return download() > t; }
  auto operator>=(const T& t) const -> bool { return download() >= t; }
  auto operator<(const T& t) const -> bool { return download() < t; }
  auto operator<=(const T& t) const -> bool { return download() <= t; }
};
//==============================================================================
/// Returned by buffer::operator[] for reading and writing single elements
/// Returned by buffer::operator[] for reading and writing single elements
template <GLsizei array_type, typename T>
class wbuffer_map_element {
 public:
  using buffer_type = buffer<array_type, T>;

 private:
  const buffer_type* m_buffer;
  std::size_t        m_idx;

 public:
  wbuffer_map_element(const buffer_type* buffer, std::size_t idx)
      : m_buffer{buffer}, m_idx{idx} {}
  wbuffer_map_element(const wbuffer_map_element& other)     = default;
  wbuffer_map_element(wbuffer_map_element&& other) noexcept = default;

  auto operator=(const wbuffer_map_element& other)
      -> wbuffer_map_element& = default;
  auto operator=(wbuffer_map_element&& other) noexcept
      -> wbuffer_map_element& = default;

  ~wbuffer_map_element() = default;
  /// for assigning single gpu data element.
  auto operator=(T const& data) -> auto& {
    if (m_idx * buffer_type::data_size >=
        m_buffer->size() * buffer_type::data_size) {
      std::cout << "attention!\n";
    }
    gl::named_buffer_sub_data(m_buffer->id(), m_idx * buffer_type::data_size,
                              buffer_type::data_size, &data);
    return *this;
  }
};
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
inline auto operator<<(std::ostream&                        out,
                       rwbuffer_map_element<array_type, T>& data) -> auto& {
  out << data.download();
  return out;
}

//==============================================================================
/// non-const buffer iterator
template <GLsizei array_type, typename T>
class buffer_iterator {
 public:
  using buffer_type = buffer<array_type, T>;
  //----------------------------------------------------------------------------
  // iterator typedefs
  using value_type        = T;
  using reference         = T&;
  using pointer           = T*;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;
  //----------------------------------------------------------------------------
  buffer_iterator(buffer_type* buffer, std::size_t idx)
      : m_buffer(buffer), m_idx(idx) {}
  //----------------------------------------------------------------------------
  buffer_iterator(buffer_iterator const&)     = default;
  buffer_iterator(buffer_iterator&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(buffer_iterator const&) -> buffer_iterator&     = default;
  auto operator=(buffer_iterator&&) noexcept -> buffer_iterator& = default;
  //----------------------------------------------------------------------------
  ~buffer_iterator() = default;
  //----------------------------------------------------------------------------
  /// get the buffer element the iterator refers to
  auto operator*() const -> T { return rbuffer_map_element(m_buffer, m_idx); }
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
  auto operator++() -> auto& {
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
  buffer_type* m_buffer;
  std::size_t  m_idx;
};

//==============================================================================
/// const buffer iterator
template <GLsizei array_type, typename T>
class cbuffer_iterator {
 public:
  using buffer_type = buffer<array_type, T>;

  // iterator typedefs
  using value_type        = T;
  using reference         = T&;
  using pointer           = T*;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;
  //----------------------------------------------------------------------------
  cbuffer_iterator(const buffer_type* buffer, std::size_t idx)
      : m_buffer(buffer), m_idx(idx) {}
  //----------------------------------------------------------------------------
  cbuffer_iterator(const cbuffer_iterator& other)     = default;
  cbuffer_iterator(cbuffer_iterator&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(const cbuffer_iterator& other) -> cbuffer_iterator& = default;
  auto operator=(cbuffer_iterator&& other) noexcept
      -> cbuffer_iterator& = default;
  //----------------------------------------------------------------------------
  ~cbuffer_iterator() = default;
  //----------------------------------------------------------------------------
  /// get the buffer element the iterator refers to
  auto operator*() const -> T { return rbuffer_map_element(m_buffer, m_idx); }
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
  const buffer_type* m_buffer;
  std::size_t        m_idx;
};

//==============================================================================
/// buffer base class for each of the OpenGL buffer types
template <GLsizei ArrayType, typename T>
class buffer : public id_holder<GLuint> {
 public:
  using parent_type = id_holder<GLuint>;
  using parent_type::id;
  friend class rbuffer_map_element<ArrayType, T>;
  friend class rwbuffer_map_element<ArrayType, T>;

  constexpr static auto    array_type = ArrayType;
  constexpr static GLsizei data_size  = static_cast<GLsizei>(sizeof(T));

  using this_type  = buffer<array_type, T>;
  using value_type = T;

  using read_only_element_type  = rbuffer_map_element<array_type, T>;
  using read_write_element_type = rwbuffer_map_element<array_type, T>;
  using write_only_element_type = wbuffer_map_element<array_type, T>;

  using iterator       = buffer_iterator<array_type, T>;
  using const_iterator = cbuffer_iterator<array_type, T>;

  using read_map_type       = rbuffer_map<array_type, T>;
  using write_map_type      = wbuffer_map<array_type, T>;
  using read_write_map_type = rwbuffer_map<array_type, T>;

 private:
  GLsizei      m_size     = 0;
  GLsizei      m_capacity = 0;
  buffer_usage m_usage;

 public:
  explicit buffer(buffer_usage usage);
  buffer(const buffer& other);
  buffer(buffer&& other) noexcept;
  auto operator=(const buffer& other) -> buffer&;
  auto operator=(buffer&& other) noexcept -> buffer&;
  auto operator=(const std::vector<T>& data) -> buffer&;

  buffer(GLsizei n, buffer_usage usage);
  buffer(GLsizei n, const T& initial, buffer_usage usage);
  buffer(const std::vector<T>& data, buffer_usage usage);
  ~buffer();

  auto create_handle() -> void;
  auto destroy_handle() -> void;

  auto               upload_data(const T& data) -> void;
  auto               upload_data(const std::vector<T>& data) -> void;
  [[nodiscard]] auto download_data() const -> std::vector<T>;

  auto        bind() const -> void;
  static auto unbind() -> void;

  auto copy(const this_type& other) -> void;

  [[nodiscard]] auto empty() const -> bool { return m_size == 0; }
  [[nodiscard]] auto size() const { return m_size; }
  [[nodiscard]] auto capacity() const { return m_capacity; }

  auto reserve(GLsizei size) -> void;
  auto resize(GLsizei size) -> void;
  auto clear() { m_size = 0; }

  auto gpu_malloc(GLsizei n) -> void;
  auto gpu_malloc(GLsizei n, const T& initial) -> void;
  auto set_usage(buffer_usage) -> void;

  auto push_back(T const&) -> void;
  auto pop_back() -> void;

  template <typename... Ts>
  auto emplace_back(Ts&&...) -> void;

  [[nodiscard]] auto read_write_element_at(std::size_t idx) {
    return read_write_element_type(this, idx);
  }
  [[nodiscard]] auto read_element_at(std::size_t idx) const {
    return read_only_element_type(this, idx);
  }
  [[nodiscard]] auto write_only_element_at(std::size_t idx) {
    return write_only_element_type(this, idx);
  }

  [[nodiscard]] auto at(std::size_t idx) { return read_write_element_at(idx); }
  [[nodiscard]] auto at(std::size_t idx) const { return read_element_at(idx); }

  [[nodiscard]] auto operator[](std::size_t idx) { return at(idx); }
  [[nodiscard]] auto operator[](std::size_t idx) const { return at(idx); }

  [[nodiscard]] auto front() { return at(0); }
  [[nodiscard]] auto front() const { return at(0); }

  [[nodiscard]] auto back() { return at(m_size - 1); }
  [[nodiscard]] auto back() const { return at(m_size - 1); }

  [[nodiscard]] auto begin() { return iterator(this, 0); }
  [[nodiscard]] auto end() { return iterator(this, m_size); }

  [[nodiscard]] auto begin() const { return const_iterator(this, 0); }
  [[nodiscard]] auto end() const { return const_iterator(this, m_size); }

  [[nodiscard]] auto rmap() const { return read_map_type(this, 0, m_size); }
  [[nodiscard]] auto wmap() { return write_map_type(this, 0, m_size); }
  [[nodiscard]] auto rwmap() { return read_write_map_type(this, 0, m_size); }

  [[nodiscard]] auto map() { return read_write_map_type(this, 0, m_size); }
  [[nodiscard]] auto map() const { return read_map_type(this, 0, m_size); }

  [[nodiscard]] auto map(std::size_t offset, std::size_t length) {
    return read_write_map_type(this, offset, length);
  }
  [[nodiscard]] auto map(std::size_t offset, std::size_t length) const {
    return read_map_type(this, offset, length);
  }
};
//==============================================================================
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(buffer_usage usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(const buffer& other) : buffer{other.m_usage} {
  copy(other);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(buffer&& other) noexcept
    : parent_type{std::move(other)},
      m_size{std::exchange(other.m_size, 0)},
      m_capacity{std::exchange(other.m_capacity, 0)},
      m_usage{other.m_usage} {}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::operator=(const buffer& other) -> buffer& {
  m_usage = other.m_usage;
  copy(other);
  return *this;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::operator=(buffer&& other) noexcept -> buffer& {
  parent_type::operator=(std::move(other));
  std::swap(m_size, other.m_size);
  std::swap(m_capacity, other.m_capacity);
  std::swap(m_usage, other.m_usage);
  return *this;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::operator=(const std::vector<T>& data) -> buffer& {
  auto mapped = map();
  mapped      = data;
  mapped.unmap();
  return *this;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(GLsizei n, buffer_usage usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
  gpu_malloc(n);
  m_size = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(GLsizei n, const T& initial, buffer_usage usage)
    : m_size{}, m_capacity{}, m_usage{usage} {
  create_handle();
  gpu_malloc(n, initial);
  m_size = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
buffer<array_type, T>::buffer(const std::vector<T>& data, buffer_usage usage)
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
auto buffer<array_type, T>::create_handle() -> void {
  gl::create_buffers(1, &id_ref());
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::destroy_handle() -> void {
  if (id() != 0) {
    gl::delete_buffers(1, &id_ref());
  }
  set_id(0);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::upload_data(const T& data) -> void {
  if constexpr (std::is_arithmetic_v<T>) {
    using s = tex::settings<T, R>;
    gl::clear_named_buffer_data(id(), s::internal_format, s::format, s::type,
                                &data);
  } else {
    std::vector<T> data(m_capacity, data);
    gl::named_buffer_data(this->id(), data_size * m_capacity, data.data(),
                          static_cast<GLenum>(m_usage));
  }
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::upload_data(const std::vector<T>& data) -> void {
  auto const s = static_cast<GLsizei>(data_size * data.size());
  if (capacity() < static_cast<GLsizei>(data.size())) {
    // reallocate new memory
    gl::named_buffer_data(id(), s, data.data(), static_cast<GLenum>(m_usage));
    m_size = m_capacity = static_cast<GLsizei>(data.size());
  } else {
    // just update buffer
    gl::named_buffer_data(id(), s, data.data(), static_cast<GLenum>(m_usage));
    m_size = static_cast<GLsizei>(data.size());
  }
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::reserve(GLsizei size) -> void {
  if (capacity() < size) {
    auto tmp = *this;
    gpu_malloc(size);
    copy(tmp);
  }
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::resize(GLsizei size) -> void {
  if (capacity() < size) {
    reserve(size);
  }
  m_size = size;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::download_data() const -> std::vector<T> {
  read_map_type  map(this, 0, size());
  std::vector<T> data(size());
  std::copy(map.begin(), map.end(), data.begin());
  return data;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::gpu_malloc(GLsizei n) -> void {
  auto const s = static_cast<GLsizei>(data_size * n);
  gl::named_buffer_data<void>(this->id(), s, nullptr,
                              static_cast<GLenum>(m_usage));
  m_capacity = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::gpu_malloc(GLsizei n, const T& initial) -> void {
  auto const s = static_cast<GLsizei>(data_size * n);
  if constexpr (std::is_arithmetic_v<T>) {
    gl::named_buffer_data<void>(this->id(), s, nullptr,
                                static_cast<GLenum>(m_usage));
    upload_data(initial);
  } else {
    std::vector<T> data(n, initial);
    gl::named_buffer_data(this->id(), s, data.data(),
                          static_cast<GLenum>(m_usage));
  }
  m_capacity = n;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::set_usage(buffer_usage u) -> void {
  m_usage = u;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::bind() const -> void {
  gl::bind_buffer(array_type, id());
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::unbind() -> void {
  gl::bind_buffer(array_type, 0);
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::copy(const this_type& other) -> void {
  if (capacity() < other.size()) {
    gpu_malloc(other.size());
  }
  gl::copy_named_buffer_sub_data(other.id(), id(), 0, 0,
                                 data_size * other.size());
  m_size = other.size();
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::push_back(T const& t) -> void {
  if (m_capacity < m_size + 1) {
    reserve(std::max<GLsizei>(m_size * 2, 1));
  }
  ++m_size;
  at(m_size - 1) = t;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
auto buffer<array_type, T>::pop_back() -> void {
  --m_size;
}
//------------------------------------------------------------------------------
template <GLsizei array_type, typename T>
template <typename... Ts>
auto buffer<array_type, T>::emplace_back(Ts&&... ts) -> void {
  if (m_capacity < m_size + 1) {
    // reallocate
    this_type tmp(*this);
    gpu_malloc(m_size * 2);
    copy(tmp);
  }
  at(size()) = T(std::forward<Ts>(ts)...);
  ++m_size;
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
