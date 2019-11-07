#ifndef TATOOINE_CUDA_VECTOR_CUH
#define TATOOINE_CUDA_VECTOR_CUH

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T>
class vector {
  //----------------------------------------------------------------------------
  // class fields
  //----------------------------------------------------------------------------
 public:
  using type = T;
  static constexpr size_t factor = 2;

  //----------------------------------------------------------------------------
  // object fields
  //----------------------------------------------------------------------------
 private:
  T*     m_data = nullptr;
  size_t m_size;
  size_t m_reserved;

  //----------------------------------------------------------------------------
  // ctors / dtor
  //----------------------------------------------------------------------------
 public:
  __device__ vector() {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ vector(size_t n)
      : m_data{static_cast<T*>(malloc(sizeof(T) * n))},
        m_size{n},
        m_reserved{n} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ vector(size_t n, const T& t)
      : m_data{static_cast<T*>(malloc(sizeof(T) * n))},
        m_size{n},
        m_reserved{n} {
    for (size_t i = 0; i < m_size; ++i) { m_data[i] = t; }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ ~vector() { free(m_data); }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  __device__ auto& at(size_t i) { return m_data[i]; }
  __device__ const auto& at(size_t i) const { return m_data[i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ auto& operator[](size_t i) { return m_data[i]; }
  __device__ const auto& operator[](size_t i) const { return m_data[i]; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ void resize(size_t n) {
    if (n > m_reserved) {
      auto new_data = static_cast<T*>(malloc(n * sizeof(T)));
      memcpy(new_data, m_data, sizeof(T) * m_reserved);
      free(m_data);
      m_data     = new_data;
      m_reserved = n;
    }
    m_size = n;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ void reserve(size_t n) {
    if (n > m_reserved) {
      auto new_data = static_cast<T*>(malloc(n * sizeof(T)));
      memcpy(new_data, m_data, sizeof(T) * m_reserved);
      free(m_data);
      m_data     = new_data;
      m_reserved = n;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ auto size() const { return m_size; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ auto reserved() const { return m_reserved; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void push_back(const T& t) {
    if (m_size == m_reserved) {
      auto new_data = static_cast<T*>(malloc(m_size * sizeof(T) * factor));
      memcpy(new_data, m_data, sizeof(T) * m_size);
      free(m_data);
      m_data     = new_data;
      m_reserved = m_size * factor;
    }
    ++m_size;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void pop_back() { --m_size; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& front() const { return at(0); }
  auto& front() { return at(0); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const auto& back() const { return at(m_size - 1); }
  auto& back() { return at(m_size - 1); }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
