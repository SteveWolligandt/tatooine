#ifndef TATOOINE_PROPERTY_H
#define TATOOINE_PROPERTY_H
//==============================================================================
#include <cassert>
#include <deque>
#include <memory>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Handle, typename T>
struct typed_vector_property;
//==============================================================================
template <typename Handle>
struct vector_property {
  using this_type = vector_property<Handle>;
  //============================================================================
  vector_property()                                 = default;
  vector_property(const vector_property& other)     = default;
  vector_property(vector_property&& other) noexcept = default;
  auto operator=(const vector_property&) -> vector_property& = default;
  auto operator=(vector_property&&) noexcept -> vector_property& = default;
  //============================================================================
  /// Destructor.
  virtual ~vector_property() = default;
  //----------------------------------------------------------------------------
  /// Reserve memory for n elements.
  virtual void reserve(size_t n) = 0;
  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void resize(size_t n) = 0;
  //----------------------------------------------------------------------------
  /// pushes element at back
  virtual void push_back() = 0;
  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void erase(size_t) = 0;
  //----------------------------------------------------------------------------
  /// Free unused memory.
  virtual void clear() = 0;
  //----------------------------------------------------------------------------
  /// for identifying type.
  [[nodiscard]] virtual auto type() const -> const std::type_info& = 0;
  //----------------------------------------------------------------------------
  template <typename T>
  auto holds_type() const {
    return type() == typeid(T);
  }
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_type> = 0;
  //----------------------------------------------------------------------------
  template <typename T>
  auto cast_to_typed() -> decltype(auto) {
    return *static_cast<typed_vector_property<Handle, T>*>(this);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto cast_to_typed() const -> decltype(auto) {
    return *static_cast<typed_vector_property<Handle, T> const*>(this);
  }
};
//==============================================================================
template <typename Handle, typename T>
struct typed_vector_property : vector_property<Handle> {
  using this_type              = typed_vector_property<Handle, T>;
  using parent_type            = vector_property<Handle>;
  using container_t            = std::vector<T>;
  using value_type             = typename container_t::value_type;
  using allocator_type         = typename container_t::allocator_type;
  using size_type              = typename container_t::size_type;
  using difference_type        = typename container_t::difference_type;
  using reference              = typename container_t::reference;
  using const_reference        = typename container_t::const_reference;
  using pointer                = typename container_t::pointer;
  using const_pointer          = typename container_t::const_pointer;
  using iterator               = typename container_t::iterator;
  using const_iterator         = typename container_t::const_iterator;
  using reverse_iterator       = typename container_t::reverse_iterator;
  using const_reverse_iterator = typename container_t::const_reverse_iterator;
  //============================================================================
 private:
  container_t m_data;
  T           m_value;
  //============================================================================
 public:
  explicit typed_vector_property(const T& value = T{}) : m_value{value} {}
  //----------------------------------------------------------------------------
  typed_vector_property(const typed_vector_property& other)
      : parent_type{other}, m_data{other.m_data}, m_value{other.m_value} {}
  //----------------------------------------------------------------------------
  typed_vector_property(typed_vector_property&& other) noexcept
      : parent_type{std::move(other)},
        m_data{std::move(other.m_data)},
        m_value{std::move(other.m_value)} {}
  //----------------------------------------------------------------------------
  auto operator=(const typed_vector_property& other) -> auto& {
    parent_type::operator=(other);
    m_data               = other.m_data;
    m_value              = other.m_value;
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(typed_vector_property&& other) noexcept -> auto& {
    parent_type::operator=(std::move(other));
    m_data               = std::move(other.m_data);
    m_value              = std::move(other.m_value);
    return *this;
  }
  //============================================================================
  auto size() { return m_data.size(); }
  //----------------------------------------------------------------------------
  void reserve(size_t n) override { m_data.reserve(n); }
  //----------------------------------------------------------------------------
  void resize(size_t n) override { m_data.resize(n, m_value); }
  //----------------------------------------------------------------------------
  void push_back() override { m_data.push_back(m_value); }
  void push_back(const T& value) { m_data.push_back(value); }
  //----------------------------------------------------------------------------
  auto front() -> auto& { return m_data.front(); }
  auto front() const -> const auto& { return m_data.front(); }
  //----------------------------------------------------------------------------
  auto back() -> auto& { return m_data.back(); }
  auto back() const -> const auto& { return m_data.back(); }
  //----------------------------------------------------------------------------
  auto erase(iterator pos) { return m_data.erase(pos); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto erase(const_iterator pos) { return m_data.erase(pos); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto erase(iterator first, iterator last) {
    return m_data.erase(first, last);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto erase(const_iterator first, const_iterator last) {
    return m_data.erase(first, last);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void erase(size_t i) override { erase(begin() + i); }
  //----------------------------------------------------------------------------
  auto begin() { return m_data.begin(); }
  auto begin() const { return m_data.begin(); }
  //----------------------------------------------------------------------------
  auto end() { return m_data.end(); }
  auto end() const { return m_data.end(); }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  void emplace_back(Ts&&... ts) {
    m_data.emplace_back(std::forward<Ts>(ts)...);
  }
  //----------------------------------------------------------------------------
  void clear() override {
    m_data.clear();
    m_data.shrink_to_fit();
  }
  //----------------------------------------------------------------------------
  auto data() const { return m_data.data(); }
  //----------------------------------------------------------------------------
  auto internal_container() const -> auto const& { return m_data; }
  //----------------------------------------------------------------------------
  auto size() const { return m_data.size(); }
  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto at(std::size_t const i) -> auto& {
    assert(i < m_data.size());
    return m_data.at(i);
  }
  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  auto at(std::size_t const i) const -> const auto& {
    assert(i < m_data.size());
    return m_data.at(i);
  }
  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto at(Handle handle) -> auto& {
    assert(handle.index() < m_data.size());
    return m_data.at(handle.index());
  }
  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  auto at(Handle handle) const -> const auto& {
    assert(handle.index() < m_data.size());
    return m_data.at(handle.index());
  }
  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto operator[](Handle handle) -> auto& {
    assert(handle.index() < m_data.size());
    return m_data[handle.index()];
  }
  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  auto operator[](Handle handle) const -> const auto& {
    assert(handle.index() < m_data.size());
    return m_data[handle.index()];
  }
  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto operator[](std::size_t const i) -> auto& {
    assert(i < m_data.size());
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  auto operator[](std::size_t const i) const -> const auto& {
    assert(i < m_data.size());
    return m_data[i];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto type() const -> const std::type_info& override {
    return typeid(T);
  }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<parent_type> override {
    return std::unique_ptr<this_type>{new this_type{*this}};
  }
};
//==============================================================================
template <typename Handle>
struct deque_property {
  using this_type = deque_property<Handle>;
  //============================================================================
  deque_property()                                = default;
  deque_property(const deque_property& other)     = default;
  deque_property(deque_property&& other) noexcept = default;
  auto operator=(const deque_property&) -> deque_property& = default;
  auto operator=(deque_property&&) noexcept -> deque_property&;
  //============================================================================
  /// Destructor.
  virtual ~deque_property() = default;
  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void resize(size_t n) = 0;
  //----------------------------------------------------------------------------
  /// pushes element at back
  virtual void push_back() = 0;
  //----------------------------------------------------------------------------
  /// pushes element at front
  virtual void push_front() = 0;
  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void erase(size_t) = 0;
  //----------------------------------------------------------------------------
  /// Free unused memory.
  virtual void clear() = 0;
  //----------------------------------------------------------------------------
  /// for identifying type.
  [[nodiscard]] virtual auto type() const -> const std::type_info& = 0;
  //----------------------------------------------------------------------------
  template <typename T>
  auto holds_type() const {
    return type() == typeid(T);
  }
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_type> = 0;

};
//==============================================================================
template <typename Handle, typename T>
struct typed_deque_property : deque_property<Handle> {
  using this_type              = typed_deque_property<Handle, T>;
  using parent_type            = deque_property<Handle>;
  using container_t            = std::deque<T>;
  using value_type             = typename container_t::value_type;
  using allocator_type         = typename container_t::allocator_type;
  using size_type              = typename container_t::size_type;
  using difference_type        = typename container_t::difference_type;
  using reference              = typename container_t::reference;
  using const_reference        = typename container_t::const_reference;
  using pointer                = typename container_t::pointer;
  using const_pointer          = typename container_t::const_pointer;
  using iterator               = typename container_t::iterator;
  using const_iterator         = typename container_t::const_iterator;
  using reverse_iterator       = typename container_t::reverse_iterator;
  using const_reverse_iterator = typename container_t::const_reverse_iterator;
  //============================================================================
 private:
  container_t m_data;
  T           m_value;
  //============================================================================
 public:
  explicit typed_deque_property(const T& value = T{}) : m_value{value} {}
  //----------------------------------------------------------------------------
  typed_deque_property(const typed_deque_property& other)
      : parent_type{other}, m_data{other.m_data}, m_value{other.m_value} {}
  //----------------------------------------------------------------------------
  typed_deque_property(typed_deque_property&& other) noexcept
      : parent_type{std::move(other)},
        m_data{std::move(other.m_data)},
        m_value{std::move(other.m_value)} {}
  //----------------------------------------------------------------------------
  auto operator=(const typed_deque_property& other) -> auto& {
    parent_type::operator=(other);
    m_data               = other.m_data;
    m_value              = other.m_value;
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(typed_deque_property&& other) noexcept -> auto& {
    parent_type::operator=(std::move(other));
    m_data               = std::move(other.m_data);
    m_value              = std::move(other.m_value);
    return *this;
  }
  //============================================================================
  auto size() { return m_data.size(); }
  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  void resize(size_t n) override { m_data.resize(n); }
  //----------------------------------------------------------------------------
  void push_back() override { m_data.push_back(m_value); }
  void push_back(const T& value) { m_data.push_back(value); }
  //----------------------------------------------------------------------------
  void push_front() override { m_data.push_front(m_value); }
  void push_front(const T& value) { m_data.push_front(value); }
  //----------------------------------------------------------------------------
  auto front() -> auto& { return m_data.front(); }
  auto front() const -> const auto& { return m_data.front(); }
  //----------------------------------------------------------------------------
  auto back() -> auto& { return m_data.back(); }
  auto back() const -> const auto& { return m_data.back(); }
  //----------------------------------------------------------------------------
  auto erase(iterator pos) { return m_data.erase(pos); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto erase(const_iterator pos) { return m_data.erase(pos); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto erase(iterator first, iterator last) {
    return m_data.erase(first, last);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto erase(const_iterator first, const_iterator last) {
    return m_data.erase(first, last);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void erase(size_t i) override { erase(next(begin(), i)); }
  //----------------------------------------------------------------------------
  auto begin() { return m_data.begin(); }
  auto begin() const { return m_data.begin(); }
  //----------------------------------------------------------------------------
  auto end() { return m_data.end(); }
  auto end() const { return m_data.end(); }
  //----------------------------------------------------------------------------
  template <typename... Ts>
  auto emplace_back(Ts&&... ts) -> void {
    m_data.emplace_back(std::forward<Ts>(ts)...);
  }
  //----------------------------------------------------------------------------
  auto clear() -> void override {
    m_data.clear();
    m_data.shrink_to_fit();
  }
  //----------------------------------------------------------------------------
  auto internal_container() const -> const auto& { return m_data; }
  //----------------------------------------------------------------------------
  auto data() const { return m_data.data(); }
  //----------------------------------------------------------------------------
  auto size() const { return m_data.size(); }
  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto at(Handle handle) -> auto& {
    assert(handle.index() < m_data.size());
    return m_data.at(handle.index());
  }
  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  auto at(Handle handle) const -> const auto& {
    assert(handle.index() < m_data.size());
    return m_data.at(handle.index());
  }
  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto operator[](Handle handle) -> auto& {
    assert(handle.index() < m_data.size());
    return m_data[handle.index()];
  }
  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  auto operator[](Handle handle) const -> const auto& {
    assert(handle.index() < m_data.size());
    return m_data[handle.index()];
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto type() const -> const std::type_info& override {
    return typeid(T);
  }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<parent_type> override {
    return std::unique_ptr<this_type>{new this_type{*this}};
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
