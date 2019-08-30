#ifndef TATOOINE_PROPERTY_H
#define TATOOINE_PROPERTY_H

#include <memory>
#include <vector>

//==============================================================================
namespace tatooine {
//==============================================================================

struct property {
  property()                      = default;
  property(const property& other) = default;
  property(property&& other)      = default;
  property& operator=(const property&) = default;
  property& operator=(property&&) = default;
  //============================================================================

  //----------------------------------------------------------------------------
  /// Destructor.
  virtual ~property() {}

  //----------------------------------------------------------------------------
  /// Reserve memory for n elements.
  virtual void reserve(size_t n) = 0;

  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void resize(size_t n) = 0;

  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void push_back() = 0;

  //----------------------------------------------------------------------------
  /// Resize storage to hold n elements.
  virtual void erase(size_t) = 0;

  //----------------------------------------------------------------------------
  /// Free unused memory.
  virtual void clear() = 0;

  //----------------------------------------------------------------------------
  /// for identifying type.
  virtual const std::type_info& type() const = 0;

  //----------------------------------------------------------------------------
  virtual std::unique_ptr<property> clone() const = 0;
};

//==============================================================================
template <typename T>
struct property_type : property {
  using this_t                 = property_type<T>;
  using vec_t                  = std::vector<T>;
  using value_type             = typename vec_t::value_type;
  using allocator_type         = typename vec_t::allocator_type;
  using size_type              = typename vec_t::size_type;
  using difference_type        = typename vec_t::difference_type;
  using reference              = typename vec_t::reference;
  using const_reference        = typename vec_t::const_reference;
  using pointer                = typename vec_t::pointer;
  using const_pointer          = typename vec_t::const_pointer;
  using iterator               = typename vec_t::iterator;
  using const_iterator         = typename vec_t::const_iterator;
  using reverse_iterator       = typename vec_t::reverse_iterator;
  using const_reverse_iterator = typename vec_t::const_reverse_iterator;

  //============================================================================
 private:
  vec_t m_data;
  T     m_value;

  //============================================================================
 public:
  property_type(const T& value = T{}) : m_value(value) {}
  //----------------------------------------------------------------------------
  property_type(const property_type& other)
      : property{other}, m_data(other.m_data), m_value(other.m_value) {}
  //----------------------------------------------------------------------------
  property_type(property_type&& other)
      : property{std::move(other)},
        m_data(std::move(other.m_data)),
        m_value(std::move(other.m_value)) {}

  //----------------------------------------------------------------------------
  auto& operator=(const property_type& other) {
    property::operator=(other);
    m_data            = other.m_data;
    m_value           = other.m_value;
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(property_type&& other) {
    property::operator=(std::move(other));
    m_data            = std::move(other.m_data);
    m_value           = std::move(other.m_value);
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
  auto&       front() { return m_data.front(); }
  const auto& front() const { return m_data.front(); }
  //----------------------------------------------------------------------------
  auto&       back() { return m_data.back(); }
  const auto& back() const { return m_data.back(); }

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
  /// Get pointer to array (does not work for T==bool)
  auto data() const { return m_data.data(); }

  //----------------------------------------------------------------------------
  auto size() const { return m_data.size(); }

  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto& at(size_t idx) {
    assert(idx < m_data.size());
    return m_data.at(idx);
  }

  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  const auto& at(size_t idx) const {
    assert(idx < m_data.size());
    return m_data.at(idx);
  }

  //----------------------------------------------------------------------------
  /// Access the i'th element.
  auto& operator[](size_t idx) {
    assert(idx < m_data.size());
    return m_data[idx];
  }

  //----------------------------------------------------------------------------
  /// Const access to the i'th element.
  const auto& operator[](size_t idx) const {
    assert(idx < m_data.size());
    return m_data[idx];
  }

  //----------------------------------------------------------------------------
  const std::type_info& type() const override { return typeid(T); }

  //----------------------------------------------------------------------------
  virtual std::unique_ptr<property> clone() const override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
