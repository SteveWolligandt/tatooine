#ifndef TATOOINE_FLOWEXPLORER_UUID_HOLDER_H
#define TATOOINE_FLOWEXPLORER_UUID_HOLDER_H
//==============================================================================
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/functional/hash.hpp>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
template <typename Id>
struct uuid_holder {
 private:
  Id m_id;

 public:
  uuid_holder(size_t const id) : m_id{id} {}
  uuid_holder(Id const& id) : m_id{id} {}
  uuid_holder()
      : m_id{boost::hash<boost::uuids::uuid>{}(
            boost::uuids::random_generator()())} {}

  auto get_id() const -> const auto& {
    return m_id;
  }
  auto get_id() -> auto& {
    return m_id;
  }
  auto get_id_number() const {
    return m_id.Get();
  }
  auto set_id(size_t const new_id) {
    m_id = new_id;
  }
  constexpr auto operator==(uuid_holder<Id> const& other) const -> bool {
    return get_id() == other.get_id();
  }
  constexpr auto operator==(Id const& id) const -> bool {
    return get_id() == id;
  }
  constexpr auto operator==(size_t const id) const -> bool {
    return get_id() == id;
  }
  constexpr auto equals() {return equals_t{*this};}
  struct equals_t {
    uuid_holder<Id> const& m_id;
    constexpr auto operator()(uuid_holder<Id> const& other) const -> bool {
      return m_id == other.get_id();
    }
    constexpr auto operator()(Id const& id) const -> bool {
      return m_id == id;
    }
    constexpr auto operator()(size_t const id) const -> bool {
      return m_id == id;
    }
    constexpr auto operator()(std::unique_ptr<uuid_holder<Id>> const& id) const
        -> bool {
      return m_id == *id;
    }
  };
};
//==============================================================================
struct uuid_equals {
  template <typename Id>
  constexpr auto operator()(uuid_holder<Id> const& lhs,
                            uuid_holder<Id> const& rhs) {
    return lhs == rhs;
  }
  template <typename Id>
  constexpr auto operator()(Id const& rhs, uuid_holder<Id> const& lhs) {
    return lhs == rhs;
  }
  template <typename Id>
  constexpr auto operator()(size_t const rhs, uuid_holder<Id> const& lhs) {
    return lhs == rhs;
  }
  template <typename Id>
  constexpr auto operator()(uuid_holder<Id> const& lhs, Id const& rhs) {
    return lhs == rhs;
  }
  template <typename Id>
  constexpr auto operator()(uuid_holder<Id> const& lhs, size_t const rhs) {
    return lhs == rhs;
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
