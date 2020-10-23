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
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
