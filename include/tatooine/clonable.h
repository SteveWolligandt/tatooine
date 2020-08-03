#ifndef TATOOINE_CLONABLE_H
#define TATOOINE_CLONABLE_H

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Base>
struct clonable {
  using base_t       = Base;
  using cloned_ptr_t                 = std::unique_ptr<base_t>;
  virtual cloned_ptr_t clone() const = 0;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
