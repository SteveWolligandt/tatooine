#ifndef TATOOINE_CLONABLE_H
#define TATOOINE_CLONABLE_H

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Parent, typename This>
struct clonable {
  std::unique_ptr<Parent> clone() const override {
    return std::unique_ptr<This>(new This{*this});
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
