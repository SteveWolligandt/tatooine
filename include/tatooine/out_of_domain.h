#ifndef TATOOINE_FIELD_H
#define TATOOINE_FIELD_H
//╔════════════════════════════════════════════════════════════════════════════╗
#include <stdexcept>
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine {
struct out_of_domain : std::runtime_error {
  out_of_domain() : std::runtime_error{""} {}
};
}  // namespace tatooine
//╚════════════════════════════════════════════════════════════════════════════╝
#endif
