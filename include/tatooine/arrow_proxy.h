#ifndef TATOOINE_ARROW_PROXY_H
#define TATOOINE_ARROW_PROXY_H
//==============================================================================
namespace tatooine {
//==============================================================================
/// from https://quuxplusone.github.io/blog/2019/02/06/arrow-proxy/
template <typename Reference>
struct arrow_proxy {
  Reference  r;
  auto       operator->() -> Reference * { return &r; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
