#include <thread>
#include <cstdio>
//==============================================================================
auto foo() {
  std::puts("foo");
  std::this_thread::sleep_for(std::chrono::seconds{3});
}
auto bar() {
  std::puts("bar");
  auto mem = new int();
  delete mem;
  std::this_thread::sleep_for(std::chrono::seconds{2});
}
//==============================================================================
auto main() -> int {
  foo();
  bar();
}
