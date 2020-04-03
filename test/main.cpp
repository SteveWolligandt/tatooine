#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <yavin/context.h>
//==============================================================================
auto main(int argc, char** argv) -> int {
  yavin::context ctx{4, 5};
  return Catch::Session().run(argc, argv);
}
