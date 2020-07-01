#include <tatooine/packages.h>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#if TATOOINE_YAVIN_AVAILABLE
#include <yavin/context.h>
#endif
//==============================================================================
auto main(int argc, char** argv) -> int {
# if TATOOINE_YAVIN_AVAILABLE
  yavin::context ctx{4, 5};
# endif
  return Catch::Session().run(argc, argv);
}
