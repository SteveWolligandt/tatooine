#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <yavin/context.h>
//==============================================================================
int main(int argc, char** argv) {
  yavin::context ctx{4, 5};
  return Catch::Session().run(argc, argv);
}
