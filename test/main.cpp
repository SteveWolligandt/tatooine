#include <tatooine/packages.h>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
//#include <reporters/catch_reporter_tap.hpp>
//#include <reporters/catch_reporter_teamcity.hpp>
//#include <reporters/catch_reporter_automake.hpp>
//#include <reporters/catch_reporter_sonarqube.hpp>

#if TATOOINE_YAVIN_AVAILABLE
#include <tatooine/rendering/gl/window.h>
#endif
//==============================================================================
auto main(int argc, char** argv) -> int {
# if TATOOINE_YAVIN_AVAILABLE
  //rendering::gl::window ctx{"test", 100, 100};
# endif
  return Catch::Session().run(argc, argv);
}
