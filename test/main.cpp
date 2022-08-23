#include <tatooine/available_libraries.h>

#include <catch2/catch_session.hpp>
//#include <reporters/catch_reporter_tap.hpp>
//#include <reporters/catch_reporter_teamcity.hpp>
//#include <reporters/catch_reporter_automake.hpp>
//#include <reporters/catch_reporter_sonarqube.hpp>

#if TATOOINE_GL_AVAILABLE
#include <tatooine/gl/window.h>
#endif
//==============================================================================
auto main(int argc, char** argv) -> int {
# if TATOOINE_GL_AVAILABLE
  auto ctx = tatooine::gl::context{};
# endif
  Catch::Session().run(argc, argv);
  return 0;
}
