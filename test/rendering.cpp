#include <tatooine/rendering/orthographic_camera.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::rendering::test {
//==============================================================================
TEST_CASE("rendering ortho cam") {
  SECTION("center of canonical view volume") {
    auto res         = Vec2<std::size_t>{100, 100};
    auto height      = real_number{5};
    auto near        = real_number{0};
    auto far         = real_number{2};
    auto eye         = vec3{0, 0, -1};
    auto lookat      = vec3{0, 0, 0};
    auto up          = vec3{0, 1, 0};
    auto cam         = orthographic_camera<real_number>{eye,  lookat, up,     height,
                                           near, far,    res(0), res(1)};
    auto query_point = lookat;
    auto projected   = cam.project(query_point);
    for (std::size_t i = 0; i < 2; ++i) {
      CAPTURE(i, query_point, projected);
      REQUIRE(projected(i) ==
              Approx(static_cast<real_number>(res(i) - 1) / 2.0).margin(1e-2));
    }
    auto unprojected = cam.unproject(projected);
    for (std::size_t i = 0; i < 3; ++i) {
      CAPTURE(i, query_point, projected, unprojected);
      REQUIRE(query_point(i) == Approx(unprojected(i)));
    }
  }
  SECTION("arbitrary") {
    auto res         = Vec2<std::size_t>{100, 100};
    auto height      = real_number{5};
    auto near        = real_number{0};
    auto far         = real_number{2};
    auto eye         = vec3{2, 1, 3};
    auto lookat      = vec3{-1, 1, 3};
    auto up          = vec3{0, 1, 0};
    auto cam         = orthographic_camera<real_number>{eye,  lookat, up,     height,
                                           near, far,    res(0), res(1)};
    auto query_point = lookat;
    auto projected   = cam.project(query_point);
    auto unprojected = cam.unproject(projected);
    for (std::size_t i = 0; i < 3; ++i) {
      CAPTURE(i, query_point, projected, unprojected);
      REQUIRE(query_point(i) == Approx(unprojected(i)));
    }
  }
  SECTION("screen to world to screen") {
    auto res         = Vec2<std::size_t>{100, 100};
    auto height      = real_number{5};
    auto near        = real_number{0};
    auto far         = real_number{2};
    auto eye         = vec3{2, 1, 3};
    auto lookat      = vec3{-1, 1, 3};
    auto up          = vec3{0, 1, 0};
    auto cam         = orthographic_camera<real_number>{eye,  lookat, up,     height,
                                           near, far,    res(0), res(1)};
    auto screen_pos  = vec2{0, 0};
    auto unproj      = cam.unproject(screen_pos);
    auto proj        = cam.project(unproj);

    for (std::size_t i = 0; i < 2; ++i) {
      CAPTURE(i, screen_pos, unproj, proj);
      REQUIRE(screen_pos(i) == Approx(proj(i)));
    }
  }
}
//==============================================================================
}  // namespace tatooine::rendering::test
//==============================================================================
