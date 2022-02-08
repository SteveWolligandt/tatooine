#include <catch2/catch.hpp>
#include <tatooine/rendering/orthographic_camera.h>
//==============================================================================
namespace tatooine::rendering::test {
//==============================================================================
TEST_CASE("rendering ortho cam") {
  SECTION("center of canonical view volume") {
    auto res         = Vec2<std::size_t>{100, 100};
    auto height      = 5;
    auto near        = 0;
    auto far         = 2;
    auto eye         = vec3{0, 0, -1};
    auto lookat      = vec3{0, 0, 0};
    auto up          = vec3{0, 1, 0};
    auto cam         = orthographic_camera<real_type>{eye,  lookat, up,     height,
                                           near, far,    res(0), res(1)};
    auto query_point = lookat;
    auto projected   = cam.project(query_point);
    for (std::size_t i = 0; i < 2; ++i) {
      CAPTURE(i, query_point, projected);
      REQUIRE(projected(i) == Approx((res(i) - 1) / 2.0).margin(1e-2));
    }
    auto unprojected = cam.unproject(projected);
    for (std::size_t i = 0; i < 3; ++i) {
      CAPTURE(i, query_point, projected, unprojected);
      REQUIRE(query_point(i) == Approx(unprojected(i)));
    }
  }
  SECTION("arbitrary") {
    auto res         = Vec2<std::size_t>{100, 100};
    auto height      = 5;
    auto near        = 0;
    auto far         = 2;
    auto eye         = vec3{2, 1, 3};
    auto lookat      = vec3{-1, 1, 3};
    auto up          = vec3{0, 1, 0};
    auto cam         = orthographic_camera<real_type>{eye,  lookat, up,     height,
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
    auto height      = 5;
    auto near        = 0;
    auto far         = 2;
    auto eye         = vec3{2, 1, 3};
    auto lookat      = vec3{-1, 1, 3};
    auto up          = vec3{0, 1, 0};
    auto cam         = orthographic_camera<real_type>{eye,  lookat, up,     height,
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
