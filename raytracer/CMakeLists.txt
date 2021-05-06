add_executable(raytracer.render_sphere render_sphere.cpp)
target_link_libraries(raytracer.render_sphere tatooine)
add_custom_target (raytracer.render_sphere.run
  COMMAND ./raytracer.render_sphere sphere.png
  DEPENDS raytracer.render_sphere)

add_executable(raytracer.render_streamsurface render_streamsurface.cpp)
target_link_libraries(raytracer.render_streamsurface tatooine)
add_custom_target (raytracer.render_streamsurface.run
  COMMAND ./raytracer.render_streamsurface streamsurface.png
  DEPENDS raytracer.render_streamsurface)
