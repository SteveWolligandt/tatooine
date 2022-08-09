if (TATOOINE_GL_AVAILABLE)
  add_executable(rendering.mesh.example triangular_mesh.cpp)
  target_link_libraries(rendering.mesh.example rendering)
endif()
