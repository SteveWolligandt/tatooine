add_executable(example.direct_volume direct_volume.cpp)
target_link_libraries(example.direct_volume tatooine)
add_custom_target(example.direct_volume.run
                  COMMAND ./example.direct_volume
                  DEPENDS example.direct_volume)
