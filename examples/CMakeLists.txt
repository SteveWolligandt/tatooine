add_executable(example.direct_volume direct_volume.cpp)
target_link_libraries(example.direct_volume tatooine)
add_custom_target(example.direct_volume.run
                  COMMAND ./example.direct_volume
                  DEPENDS example.direct_volume)

add_executable(example.direct_iso direct_iso.cpp)
target_link_libraries(example.direct_iso tatooine)
add_custom_target(example.direct_iso.run
                  COMMAND ./example.direct_iso
                  DEPENDS example.direct_iso)
