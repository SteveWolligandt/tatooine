enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
  get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
  if(NOT "${old_flags}" STREQUAL "")
    string(REPLACE ";" "," CUDA_flags "${old_flags}")
    set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
        "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>")
  endif()
endfunction()

add_library(tatooine_gpu SHARED ${CMAKE_CURRENT_SOURCE_DIR}/cuda/add.cu)
set_target_properties(tatooine_gpu
  PROPERTIES
    CUDA_SEPERABLE_COMPILATION ON)
target_include_directories(tatooine_gpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(test)
