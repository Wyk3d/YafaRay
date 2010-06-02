include_directories(${YAF_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})

add_executable(opencl_test opencl_test.cc)
target_link_libraries(opencl_test ${OPENCL_LIBRARIES})