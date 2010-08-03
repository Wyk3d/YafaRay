include_directories(${YAF_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})

add_executable(test_opencl opencl_test.cc)
target_link_libraries(test_opencl opencl_wrapper)
install (TARGETS test_opencl RUNTIME DESTINATION ${YAF_BIN_DIR})

add_executable(test_opencl_mandelbrot mandelbrot.cc)
target_link_libraries(test_opencl_mandelbrot yafaraycore opencl_wrapper)
install (TARGETS test_opencl_mandelbrot RUNTIME DESTINATION ${YAF_BIN_DIR})