include_directories(${YAF_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})

set(YF_OPENCL_SOURCES 	cl_error.cc
						cl_platform.cc
						cl_util.cc
						cl_kernel.cc)

add_definitions(-DBUILDING_OPENCL_WRAPPER)

add_library(opencl_wrapper SHARED ${YF_OPENCL_SOURCES})

target_link_libraries(opencl_wrapper yafaraycore ${OPENCL_LIBRARIES})

install (TARGETS opencl_wrapper ${YAF_TARGET_TYPE} DESTINATION ${YAF_LIB_DIR})
