include_directories(${YAF_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})

set(YF_OPENCL_SOURCES 	cl_error.cc
						cl_platform.cc
						cl_util.cc)

add_library(opencl_wrapper STATIC ${YF_OPENCL_SOURCES})

if (UNIX)
	set (DLLOAD_LIB "dl" )
else (UNIX)
	set (DLLOAD_LIB "")
endif (UNIX)

target_link_libraries(opencl_wrapper ${OPENCL_LIBRARIES})

install (TARGETS opencl_wrapper ${YAF_TARGET_TYPE} DESTINATION ${YAF_LIB_DIR})
