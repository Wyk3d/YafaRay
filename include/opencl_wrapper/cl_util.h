#ifndef _CL_UTIL_H_
#define _CL_UTIL_H_

#include "cl_wrapper.h"
#include <time.h>
#include <stdlib.h>

inline void
checkErr(CLCombinedError err, const char * message = "", bool fatal = true)
{
	if (!err.isCombinedSuccess()) {
		std::cerr << "ERROR: " << message;
		if(err.hasFailed())
			std::cerr << " " << err.getString() << " (" << err.getCode() << ")";
		std::cerr << std::endl;
		if(fatal) exit(EXIT_FAILURE);
	}
}

OPENCL_WRAPPER_EXPORT CLProgram * buildCLProgram(const char *kernel_source, CLContext *context, CLDevice device, const char *options = NULL);

class OPENCL_WRAPPER_EXPORT CLApplication
{
	public:
		CLApplication();
		~CLApplication();
	protected:
		CLPlatform platform;
		CLDevice device;
		CLContext *context;
		CLCommandQueue *queue;

		std::string cl_build_options;
};

#endif //_CL_UTIL_H_