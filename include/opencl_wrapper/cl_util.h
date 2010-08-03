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

CLProgram * buildCLProgram(const char *kernel_source, CLContext *context, CLDevice device) {
	CLError err;
	CLProgram *program = context->createProgram(kernel_source, &err);
	checkErr(err || program == NULL, "failed to create program");

	std::cout << "building program ...";
	CLError build_error;

	clock_t start = clock();
	program->build(&build_error);
	clock_t end = clock();
	std::cout << "[" << (float)(end-start)/(float)CLOCKS_PER_SEC << "s]: ";

	std::string log = program->getBuildLog(device, &err);

	if(build_error) std::cout << "FAILED: " << build_error.getString() << " ("
		<< build_error.getCode() << ")" << std::endl;
	else std::cout << "SUCCESS." << std::endl;

	checkErr(err, "failed to get build log");
	if(log != "") std::cout << log << std::endl;
	if(build_error)
		exit(EXIT_FAILURE);

	return program;
}

#endif //_CL_UTIL_H_