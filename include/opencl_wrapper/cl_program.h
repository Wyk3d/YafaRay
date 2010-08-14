#ifndef _CL_PROGRAM_H
#define _CL_PROGRAM_H

typedef CLObjectReleasableInfoBase < 
	cl_program,
	&clReleaseProgram,
	cl_program_info,
	&clGetProgramInfo 
> CLProgramBase;

class CLProgram :
	public CLProgramBase
{
protected:
	CLProgram(cl_program id) : CLProgramBase(id) {

	}

	~CLProgram() {

	}
public:
	friend class CLContext;

	void build(CLError *error, const char *options = NULL) {
		CLErrGuard err(error);
		err = clBuildProgram(id, 0, NULL, options, NULL, NULL);
	}

	CLKernel* createKernel(const char *kernel_name, CLError *error) {
		CLErrGuard err(error);
		cl_kernel kernel = clCreateKernel(id, kernel_name, &err.getCode());
		return new CLKernel(kernel);
	}

	std::string getBuildLog(CLDevice device, CLError *error = NULL) {
		CLErrGuard err(error);

		// get the size of the info string
		size_t size;
		if(err = clGetProgramBuildInfo(id,
			device.getId(),
			CL_PROGRAM_BUILD_LOG,
			0,
			NULL,
			&size)) 
			return "";

		// get the info string
		char *pbuf = new char[size];
		if(err = clGetProgramBuildInfo(id,
			device.getId(),
			CL_PROGRAM_BUILD_LOG,
			size,
			pbuf,
			NULL)) 
		{
			delete[] pbuf;
			return "";
		}

		return pbuf;
	}
};

#endif //_CL_PROGRAM_H