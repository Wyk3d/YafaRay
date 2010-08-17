#ifndef _CL_PROGRAM_H
#define _CL_PROGRAM_H

class CLProgram;

typedef CLObjectReleasableInfoBase < 
	cl_program,
	cl_program_info,
	CLProgram 
> CLProgramBase;

class CLProgram :
	public CLProgramBase
{
protected:
	CLProgram(cl_program id) : CLProgramBase(id) {

	}

	friend void CLProgramBase::free(CLError *error);
	~CLProgram() {

	}
public:
	static cl_int InfoFunc(cl_program id, cl_program_info info, size_t param_size, void* param_value, size_t* param_size_ret) {
		return clGetProgramInfo(id, info, param_size, param_value, param_size_ret);
	}

	static cl_int ReleaseFunc(cl_program id) {
		return clReleaseProgram(id);
	}

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