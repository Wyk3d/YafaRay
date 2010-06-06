#ifndef _CL_PROGRAM_H
#define _CL_PROGRAM_H

class CLProgram :
	public CLObjectBase< cl_program, cl_program_info, &clGetProgramInfo >
{
protected:
	CLProgram(cl_program id) : CLObjectBase(id) {

	}
public:
	friend class CLContext;

	void build(CLError *error) {
		CLErrGuard err(error);
		char *options = NULL;
		err = clBuildProgram(id, 0, NULL, options, NULL, NULL);
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