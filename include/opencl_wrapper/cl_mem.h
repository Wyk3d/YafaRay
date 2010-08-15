#ifndef _CL_MEM_H
#define _CL_MEM_H

class CLMem;

typedef	CLObjectReleasableInfoBase <
	cl_mem,
	cl_mem_info,
	CLMem
> CLMemBase;

class CLMem :
	public CLMemBase
{
protected:
	CLMem(cl_mem mem) : CLMemBase(mem) {

	}

	~CLMem() {

	}
public:
	static cl_int InfoFunc(cl_mem id, cl_mem_info info, size_t param_size, void* param_value, size_t* param_size_ret) {
		return clGetMemObjectInfo(id, info, param_size, param_value, param_size_ret);
	}

	static cl_int ReleaseFunc(cl_mem id) {
		return clReleaseMemObject(id);
	}

	cl_mem_object_type getType(CLError *error = NULL) const {
		return getInfo<cl_mem_object_type>(CL_MEM_TYPE, error);
	}

	size_t getSize(CLError *error = NULL) const {
		return getInfo<size_t>(CL_MEM_SIZE, error);
	}
};


#endif //_CL_MEM_H