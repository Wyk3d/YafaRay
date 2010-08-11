#ifndef _CL_MEM_H
#define _CL_MEM_H

typedef	CLObjectReleasableInfoBase <
	cl_mem,
	&clReleaseMemObject,
	cl_mem_info,
	&clGetMemObjectInfo
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
	cl_mem_object_type getType(CLError *error = NULL) const {
		return getInfo<cl_mem_object_type>(CL_MEM_TYPE, error);
	}

	size_t getSize(CLError *error = NULL) const {
		return getInfo<size_t>(CL_MEM_SIZE, error);
	}
};


#endif //_CL_MEM_H