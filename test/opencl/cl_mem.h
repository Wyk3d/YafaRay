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
};


#endif //_CL_MEM_H