#ifndef _CL_MEM_H
#define _CL_MEM_H

class CLMem :
	public
		CLObjectReleasableInfoBase <
			cl_mem,
			&clReleaseMemObject,
			cl_mem_info,
			&clGetMemObjectInfo
		>
{
protected:
	CLMem(cl_mem mem) : CLObjectReleasableInfoBase(mem) {

	}

	~CLMem() {

	}
};


#endif //_CL_MEM_H