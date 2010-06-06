#ifndef _CL_MEM_H
#define _CL_MEM_H

class CLMem
{
protected:
	cl_mem mem;
	CLMem(cl_mem mem) : mem(mem) {

	}
public:
	cl_mem getId() {
		return mem;
	}
};


#endif //_CL_MEM_H