#ifndef _CL_BUFFER_H
#define _CL_BUFFER_H

class CLBuffer : public CLMem
{
private:
	CLBuffer(cl_mem mem) : CLMem(mem) {

	}
public:
	friend class CLContext;
};

#endif //_CL_BUFFER_H