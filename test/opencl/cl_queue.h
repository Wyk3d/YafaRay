#ifndef _CL_QUEUE_H
#define _CL_QUEUE_H

class CLCommandQueue
{
private:
	cl_command_queue id;
	CLCommandQueue(cl_command_queue id) : id(id) {

	}
public:
	friend class CLContext;
	cl_command_queue getId() {
		return id;
	}
};

#endif //_CL_QUEUE_H