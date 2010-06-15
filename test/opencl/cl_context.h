#ifndef _CL_CONTEXT_H
#define _CL_CONTEXT_H

class CLContext :
	public
		CLObjectReleasableInfoBase <  
			cl_context,
			&clReleaseContext,
			cl_context_info,
			&clGetContextInfo
		>
{
private:
	CLContext(cl_context id) : CLObjectReleasableInfoBase(id) {

	}

	~CLContext() {

	}

	friend class CLPlatform;
public:

	std::list<CLDevice> getDevices(CLError *error) {
		CLErrGuard err(error);
		std::list<CLDevice> dlist;

		// get the number of devices
		size_t size;
		if(err = clGetContextInfo(id, CL_CONTEXT_DEVICES, 0, 0, &size))
			return dlist;

		// get the devices 
		cl_device_id* devices = new cl_device_id[size/sizeof(cl_device_id)];
		if(err = clGetContextInfo(id, CL_CONTEXT_DEVICES, size, devices, 0)) {
			delete[] devices;
			return dlist;
		}

		// add the devices to the list
		for(int i = 0; i < size / sizeof(cl_device_id); ++i)
			dlist.push_back(devices[i]);

		return dlist;
	}

	CLCommandQueue *createCommandQueue(CLDevice device, CLError *error) {
		CLErrGuard err(error);

		cl_command_queue queue = clCreateCommandQueue(id, device.getId(), 0, &err.getCode());
		return err ? NULL : new CLCommandQueue(queue);
	}

	CLBuffer *createBuffer(cl_mem_flags flags, size_t size, void *host_ptr, CLError *error) {
		CLErrGuard err(error);
		cl_mem mem = clCreateBuffer(id, flags, size, host_ptr, &err.getCode());

		return err ? NULL : new CLBuffer(mem);
	}

	CLProgram *createProgram(const char *source, CLError *error) {
		CLErrGuard err(error);

		size_t size = strlen(source);
		cl_program program = clCreateProgramWithSource(id, 1, &source, &size, &err.getCode());

		return err ? NULL : new CLProgram(program);
	}
};

#endif //_CL_CONTEXT_H