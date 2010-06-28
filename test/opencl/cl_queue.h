#ifndef _CL_QUEUE_H
#define _CL_QUEUE_H

typedef CLObjectReleasableInfoBase <
	cl_command_queue,
	&clReleaseCommandQueue,
	cl_command_queue_info,
	&clGetCommandQueueInfo
> CLCommandQueueBase;

class CLCommandQueue
	: public CLCommandQueueBase
{
private:
	CLCommandQueue(cl_command_queue id) : CLCommandQueueBase(id) {

	}
public:
	friend class CLContext;

	void writeBuffer(const CLBuffer *buffer, size_t offset, size_t size, const void *mem, CLError *error) {
		CLErrGuard err(error);
		clEnqueueWriteBuffer(id, buffer->getId(), true, offset, size, mem, 0, NULL, NULL);
	}

	void readBuffer(const CLBuffer *buffer, size_t offset, size_t size, void *mem, CLError *error) {
		CLErrGuard err(error);
		clEnqueueReadBuffer(id, buffer->getId(), true, offset, size, mem, 0, NULL, NULL);
	}

	template<class Range>
	void runKernel(const CLKernel *kernel, Range &r, CLError *error) {
		CLErrGuard err(error);
		cl_event ev;
		err = clEnqueueNDRangeKernel(
			id, kernel->getId(), 
			r.getWorkDim(), 
			NULL, 
			r.getGlobalSizes(), r.getLocalSizes(), 
			NULL, NULL, &ev);
		clWaitForEvents(1, &ev);
	}
};

class Range1D
{
	private:
		size_t global, local;
	public:
		Range1D(size_t global_size, size_t local_size) : global(global_size), local(local_size) {

		}

		cl_uint getWorkDim() {
			return 1;
		}

		size_t *getGlobalSizes() {
			return &global;
		}

		size_t *getLocalSizes() {
			return &local;
		}
};

class Range2D
{
private:
	size_t global[2], local[2];
public:
	Range2D(size_t global_x, size_t global_y,  size_t local_x, size_t local_y) {
		global[0] = global_x;
		global[1] = global_y;
		local[0] = local_x;
		local[1] = local_y;
	}

	cl_uint getWorkDim() {
		return 2;
	}

	size_t *getGlobalSizes() {
		return global;
	}

	size_t *getLocalSizes() {
		return local;
	}
};

#endif //_CL_QUEUE_H