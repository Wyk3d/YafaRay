#ifndef _CL_QUEUE_H
#define _CL_QUEUE_H

#include <yafraycore/ccthreads.h>

class CLCommandQueue;

typedef CLObjectReleasableInfoBase <
	cl_command_queue,
	cl_command_queue_info,
	CLCommandQueue
> CLCommandQueueBase;

class CLCommandQueue
	: public CLCommandQueueBase
{
	private:
		CLCommandQueue(cl_command_queue id) : CLCommandQueueBase(id) {

		}

		friend void CLCommandQueueBase::free(CLError *error);
		~CLCommandQueue()
		{
		}

		mutable yafthreads::mutex_t mutex;
	public:
		static cl_int InfoFunc(cl_command_queue id, cl_command_queue_info info, size_t param_size, void* param_value, size_t* param_size_ret) {
			return clGetCommandQueueInfo(id, info, param_size, param_value, param_size_ret);
		}

		static cl_int ReleaseFunc(cl_command_queue id) {
			return clReleaseCommandQueue(id);
		}

		friend class CLContext;

		cl_context getContext(CLError *error = NULL) {
			return getInfo<cl_context>(CL_QUEUE_CONTEXT, error);
		}

		void writeBuffer(const CLBuffer *buffer, size_t offset, size_t size, const void *mem, CLError *error = NULL) {
			yafthreads::guard_t guard(mutex);

			CLErrGuard err(error);
			err = clEnqueueWriteBuffer(id, buffer->getId(), true, offset, size, mem, 0, NULL, NULL);
		}

		void writeBuffer(const CLBuffer *buffer, void *mem, CLError *error = NULL) {
			CLErrGuard err(error);
			size_t size = buffer->getSize(&err);
			if(err) return;
			writeBuffer(buffer, 0, size, mem, &err);
		}

		template<class T>
		void writeBuffer(CLVectorBuffer<T> &vec, CLError *error = NULL)  {
			writeBuffer(vec.buffer, vec, error);
		}

		template<class T>
		void writeBuffer(CLBuffer *&buffer, std::vector<T> &vec, CLError *error = NULL)  {
			CLErrGuard err(error);

			cl_context context = getContext(&err);
			if(err) return;
			CLVectorBuffer<T>::initBuffer(context, vec, buffer, &err);
			if(err) return;

			writeBuffer(buffer, 0, vec.size() * sizeof(T), &vec[0], &err);
		}

		void readBuffer(const CLBuffer *buffer, size_t offset, size_t size, void *mem, CLError *error = NULL) {
			yafthreads::guard_t guard(mutex);

			CLErrGuard err(error);
			err = clEnqueueReadBuffer(id, buffer->getId(), true, offset, size, mem, 0, NULL, NULL);
		}

		void readBuffer(const CLBuffer *buffer, void *mem, CLError *error = NULL) {
			CLErrGuard err(error);
			size_t size = buffer->getSize(&err);
			if(err) return;
			readBuffer(buffer, 0, size, mem, &err);
		}

		template<class T>
		void readBuffer(CLVectorBuffer<T> &vec, CLError *error = NULL) {
			readBuffer(vec.buffer, vec, error);
		}

		template<class T>
		void readBuffer(CLBuffer *&buffer, std::vector<T> &vec, CLError *error = NULL) {
			CLErrGuard err(error);

			if(!buffer) return;
			size_t size = buffer->getSize(&err);
			if(err) return;

			if(size > vec.size() * sizeof(T)) {
				vec.resize(size / sizeof(T));
			}

			readBuffer(buffer, 0, size, &vec[0], &err);
		}

		template<class Range>
		void runKernel(const CLKernel *kernel, const Range &r, CLError *error = NULL) {
			yafthreads::guard_t guard(mutex);

			CLErrGuard err(error);
			cl_event ev;
			err = clEnqueueNDRangeKernel(
				id, kernel->getId(),
				r.getWorkDim(),
				NULL,
				r.getGlobalSizes(), r.getLocalSizes(),
				NULL, NULL, &ev);
			if(err) return;
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

		cl_uint getWorkDim() const {
			return 1;
		}

		const size_t *getGlobalSizes() const {
			return &global;
		}

		const size_t *getLocalSizes() const {
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

	cl_uint getWorkDim() const {
		return 2;
	}

	const size_t *getGlobalSizes() const {
		return global;
	}

	const size_t *getLocalSizes() const {
		return local;
	}
};

#endif //_CL_QUEUE_H

