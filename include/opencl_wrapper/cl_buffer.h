#ifndef _CL_BUFFER_H
#define _CL_BUFFER_H

#include <vector>

class CLBuffer : public CLMem
{
private:
	CLBuffer(cl_mem mem) : CLMem(mem) {

	}

	~CLBuffer() {

	}
public:
	friend class CLContext;
	template<class T>
	friend class CLVectorBuffer;
};

class CLKernel;
class CLContext;

template<class T>
class CLVectorBufferRange;

template<class T>
class CLVectorBuffer : public std::vector<T>
{
	public:
		CLBuffer *buffer;

		CLVectorBuffer() : std::vector<T>(), buffer(NULL) {}

		CLVectorBuffer(typename std::vector<T>::size_type _Count) : std::vector<T>(_Count), buffer(NULL) {}

		static void initBuffer(cl_context context, size_t offset, size_t length, CLBuffer *&buffer, CLError *error)
		{
			CLErrGuard err(error);

			if(buffer) {
				size_t buf_size = buffer->getSize(&err);
				if(err) return;

				if((offset + length) * sizeof(T) > buf_size) {
					buffer->free(&err);
					if(err) return;
					buffer = NULL;
				}
			}

			if(!buffer && length > 0) {
				cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (offset+length) * sizeof(T), NULL, &err.getCode());
				if(err) return;
				buffer = new CLBuffer(mem);
			}
		}

		void init(CLContext *context, CLError *error); // in cl_context.h

		~CLVectorBuffer() {
			if(buffer) {
				CLError err;
				buffer->free(&err);
				assert(!err);
			}
		}

		CLVectorBufferRange<T> range(size_t offset, size_t length);
		CLVectorBufferRange<T> range(size_t vec_offset, size_t buf_offset, size_t length);

		operator CLVectorBufferRange<T>();
};

template<class T>
class CLVectorBufferRange
{
	public:
		CLVectorBuffer<T> &vec;
		size_t vec_offset;
		size_t buf_offset;
		size_t length;

		CLVectorBufferRange(CLVectorBuffer<T> &vec, size_t vec_offset, size_t buf_offset, size_t length) 
			: vec(vec), vec_offset(vec_offset), buf_offset(buf_offset), length(length) 
		{

		}

		CLVectorBufferRange(const CLVectorBufferRange<T> &range) 
			: vec(range.vec), vec_offset(range.vec_offset), buf_offset(range.buf_offset), length(range.length) {
		}

		operator typename std::vector<T>::iterator()
		{
			return vec.begin() + vec_offset;
		}
};

template<class T>
CLVectorBufferRange<T> CLVectorBuffer<T>::range(size_t offset, size_t length) {
	return CLVectorBufferRange<T>(*this, offset, offset, length);
}

template<class T>
CLVectorBufferRange<T> CLVectorBuffer<T>::range(size_t vec_offset, size_t buf_offset, size_t length) {
	return CLVectorBufferRange<T>(*this, vec_offset, buf_offset, length);
}

template<class T>
CLVectorBuffer<T>::operator CLVectorBufferRange<T>() {
	return CLVectorBufferRange<T>(*this, 0, 0, this->size());
}

#endif //_CL_BUFFER_H
