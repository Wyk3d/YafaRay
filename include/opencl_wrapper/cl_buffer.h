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

template<class T>
class CLVectorBuffer : public std::vector<T>
{
	protected:
		CLBuffer *buffer;
		friend class CLCommandQueue;
	public:
		CLVectorBuffer() : std::vector<T>(), buffer(NULL) {}

		CLVectorBuffer(size_type _Count) : std::vector<T>(_Count), buffer(NULL) {}

		CLBuffer *getBuffer() const { return buffer; }

		void initBuffer(cl_context context, CLError *error) {
			CLErrGuard err(error);

			if(buffer && size() * sizeof(T) != buffer->getSize()) {
				buffer->free(&err);
				if(err) return;
				buffer = NULL;
			}

			if(!buffer) {
				cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, size() * sizeof(T), NULL, &err.getCode());
				if(err) return;
				buffer = new CLBuffer(mem);
			}
		}
};

#endif //_CL_BUFFER_H