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

template<class T>
class CLVectorBuffer : public std::vector<T>
{
	public:
		CLBuffer *buffer;

		CLVectorBuffer() : std::vector<T>(), buffer(NULL) {}

		CLVectorBuffer(size_type _Count) : std::vector<T>(_Count), buffer(NULL) {}

		static void initBuffer(cl_context context, std::vector<T> &vec, CLBuffer *&buffer, CLError *error) {
			CLErrGuard err(error);

			if(buffer && vec.size() * sizeof(T) != buffer->getSize()) {
				buffer->free(&err);
				if(err) return;
				buffer = NULL;
			}

			if(!buffer && !vec.empty()) {
				cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, vec.size() * sizeof(T), NULL, &err.getCode());
				if(err) return;
				buffer = new CLBuffer(mem);
			}
		}

		~CLVectorBuffer() {
			if(buffer) {
				CLError err;
				buffer->free(&err);
				assert(!err);
			}
		}
};

#endif //_CL_BUFFER_H