#ifndef _CL_KERNEL_H
#define _CL_KERNEL_H

// AMD hack, declared here, initialized by context, used in the kernel setargs
class NullMemStore
{
	public:
		OPENCL_WRAPPER_EXPORT static cl_mem null_mem;
};

class CLKernel;

typedef CLObjectReleasableInfoBase< 
	cl_kernel,
	cl_kernel_info,
	CLKernel
> CLKernelBase;

class CLKernel :
	public CLKernelBase
{
	private:
		CLKernel(cl_kernel id) : CLKernelBase(id) {

		}

		friend void CLKernelBase::free(CLError *error);
		~CLKernel() {
			
		}
	public:
		static cl_int InfoFunc(cl_kernel id, cl_kernel_info info, size_t param_size, void* param_value, size_t* param_size_ret) {
			return clGetKernelInfo(id, info, param_size, param_value, param_size_ret);
		}

		static cl_int ReleaseFunc(cl_kernel id) {
			return clReleaseKernel(id);
		}
		friend class CLProgram;

		std::string getFuncName(CLError *error = NULL) {
			return getStringInfo(CL_KERNEL_FUNCTION_NAME, error);
		}

		cl_uint getNumArgs(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_KERNEL_NUM_ARGS, error);
		}

		template< typename T >
		class SetArgHelper {
			public:
				bool set(cl_kernel id, cl_uint idx, const T& arg, CLError *error) {
					CLErrGuard err(error);
					return err = clSetKernelArg(id, idx, sizeof(T), &arg);
				}
		};

		template<typename D, typename B>
		class IsDerivedFrom
		{
			static void Constraints(D* p)
			{
				B* pb = p; // throw an error only if not derived
				pb = p; // suppress warnings about unused variables
			}

		protected:
			IsDerivedFrom() {
				reference(&Constraints); // don't execute it, just reference it
			}

			void reference(void(*)(D*)) {}
		};

		template< class Mem >
		class SetArgHelper< Mem *>
			: IsDerivedFrom< Mem, CLMem >
		{
			public:
				bool set(cl_kernel id, cl_uint idx, Mem* const& mem, CLError *error) {
					CLErrGuard err(error);

					const void * mem_id_ptr;
					if(mem != NULL) {
						mem_id_ptr = &mem->getId();
					} else {
						// a crude hack to work around a bug in AMD's implementation
						// which gives an error if NULL is passed
						mem_id_ptr = &NullMemStore::null_mem;
						assert(mem_id_ptr != NULL);
					}

					return err = clSetKernelArg(id, idx, sizeof(cl_mem), mem_id_ptr);
				}
		};

		template< class T >
		class SetArgHelper< CLVectorBuffer<T> >
		{
			public:
				bool set(cl_kernel id, cl_uint idx, CLVectorBuffer<T> const& c_vec, CLError *error) {
					CLErrGuard err(error);

					CLVectorBuffer<T> &vec = *((CLVectorBuffer<T>*)&c_vec);
					
					cl_context context;
					clGetKernelInfo(id, CL_KERNEL_CONTEXT, sizeof(cl_context), &context, NULL);
					CLVectorBuffer<T>::initBuffer(context, 0, vec.size(), vec.buffer, &err);
					if(err) return false;

					const void *mem_id_ptr;
					if(vec.buffer != NULL) {
						mem_id_ptr = &vec.buffer->getId();
					} else {
						// a crude hack to work around a bug in AMD's implementation
						// which gives an error if NULL is passed
						mem_id_ptr = &NullMemStore::null_mem;
						assert(mem_id_ptr != NULL);
					}
					
					return err = clSetKernelArg(id, idx, sizeof(cl_mem), mem_id_ptr);
				}
		};

		template< class T >
		class SetArgHelper< CLVectorBufferRange<T> >
		{
			public:
				bool set(cl_kernel id, cl_uint idx, CLVectorBufferRange<T> const& c_range, CLError *error) {
					CLErrGuard err(error);

					CLVectorBufferRange<T> &range = *((CLVectorBufferRange<T>*)&c_range);
					
					cl_context context;
					clGetKernelInfo(id, CL_KERNEL_CONTEXT, sizeof(cl_context), &context, NULL);
					CLVectorBuffer<T>::initBuffer(context, range.buf_offset, range.length, range.vec.buffer, &err);
					if(err) return false;

					const void *mem_id_ptr;
					if(range.vec.buffer != NULL) {
						mem_id_ptr = &range.vec.buffer->getId();
					} else {
						// a crude hack to work around a bug in AMD's implementation
						// which gives an error if NULL is passed
						mem_id_ptr = &NullMemStore::null_mem;
						assert(mem_id_ptr != NULL);
					}
					
					return err = clSetKernelArg(id, idx, sizeof(cl_mem), mem_id_ptr);
				}
		};

		template<typename T>
		bool setArg(cl_uint idx, const T& arg, CLError *error = NULL) {
			SetArgHelper<T> helper;
			return helper.set(id, idx, arg, error);
		}

		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10, const A11& a11, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 12) return false;
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			if(setArg(4, a4, &err)) return false;
			if(setArg(5, a5, &err)) return false;
			if(setArg(6, a6, &err)) return false;
			if(setArg(7, a7, &err)) return false;
			if(setArg(8, a8, &err)) return false;
			if(setArg(9, a9, &err)) return false;
			if(setArg(10, a10, &err)) return false;
			if(setArg(11, a11, &err)) return false;
			return true;
		}

		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 11) return false;
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			if(setArg(4, a4, &err)) return false;
			if(setArg(5, a5, &err)) return false;
			if(setArg(6, a6, &err)) return false;
			if(setArg(7, a7, &err)) return false;
			if(setArg(8, a8, &err)) return false;
			if(setArg(9, a9, &err)) return false;
			if(setArg(10, a10, &err)) return false;
			return true;
		}

		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 10) return false;
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			if(setArg(4, a4, &err)) return false;
			if(setArg(5, a5, &err)) return false;
			if(setArg(6, a6, &err)) return false;
			if(setArg(7, a7, &err)) return false;
			if(setArg(8, a8, &err)) return false;
			if(setArg(9, a9, &err)) return false;
			return true;
		}

		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 9) return false;
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			if(setArg(4, a4, &err)) return false;
			if(setArg(5, a5, &err)) return false;
			if(setArg(6, a6, &err)) return false;
			if(setArg(7, a7, &err)) return false;
			if(setArg(8, a8, &err)) return false;
			return true;
		}

		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 8) return false;
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			if(setArg(4, a4, &err)) return false;
			if(setArg(5, a5, &err)) return false;
			if(setArg(6, a6, &err)) return false;
			if(setArg(7, a7, &err)) return false;
			return true;
		}

		template<typename A0, typename A1, typename A2, typename A3, typename A4>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, const A4& a4, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 5) return false;
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			if(setArg(4, a4, &err)) return false;
			return true;
		}


		template<typename A0, typename A1, typename A2, typename A3>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, const A3& a3, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 4) return false;	
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			if(setArg(3, a3, &err)) return false;
			return true;
		}

		template<typename A0, typename A1, typename A2>
		bool setArgs(const A0& a0, const A1& a1, const A2& a2, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 3) return false;	
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			if(setArg(2, a2, &err)) return false;
			return true;
		}

		template<typename A0, typename A1>
		bool setArgs(const A0& a0, const A1& a1, CLError *error = NULL) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 2) return false;	
			if(setArg(0, a0, &err)) return false;
			if(setArg(1, a1, &err)) return false;
			return true;
		}

		template<typename A0>
		bool setArgs(const A0& a0, CLError *error) {
			CLErrGuard err(error);
			if(getNumArgs(&err) != 1) return false;
			if(setArg(0, a0, &err)) return false;
			return true;
		}

		class ArgStream
		{
			private:
				CLKernel &kernel;
				CLError err;
				int idx;
			public:
				ArgStream(CLKernel &kernel, int idx) : kernel(kernel), idx(idx) {

				}

				ArgStream(const ArgStream &as) : kernel(as.kernel), err(as.err), idx(as.idx + 1) {
					
				}
			
				friend class CLKernel;

				template<class A>
				ArgStream operator << (const A &arg) {
					if(err) return;
					kernel.setArg(idx++, arg, &err);
				}

				void getErr(CLError *error) {
					if(error != NULL) *error = err;
				}
		};
};



#endif //_CL_KERNEL_H