#ifndef _CL_KERNEL_H
#define _CL_KERNEL_H

typedef CLObjectReleasableInfoBase< 
	cl_kernel,
	&clReleaseKernel,
	cl_kernel_info,
	&clGetKernelInfo 
> CLKernelBase;

class CLKernel :
	public CLKernelBase
{
	private:
		CLKernel(cl_kernel id) : CLKernelBase(id) {

		}

		~CLKernel() {
			
		}
	public:
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
					return err = clSetKernelArg(id, idx, sizeof(cl_mem), &mem->getId());
				}
		};

		template< class T >
		class SetArgHelper< CLVectorBuffer<T> >
		{
			public:
				bool set(cl_kernel id, cl_uint idx, CLVectorBuffer<T> const& vec, CLError *error) {
					CLErrGuard err(error);
					
					cl_context context;
					clGetKernelInfo(id, CL_KERNEL_CONTEXT, sizeof(cl_context), &context, NULL);
					((CLVectorBuffer<T>*)&vec)->initBuffer(context, &err);
					if(err) return false;

					return err = clSetKernelArg(id, idx, sizeof(cl_mem), &vec.getBuffer()->getId());
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
};



#endif //_CL_KERNEL_H