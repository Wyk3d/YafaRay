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
};

#endif //_CL_KERNEL_H