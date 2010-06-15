#ifndef _CL_KERNEL_H
#define _CL_KERNEL_H

class CLKernel :
	public 
		CLObjectReleasableInfoBase< 
			cl_kernel,
			&clReleaseKernel,
			cl_kernel_info,
			&clGetKernelInfo 
		>
{
	private:
		CLKernel(cl_kernel id) : CLObjectReleasableInfoBase(id) {

		}

		~CLKernel() {
			
		}
	public:
		friend class CLProgram;
};

#endif //_CL_KERNEL_H