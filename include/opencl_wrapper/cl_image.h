#ifndef _CL_IMAGE_H
#define _CL_IMAGE_H

class CLImage :
	public CLMem
{
	private:
		CLImage(cl_mem mem) : CLMem(mem) {

		}

		~CLImage() {

		}
};

#endif //_CL_IMAGE_H