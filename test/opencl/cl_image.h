#ifndef _CL_IMAGE_H
#define _CL_IMAGE_H

class CLImage :
	public CLMem
{
	private:
		CLInfoTraits< cl_mem, cl_image_info, &clGetImageInfo > imageInfo;

		CLImage(cl_mem mem) : CLMem(mem) {

		}

		~CLImage() {

		}
};

#endif //_CL_IMAGE_H