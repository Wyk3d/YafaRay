#include <yafray_config.h>
#include <opencl_wrapper/cl_wrapper.h>

CLPlatform CLDevice::getPlatform(CLError *error)
{
	return getInfo<cl_platform_id>(CL_DEVICE_PLATFORM, error);
}
