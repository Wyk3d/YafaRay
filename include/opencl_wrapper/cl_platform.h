#ifndef _CL_PLATFORM_H
#define _CL_PLATFORM_H

class CLPlatform;

typedef CLObjectInfoBase <
	cl_platform_id,
	cl_platform_info,
	CLPlatform
> CLPlatformBase;

class CLPlatform 
	: public CLPlatformBase
{
private:
	CLPlatform(cl_platform_id id) : CLPlatformBase(id) {

	}

public:
	CLPlatform() : CLPlatformBase(0) {
		
	}

	static cl_int InfoFunc(cl_platform_id id, cl_platform_info info, size_t param_size, void* param_value, size_t* param_size_ret) {
		return clGetPlatformInfo(id, info, param_size, param_value, param_size_ret);
	}

	std::string getProfile(CLError *error = NULL) {
		return getStringInfo(CL_PLATFORM_PROFILE, error);
	}

	std::string getVersion(CLError *error = NULL) {
		return getStringInfo(CL_PLATFORM_VERSION, error);
	}

	std::string getVendor(CLError *error = NULL) {
		return getStringInfo(CL_PLATFORM_VENDOR, error);
	}

	std::string getName(CLError *error = NULL) {
		return getStringInfo(CL_PLATFORM_NAME, error);
	}

	std::string getExtensions(CLError *error = NULL) {
		return getStringInfo(CL_PLATFORM_EXTENSIONS, error);
	}

	std::list<CLDevice> getDevices(CLError *error) {
		CLErrGuard err(error);
		std::list<CLDevice> dlist;

		cl_uint num_devices;
		if(err = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices))
			return dlist;

		cl_device_id *devices = new cl_device_id[num_devices];
		if(err = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL))
			return dlist;

		for(size_t i = 0; i < num_devices; ++i)
			dlist.push_back(devices[i]);

		return dlist;
	}

	CLContext *createContext(cl_device_type device_type, CLError *error = NULL) {
		CLErrGuard err(error);

		cl_context_properties cps[3] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)id,
			0
		};

		cl_context context = clCreateContextFromType(cps,
			device_type,
			NULL,
			NULL,
			&err.getCode());

		return err ? NULL : new CLContext(context);
	}

	CLContext *createContext(CLDevice device, CLError *error = NULL) {
		CLErrGuard err(error);

		cl_context_properties cps[3] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)id,
			0
		};
		cl_device_id did = device.getId();

		cl_context context = clCreateContext(cps,
			1,
			&did,
			NULL,
			NULL,
			&err.getCode());

		return err ? NULL : new CLContext(context);
	}

	cl_platform_id getId() {
		return id;
	}

public:
	friend class CLMain;
	friend class CLDevice;
};

#endif //_CL_PLATFORM_H
