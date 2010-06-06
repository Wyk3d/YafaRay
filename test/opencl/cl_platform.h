#ifndef _CL_PLATFORM_H
#define _CL_PLATFORM_H

class CLPlatform
{
private:
	cl_platform_id id;
	CLPlatform(cl_platform_id id) : id(id) {

	}

public:
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

	std::string getStringInfo(cl_platform_info info, CLError *error = NULL) {
		CLErrGuard err(error);

		// get the size of the info string
		size_t size;
		if(err = clGetPlatformInfo(id,
			info,
			0,
			NULL,
			&size)) 
			return "";

		// get the info string
		char *pbuf = new char[size];
		if(err = clGetPlatformInfo(id,
			info,
			size,
			pbuf,
			NULL)) 
		{
			delete[] pbuf;
			return "";
		}

		return pbuf;
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

		for(int i = 0; i < num_devices; ++i)
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

CLPlatform CLDevice::getPlatform(CLError *error)
{
	return getInfo<cl_platform_id>(CL_DEVICE_PLATFORM, error);
}

#endif //_CL_PLATFORM_H