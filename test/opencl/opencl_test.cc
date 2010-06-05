#include <cassert>
#include <cmath>
#include <cstring>

#include <CL/cl.h>

#include <string>
#include <sstream>
#include <list>

#define CL_SRC(a) #a

const char *kernel_source = CL_SRC(
  __kernel void vectorAdd(int n,
                          __global const float* a,
                          __global const float* b,
                          __global float* c)
  {
    int gid = get_global_id(0);
    if (gid >= n)
      return;
    c[gid] = a[gid] + b[gid];
  }
);

/*const char *kernel_source = CL_SRC(
  #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
  __constant char hw[] = "Hello World\n";
  __kernel void hello(__global char * out)
  {
    size_t tid = get_global_id(0);
    out[tid] = hw[tid];
  }
);*/



#define N 1024
#define LOCAL_WORK_SIZE 256

#include <iostream>

class CLCombinedError;

class CLError
{
	protected:
		cl_int code;
	public:
		CLError(cl_int code) : code(code) {}
		CLError() : code(CL_SUCCESS) {}
		bool hasSucceeded() const { return code == CL_SUCCESS; }
		bool hasFailed() const { return code != CL_SUCCESS; }

		bool operator= (cl_int error) {
			code = error;
			return hasFailed();
		}

		CLCombinedError operator || (bool error_condition);

		operator bool () const {
			return hasFailed();
		}

		cl_int &getCode() {
			return code;
		}

		const char * getString() const {
			switch(code)
			{
			case CL_DEVICE_NOT_FOUND:
				return "CL_DEVICE_NOT_FOUND";
			case CL_DEVICE_NOT_AVAILABLE:
				return "CL_DEVICE_NOT_AVAILABLE";               
			case CL_COMPILER_NOT_AVAILABLE:
				return "CL_COMPILER_NOT_AVAILABLE";           
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				return "CL_MEM_OBJECT_ALLOCATION_FAILURE";      
			case CL_OUT_OF_RESOURCES:
				return "CL_OUT_OF_RESOURCES";                    
			case CL_OUT_OF_HOST_MEMORY:
				return "CL_OUT_OF_HOST_MEMORY";                 
			case CL_PROFILING_INFO_NOT_AVAILABLE:
				return "CL_PROFILING_INFO_NOT_AVAILABLE";        
			case CL_MEM_COPY_OVERLAP:
				return "CL_MEM_COPY_OVERLAP";                    
			case CL_IMAGE_FORMAT_MISMATCH:
				return "CL_IMAGE_FORMAT_MISMATCH";               
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:
				return "CL_IMAGE_FORMAT_NOT_SUPPORTED";         
			case CL_BUILD_PROGRAM_FAILURE:
				return "CL_BUILD_PROGRAM_FAILURE";              
			case CL_MAP_FAILURE:
				return "CL_MAP_FAILURE";                         
			case CL_INVALID_VALUE:
				return "CL_INVALID_VALUE";                      
			case CL_INVALID_DEVICE_TYPE:
				return "CL_INVALID_DEVICE_TYPE";               
			case CL_INVALID_PLATFORM:
				return "CL_INVALID_PLATFORM";                   
			case CL_INVALID_DEVICE:
				return "CL_INVALID_DEVICE";                    
			case CL_INVALID_CONTEXT:
				return "CL_INVALID_CONTEXT";                    
			case CL_INVALID_QUEUE_PROPERTIES:
				return "CL_INVALID_QUEUE_PROPERTIES";           
			case CL_INVALID_COMMAND_QUEUE:
				return "CL_INVALID_COMMAND_QUEUE";              
			case CL_INVALID_HOST_PTR:
				return "CL_INVALID_HOST_PTR";                   
			case CL_INVALID_MEM_OBJECT:
				return "CL_INVALID_MEM_OBJECT";                  
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
				return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";    
			case CL_INVALID_IMAGE_SIZE:
				return "CL_INVALID_IMAGE_SIZE";                 
			case CL_INVALID_SAMPLER:
				return "CL_INVALID_SAMPLER";                    
			case CL_INVALID_BINARY:
				return "CL_INVALID_BINARY";                     
			case CL_INVALID_BUILD_OPTIONS:
				return "CL_INVALID_BUILD_OPTIONS";              
			case CL_INVALID_PROGRAM:
				return "CL_INVALID_PROGRAM";                    
			case CL_INVALID_PROGRAM_EXECUTABLE:
				return "CL_INVALID_PROGRAM_EXECUTABLE";          
			case CL_INVALID_KERNEL_NAME:
				return "CL_INVALID_KERNEL_NAME";                
			case CL_INVALID_KERNEL_DEFINITION:
				return "CL_INVALID_KERNEL_DEFINITION";          
			case CL_INVALID_KERNEL:
				return "CL_INVALID_KERNEL";                     
			case CL_INVALID_ARG_INDEX:
				return "CL_INVALID_ARG_INDEX";                   
			case CL_INVALID_ARG_VALUE:
				return "CL_INVALID_ARG_VALUE";                   
			case CL_INVALID_ARG_SIZE:
				return "CL_INVALID_ARG_SIZE";                    
			case CL_INVALID_KERNEL_ARGS:
				return "CL_INVALID_KERNEL_ARGS";                
			case CL_INVALID_WORK_DIMENSION:
				return "CL_INVALID_WORK_DIMENSION";              
			case CL_INVALID_WORK_GROUP_SIZE:
				return "CL_INVALID_WORK_GROUP_SIZE";             
			case CL_INVALID_WORK_ITEM_SIZE:
				return "CL_INVALID_WORK_ITEM_SIZE";             
			case CL_INVALID_GLOBAL_OFFSET:
				return "CL_INVALID_GLOBAL_OFFSET";              
			case CL_INVALID_EVENT_WAIT_LIST:
				return "CL_INVALID_EVENT_WAIT_LIST";             
			case CL_INVALID_EVENT:
				return "CL_INVALID_EVENT";                      
			case CL_INVALID_OPERATION:
				return "CL_INVALID_OPERATION";                 
			case CL_INVALID_GL_OBJECT:
				return "CL_INVALID_GL_OBJECT";                  
			case CL_INVALID_BUFFER_SIZE:
				return "CL_INVALID_BUFFER_SIZE";                 
			case CL_INVALID_MIP_LEVEL:
				return "CL_INVALID_MIP_LEVEL";                   
			case CL_INVALID_GLOBAL_WORK_SIZE:
				return "CL_INVALID_GLOBAL_WORK_SIZE";            
			default:
				return "unknown error code";
			}

			return "unknown error code";
		}
};

class CLCombinedError : public CLError
{
private:
	bool success;
public:
	CLCombinedError(bool success, CLError err) : CLError(err), success(success) {
		
	}

	CLCombinedError(CLError err) : CLError(err) {

	}

	bool isCombinedSuccess() const {
		return success && CLError::hasSucceeded();
	}
};

CLCombinedError CLError::operator||(bool error_condition)
{
	if(error_condition) {
		return CLCombinedError(false, *this);
	} else
		return *this;
}

class CLErrGuard : public CLError
{
protected:
	CLError *ret_err;
public:
	CLErrGuard(CLError *err) {
		ret_err = err;
	}

	~CLErrGuard() {
		if(ret_err)
			*ret_err = code;
	}

	bool operator =(cl_int err) {
		return CLError::operator =(err);
	}
};

class CLPlatform;

class CLDevice
{
	private:
		cl_device_id id;
		CLDevice(cl_device_id id) : id(id) {

		}
	public:
		friend class CLContext;
		friend class CLPlatform;

		cl_device_id getId() {
			return id;
		}

		cl_device_type getType(CLError *error = NULL) {
			return getInfo<cl_device_type>(CL_DEVICE_TYPE, error);
		}

		cl_uint getVendorId(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_VENDOR_ID, error);
		}

		cl_uint getMaxComputeUnits(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS, error);
		}

		cl_uint getMaxWorkItemDimensions(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, error);
		}

		std::list<size_t> getMaxWorkItemSizes(CLError *error = NULL) {
			return getListInfo<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES, error);
		}

		size_t getMaxWorkGroupSize(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE, error);
		}

		cl_uint getPreferredVectorWidthChar(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, error);
		}

		cl_uint getPreferredVectorWidthShort(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, error);
		}

		cl_uint getPreferredVectorWidthInt(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, error);
		}

		cl_uint getPreferredVectorWidthLong(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, error);
		}

		cl_uint getPreferredVectorWidthFloat(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, error);
		}

		cl_uint getPreferredVectorWidthDouble(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, error);
		}

		cl_uint getMaxClockFrequency(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY, error);
		}

		cl_uint getAddressBits(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_ADDRESS_BITS, error);
		}

		cl_ulong getMaxMemAllocSize(CLError *error = NULL) {
			return getInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE, error);
		}

		cl_bool hasImageSupport(CLError *error = NULL) {
			return getInfo<cl_bool>(CL_DEVICE_IMAGE_SUPPORT, error);
		}

		cl_uint getMaxReadImageArgs(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_READ_IMAGE_ARGS, error);
		}


		cl_uint getMaxWriteImageArgs(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, error);
		}


		size_t getImage2DMaxWidth(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH, error);
		}

		size_t getImage2DMaxHeight(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT, error);
		}

		size_t getImage3DMaxWidth(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH, error);
		}

		size_t getImage3DMaxHeight(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT, error);
		}

		size_t getImage3DMaxDepth(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH, error);
		}

		cl_uint getMaxSamplers(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_SAMPLERS, error);
		}

		size_t getMaxParameterSize(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_MAX_PARAMETER_SIZE, error);
		}

		cl_uint getMemBaseAddrAlign(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN, error);
		}

		cl_uint getMinDataTypeAlignSize(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, error);
		}

		cl_device_fp_config getSingleFPConfig(CLError *error = NULL) {
			return getInfo<cl_device_fp_config>(CL_DEVICE_SINGLE_FP_CONFIG, error);
		}

		cl_device_mem_cache_type getGlobalMemCacheType(CLError *error = NULL) {
			return getInfo<cl_device_mem_cache_type>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, error);
		}

		cl_uint getGlobalMemCacheLineSize(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, error);
		}

		cl_ulong getGlobalMemCacheSize(CLError *error = NULL) {
			return getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, error);
		}

		cl_ulong getGlobalMemSize(CLError *error = NULL) {
			return getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE, error);
		}

		cl_ulong getMaxConstantBufferSize(CLError *error = NULL) {
			return getInfo<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, error);
		}

		cl_uint getMaxConstantArgs(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_MAX_CONSTANT_ARGS, error);
		}

		cl_device_local_mem_type getLocalMemType(CLError *error = NULL) {
			return getInfo<cl_device_local_mem_type>(CL_DEVICE_LOCAL_MEM_TYPE, error);
		}

		cl_ulong getLocalMemSize(CLError *error = NULL) {
			return getInfo<cl_uint>(CL_DEVICE_LOCAL_MEM_SIZE, error);
		}

		cl_bool hasErrorCorrectionSupport(CLError *error = NULL) {
			return getInfo<cl_bool>(CL_DEVICE_ERROR_CORRECTION_SUPPORT, error);
		}

		size_t getProfilingTimerResolution(CLError *error = NULL) {
			return getInfo<size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION, error);
		}

		cl_bool isLittleEndian(CLError *error = NULL) {
			return getInfo<cl_bool>(CL_DEVICE_ENDIAN_LITTLE, error);
		}

		cl_uint hasCompiler(CLError *error = NULL) {
			return getInfo<cl_bool>(CL_DEVICE_COMPILER_AVAILABLE, error);
		}

		cl_device_exec_capabilities getExecutionCapabilities (CLError *error = NULL) {
			return getInfo<cl_device_exec_capabilities>(CL_DEVICE_EXECUTION_CAPABILITIES, error);
		}

		cl_command_queue_properties getQueueProperties(CLError *error = NULL) {
			return getInfo<cl_command_queue_properties>(CL_DEVICE_QUEUE_PROPERTIES, error);
		}

		CLPlatform getPlatform(CLError *error = NULL);

		std::string getName(CLError *error = NULL) {
			return getStringInfo(CL_DEVICE_NAME, error);
		}

		std::string getVendor(CLError *error = NULL) {
			return getStringInfo(CL_DEVICE_VENDOR, error);
		}

		std::string getDriverVersion(CLError *error = NULL) {
			return getStringInfo(CL_DRIVER_VERSION, error);
		}

		std::string getProfile(CLError *error = NULL) {
			return getStringInfo(CL_DEVICE_PROFILE, error);
		}

		std::string getVersion(CLError *error = NULL) {
			return getStringInfo(CL_DEVICE_VERSION, error);
		}

		std::string getExtensions(CLError *error = NULL) {
			return getStringInfo(CL_DEVICE_EXTENSIONS, error);
		}

		template<typename T>
		T getInfo(cl_device_info info, CLError *error = NULL) {
			CLErrGuard err(error);
			T ret = 0;

			// get the size of the info string
			size_t size;
			if((err = clGetDeviceInfo(id,
				info,
				0,
				NULL,
				&size)) || sizeof(T) != size)
				return ret;

			// get the info string

			if(err = clGetDeviceInfo(id,
				info,
				size,
				&ret,
				NULL)) 
				return ret;

			return ret;
		}

		std::string getStringInfo(cl_device_info info, CLError *error) {
			CLErrGuard err(error);

			// get the size of the info string
			size_t size;
			if(err = clGetDeviceInfo(id,
				info,
				0,
				NULL,
				&size)) 
				return "";

			// get the info string
			char *pbuf = new char[size];
			if(err = clGetDeviceInfo(id,
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

		template<typename T>
		std::list<T> getListInfo(cl_device_info info, CLError *error) {
			CLErrGuard err(error);
			std::list<T> ilist;

			// get the size of the info list
			size_t size;
			if(err = clGetDeviceInfo(id,
				info,
				0,
				NULL,
				&size)) 
				return ilist;

			// get the info list
			T *info_buf = new T[size/sizeof(T)];
			if(err = clGetDeviceInfo(id,
				info,
				size,
				info_buf,
				NULL)) 
			{
				delete[] info_buf;
				return ilist;
			}

			for(int i = 0; i < size / sizeof(T); ++i)
				ilist.push_back(info_buf[i]);

			return ilist;
		}
};

class CLContext
{
	private:
		cl_context id;
		CLContext(cl_context id) : id(id) {

		}

		friend class CLPlatform;
	public:
		cl_context getId() {
			return id;
		}

		std::list<CLDevice> getDevices(CLError *error) {
			CLErrGuard err(error);
			std::list<CLDevice> dlist;

			// get the number of devices
			size_t size;
			if(err = clGetContextInfo(id, CL_CONTEXT_DEVICES, 0, 0, &size))
				return dlist;
			
			// get the devices 
			cl_device_id* devices = new cl_device_id[size/sizeof(cl_device_id)];
			if(err = clGetContextInfo(id, CL_CONTEXT_DEVICES, size, devices, 0)) {
				delete[] devices;
				return dlist;
			}

			// add the devices to the list
			for(int i = 0; i < size / sizeof(cl_device_id); ++i)
				dlist.push_back(devices[i]);

			return dlist;
		}
};

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

class CLMain
{
public:

	std::list<CLPlatform> getPlatforms(CLError *error = NULL)
	{
		CLErrGuard err(error);
		std::list<CLPlatform> plist;

		// find the number of platforms
		cl_uint numPlatforms;
		if(err = clGetPlatformIDs(0, NULL, &numPlatforms))
			return plist;

		// read all the platform ids into an array
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		if(err = clGetPlatformIDs(numPlatforms, platforms, NULL)) {
			delete[] platforms;
			return plist;
		}

		// add the ids from the array into the list
		for(cl_uint i = 0; i < numPlatforms; i++)
			plist.push_back(platforms[i]);

		delete[] platforms;
		return plist;
	}
};


inline void
checkErr(CLCombinedError err, const char * message = "")
{
	if (!err.isCombinedSuccess()) {
		std::cerr << "ERROR: " << message;
		if(err.hasFailed())
			std::cerr << " " << err.getString() << " (" << err.getCode() << ")";
		std::cerr << std::endl;
		exit(EXIT_FAILURE);
	}
}

void printPlatformInfo(CLPlatform platform) {
	std::cout << "Platform " << platform.getName() << " (" << platform.getId() << ")" << std::endl;
	std::cout << "vendor: " << platform.getVendor() << std::endl;
	std::cout << "version: " << platform.getVersion() << std::endl;
	std::cout << "profile: " << platform.getProfile() << std::endl;
	std::cout << "extensions: " << platform.getExtensions() << std::endl;
	std::cout << std::endl;
}

void printDeviceInfo(CLDevice device) {
	std::cout << "Device " << device.getName() << " (" << device.getId() << ")" << std::endl;
	std::cout << "type: " << device.getType() << std::endl;
	std::cout << "vendor: " << device.getVendor() << " (" << device.getVendorId() << ")" << std::endl;
	std::cout << "version: " << device.getVersion() << std::endl;
	std::cout << "driver version: " << device.getDriverVersion() << std::endl;
	std::cout << "profile: " << device.getProfile() << std::endl;
	std::cout << "extensions: " << device.getExtensions() << std::endl;
	std::cout << "max compute units: " << device.getMaxComputeUnits() << std::endl;
	int dim =  device.getMaxWorkItemDimensions();
	std::cout << "max work item dimensions: " << dim << std::endl;
	std::cout << "max work item sizes: " << std::endl;
	std::list<size_t> sizes = device.getMaxWorkItemSizes();
	int i = 0;
	for(std::list<size_t>::iterator itr = sizes.begin(); itr != sizes.end(); ++itr)
		std::cout << " " << i++ << " - " << *itr << std::endl;
	std::cout << "max work group size: " << device.getMaxWorkGroupSize() << std::endl;
	std::cout << "preferred char vector width: " << device.getPreferredVectorWidthChar() << std::endl;
	std::cout << "preferred short vector width: " << device.getPreferredVectorWidthShort() << std::endl;
	std::cout << "preferred int vector width: " << device.getPreferredVectorWidthInt() << std::endl;
	std::cout << "preferred long vector width: " << device.getPreferredVectorWidthLong() << std::endl;
	std::cout << "preferred float vector width: " << device.getPreferredVectorWidthFloat() << std::endl;
	std::cout << "preferred double vector width: " << device.getPreferredVectorWidthDouble() << std::endl;
	std::cout << "max clock frequency: " << device.getMaxClockFrequency() << std::endl;
	std::cout << "address bits: " << device.getAddressBits() << std::endl;
	std::cout << "max mem alloc size: " << device.getMaxMemAllocSize() << " bytes" << std::endl;
	std::cout << "supports images: " << (device.hasImageSupport() ? "yes" : "no") << std::endl;
	if(device.hasImageSupport()) {
		std::cout << "max read image args: " << device.getMaxReadImageArgs() << std::endl;
		std::cout << "max write image args: " << device.getMaxWriteImageArgs() << std::endl;
		std::cout << "image 2d width: " << device.getImage2DMaxWidth() << std::endl;
		std::cout << "image 2d height: " << device.getImage2DMaxHeight() << std::endl;
		std::cout << "image 3d width: " << device.getImage3DMaxWidth() << std::endl;
		std::cout << "image 3d height: " << device.getImage3DMaxHeight() << std::endl;
		std::cout << "image 3d depth: " << device.getImage3DMaxDepth() << std::endl;
	}
	std::cout << "max samplers: " << device.getMaxSamplers() << std::endl;
	std::cout << "max parameter size: " << device.getMaxParameterSize() << " bytes" << std::endl;
	std::cout << "mem base address align: " << device.getMemBaseAddrAlign() << " bits" << std::endl;
	std::cout << "single precision FP properties: " << std::endl;
	cl_device_fp_config fp_config = device.getSingleFPConfig();
	if(fp_config & CL_FP_DENORM)
		std::cout << " denorms supported" << std::endl;
	if(fp_config & CL_FP_INF_NAN)
		std::cout << " INF and quiet NANs are supported" << std::endl;
	if(fp_config & CL_FP_ROUND_TO_NEAREST)
		std::cout << " round to nearest even rounding mode supported" << std::endl;
	if(fp_config & CL_FP_ROUND_TO_ZERO)
		std::cout << " round to zero rounding mode supported" << std::endl;
	if(fp_config & CL_FP_ROUND_TO_INF)
		std::cout << " round to +ve and -ve infinity rounding modes supported" << std::endl;
	if(fp_config & CL_FP_FMA)
		std::cout << " IEEE754-2008 fused multiply-add is supported" << std::endl;
	std::cout << "global cache type: ";
	cl_device_mem_cache_type cache_type = device.getGlobalMemCacheType();
	switch(cache_type) {
		case CL_NONE:
			std::cout << "none" << std::endl;
			break;
		case CL_READ_ONLY_CACHE:
			std::cout << "read only" << std::endl;
			break;
		case CL_READ_WRITE_CACHE:
			std::cout << "read write" << std::endl;
			break;
	}
	if(cache_type != CL_NONE) {
		 std::cout << "global cacheline size: " << device.getGlobalMemCacheLineSize() << " bytes" << std::endl;
		 std::cout << "global cache size: " << device.getGlobalMemCacheSize() << " bytes" << std::endl;
	}
	std::cout << "global memory size: " << device.getGlobalMemSize() << " bytes" << std::endl;
	std::cout << "max constant buffer size: " << device.getMaxConstantBufferSize() << " bytes" << std::endl;
	std::cout << "max constant args: " << device.getMaxConstantArgs() << std::endl;
	cl_device_local_mem_type local_mem_type = device.getLocalMemType();
	std::cout << "local mem type: ";
	switch(local_mem_type) {
		case CL_LOCAL:
			std::cout << "local" << std::endl;
			break;
		case CL_GLOBAL:
			std::cout << "global" << std::endl;
			break;
	}
	std::cout << "local mem size: " << device.getLocalMemSize() << " bytes" << std::endl;
	std::cout << "supports error correction: " << (device.hasErrorCorrectionSupport() ? "yes" : "no") << std::endl;
	std::cout << "profiling timer resolution: " << device.getType() << " ns" << std::endl;
	std::cout << "little endian: " << (device.isLittleEndian() ? "yes" : "no") << std::endl;
	std::cout << "compiler available: " << (device.hasCompiler() ? "yes" : "no") << std::endl;
	std::cout << "execution capabilities: " << std::endl;
	cl_device_exec_capabilities exec_cap = device.getExecutionCapabilities();
	if(exec_cap & CL_EXEC_KERNEL)
		std::cout << " can execute OpenCL kernels" << std::endl;
	if(exec_cap & CL_EXEC_NATIVE_KERNEL) 
		std::cout << " can execute native kernels" << std::endl;
	std::cout << "command queue properties: " << std::endl;
	cl_command_queue_properties queue_props = device.getQueueProperties();
	if(queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		std::cout << " supports out of order execution";
	if(queue_props & CL_QUEUE_PROFILING_ENABLE)
		std::cout << " supports profiling";

	std::cout << std::endl;
}

int
main(void)
{
        //const char* source = kernel_source;
        const int n = N;
        size_t size;
        cl_int error = 0;

        float* a = new float[n];
        for (int i = 0; i < n; ++i)
                a[i] = i;

        float* b = new float[n];
        for (int i = 0; i < n; ++i)
                b[i] = n - i;

        float* c = new float[n];

        cl_device_type device_type = CL_DEVICE_TYPE_CPU;

		CLMain cl;
		CLError err;

		std::list<CLPlatform> platforms = cl.getPlatforms(&err);
		checkErr(err || platforms.empty(), "failed to find platforms\n");
		CLPlatform platform = *platforms.begin();

		for(std::list<CLPlatform>::iterator itr = platforms.begin(); itr != platforms.end(); ++itr)
			printPlatformInfo(*itr);

		CLContext* context = platform.createContext(CL_DEVICE_TYPE_CPU, &err);
		checkErr(err || context == NULL, "failed to get a context for the chosen device\n");

		std::list<CLDevice> devices = context->getDevices(&err);
		checkErr(err || devices.empty(), "failed to find devices for context");
		CLDevice device = *devices.begin();

		for(std::list<CLDevice>::iterator itr = devices.begin(); itr != devices.end(); ++itr)
			printDeviceInfo(*itr);

		
		cl_context cid = context->getId();
		cl_device_id did = device.getId();
        
        cl_command_queue queue = clCreateCommandQueue(cid, did, 0, &error);
        assert(error == CL_SUCCESS);

        cl_mem ad = clCreateBuffer(cid, CL_MEM_READ_ONLY, n * sizeof(cl_float), 0, &error);
        assert(error == CL_SUCCESS);
        cl_mem bd = clCreateBuffer(cid, CL_MEM_READ_ONLY, n * sizeof(cl_float), 0, &error);
        assert(error == CL_SUCCESS);
        cl_mem cd = clCreateBuffer(cid, CL_MEM_READ_ONLY, n * sizeof(cl_float), 0, &error);
        assert(error == CL_SUCCESS);
       
        size = strlen(kernel_source);
        cl_program program = clCreateProgramWithSource(cid, 1, &kernel_source, &size, &error);
        assert(error == CL_SUCCESS);
        error = clBuildProgram(program, 0, 0, 0, 0, 0);
        assert(error == CL_SUCCESS);
        
        cl_kernel kernel = clCreateKernel(program, "vectorAdd", &error);
        assert(error == CL_SUCCESS);

        error = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &n);
        assert(error == CL_SUCCESS);
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &ad);
        assert(error == CL_SUCCESS);
        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bd);
        assert(error == CL_SUCCESS);
        error = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &cd);
        assert(error == CL_SUCCESS);

        error = clEnqueueWriteBuffer(queue, ad, CL_FALSE, 0, n * sizeof(cl_float), a, 0, 0, 0);
        assert(error == CL_SUCCESS);
        error = clEnqueueWriteBuffer(queue, bd, CL_FALSE, 0, n * sizeof(cl_float), b, 0, 0, 0);
        assert(error == CL_SUCCESS);

        size_t local_work_size = LOCAL_WORK_SIZE;
        size_t global_work_size = (size_t) ceil((double) n / local_work_size) * local_work_size;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global_work_size, &local_work_size, 0, 0, 0);
        assert(error == CL_SUCCESS);

        error = clEnqueueReadBuffer(queue, cd, CL_TRUE, 0, n * sizeof(cl_float), c, 0, 0, 0);
        assert(error == CL_SUCCESS);

        for (int i = 0; i < n; ++i)
                assert(c[i] == n);

        delete [] a;
        delete [] b;
        delete [] c;

        error = clReleaseMemObject(ad);
        assert(error == CL_SUCCESS);
        error = clReleaseMemObject(bd);
        assert(error == CL_SUCCESS);
        error = clReleaseMemObject(cd);
        assert(error == CL_SUCCESS);
        error = clReleaseKernel(kernel);
        assert(error == CL_SUCCESS);
        error = clReleaseProgram(program);
        assert(error == CL_SUCCESS);
        error = clReleaseCommandQueue(queue);
        assert(error == CL_SUCCESS);
        error = clReleaseContext(cid);
        assert(error == CL_SUCCESS);
}
