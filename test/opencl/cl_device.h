#ifndef _CL_DEVICE_H
#define _CL_DEVICE_H

typedef CLObjectInfoBase <
	cl_device_id,
	cl_device_info,
	&clGetDeviceInfo
> CLDeviceBase;

class CLPlatform;

class CLDevice :
	public CLDeviceBase
{
private:
	CLDevice(cl_device_id id) : CLDeviceBase(id) {

	}
public:
	friend class CLContext;
	friend class CLPlatform;

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
};

#endif //_CL_DEVICE_H