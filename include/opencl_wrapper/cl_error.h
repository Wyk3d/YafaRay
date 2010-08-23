#ifndef _CL_ERROR_H
#define _CL_ERROR_H

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

	OPENCL_WRAPPER_EXPORT CLCombinedError operator || (bool error_condition);

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
#ifdef CL_INVALID_GLOBAL_WORK_SIZE
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
#endif
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

	CLCombinedError(CLError err) : CLError(err), success(true) {

	}

	CLCombinedError(bool success) : success(success) {

	}

	bool isCombinedSuccess() const {
		return success && CLError::hasSucceeded();
	}
};

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

#endif // _CL_ERROR_H
