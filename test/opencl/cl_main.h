#ifndef _CL_MAIN_H
#define _CL_MAIN_H

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

#endif //_CL_MAIN_H
