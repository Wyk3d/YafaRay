#ifndef _CL_BASE
#define _CL_BASE

template <
	typename ObjType,
	typename InfoType, 
	CL_API_ENTRY cl_int (CL_API_CALL *InfoFunc)(ObjType, InfoType, size_t, void*, size_t*) 
>
class CLObjectBase
{
protected:
	ObjType id;
	CLObjectBase(ObjType id) : id(id) {

	}
public:
	ObjType getId() {
		return id;
	}

	template<typename T>
	T getInfo(InfoType info, CLError *error = NULL) {
		CLErrGuard err(error);
		T ret = 0;

		// get the size of the info string
		size_t size;
		if((err = InfoFunc(id,
			info,
			0,
			NULL,
			&size)) || sizeof(T) != size)
			return ret;

		// get the info string

		if(err = InfoFunc(id,
			info,
			size,
			&ret,
			NULL)) 
			return ret;

		return ret;
	}

	std::string getStringInfo(InfoType info, CLError *error) {
		CLErrGuard err(error);

		// get the size of the info string
		size_t size;
		if(err = InfoFunc(id,
			info,
			0,
			NULL,
			&size)) 
			return "";

		// get the info string
		char *pbuf = new char[size];
		if(err = InfoFunc(id,
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
	std::list<T> getListInfo(InfoType info, CLError *error) {
		CLErrGuard err(error);
		std::list<T> ilist;

		// get the size of the info list
		size_t size;
		if(err = InfoFunc(id,
			info,
			0,
			NULL,
			&size)) 
			return ilist;

		// get the info list
		T *info_buf = new T[size/sizeof(T)];
		if(err = InfoFunc(id,
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

#endif //_CL_BASE