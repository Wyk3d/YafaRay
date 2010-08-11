#ifndef _CL_BASE
#define _CL_BASE

template <
	typename ObjectType
>
class CLObjectBase
{
	public:
		typedef ObjectType ObjType;
	protected:
		ObjType id;
		CLObjectBase(ObjType id) : id(id) {

		}
	public:
		const ObjType& getId() const {
			return id;
		}
};

template <
	typename ObjType,
	typename InfoType,
	CL_API_ENTRY cl_int (CL_API_CALL *InfoFunc)(ObjType, InfoType, size_t, void*, size_t*)
>
class CLInfoTraits
{
	public:
		template<typename T>
		T getInfo(ObjType id, InfoType info, CLError *error = NULL) const {
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

		std::string getStringInfo(ObjType id, InfoType info, CLError *error) const {
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
		std::list<T> getListInfo(ObjType id, InfoType info, CLError *error) const {
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

			for(size_t i = 0; i < size / sizeof(T); ++i)
				ilist.push_back(info_buf[i]);

			return ilist;
		}
};

template <
	typename ObjType,
	typename InfoType,
	CL_API_ENTRY cl_int (CL_API_CALL *InfoFunc)(ObjType, InfoType, size_t, void*, size_t*)
>
class CLObjectInfoBase :
	public CLObjectBase< ObjType >
{
	private:
		CLInfoTraits< ObjType, InfoType, InfoFunc> infoTraits;
	protected:
		CLObjectInfoBase(ObjType id) : CLObjectBase<ObjType>(id) {

		}
	public:
		template<typename T>
		T getInfo(InfoType info, CLError *error = NULL) const {
			return infoTraits.getInfo<T>(this->id, info, error);
		}

		std::string getStringInfo(InfoType info, CLError *error) const {
			return infoTraits.getStringInfo(this->id, info, error);
		}

		template<typename T>
		std::list<T> getListInfo(InfoType info, CLError *error) const {
			return infoTraits.getListInfo<T>(this->id, info, error);
		}
};

template <
	typename ObjType,
	CL_API_ENTRY cl_int (CL_API_CALL *ReleaseFunc)(ObjType),
	typename InfoType,
	CL_API_ENTRY cl_int (CL_API_CALL *InfoFunc)(ObjType, InfoType, size_t, void*, size_t*)
>
class CLObjectReleasableInfoBase :
	public CLObjectInfoBase< ObjType, InfoType, InfoFunc >
{
	protected:
		CLObjectReleasableInfoBase(ObjType id) :
			 CLObjectInfoBase< ObjType, InfoType, InfoFunc >(id) {
		}

		~CLObjectReleasableInfoBase() {
			assert(this->id == NULL);
		}
	public:
		void free(CLError *error = NULL) {
			CLErrGuard err(error);

			if(!(err = ReleaseFunc(this->id)) || error == NULL) {
				this->id = NULL;
				delete this;
			}
		}
};

#endif //_CL_BASE
