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
	typename InfoType
>
class CLInfo
{
	public:
		typedef cl_int (*InfoFuncType)(ObjType, InfoType, size_t, void*, size_t*);

		template<typename T>
		static T getInfo(ObjType id, InfoType info,	InfoFuncType InfoFunc, CLError *error = NULL)
		{
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

		static std::string getStringInfo(ObjType id, InfoType info,	InfoFuncType InfoFunc, CLError *error = NULL)
		{
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
		static std::list<T> getListInfo(ObjType id, InfoType info,	InfoFuncType InfoFunc, CLError *error = NULL)
		{
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
	typename CLObject
>
class CLObjectInfoBase :
	public CLObjectBase< ObjType >
{
	protected:
		typedef CLInfo< ObjType, InfoType> Info;

		CLObjectInfoBase(ObjType id) : CLObjectBase<ObjType>(id) {

		}
	public:
		template<typename T>
		T getInfo(InfoType info, CLError *error = NULL) const {
			return Info::template getInfo<T>(this->id, info, &CLObject::InfoFunc, error);
		}

		std::string getStringInfo(InfoType info, CLError *error) const {
			return Info::getStringInfo(this->id, info, &CLObject::InfoFunc, error);
		}

		template<typename T>
		std::list<T> getListInfo(InfoType info, CLError *error) const {
			return Info::template getListInfo<T>(this->id, info, &CLObject::InfoFunc, error);
		}
};

template <
	typename ObjType,
	typename InfoType,
	typename CLObject
>
class CLObjectReleasableInfoBase :
	public CLObjectInfoBase< ObjType, InfoType, CLObject >
{
	protected:
		CLObjectReleasableInfoBase(ObjType id) :
			 CLObjectInfoBase< ObjType, InfoType, CLObject >(id) 
		{ }

		 ~CLObjectReleasableInfoBase() {
			 assert(this->id == NULL);
		 }
	public:
		void free(CLError *error = NULL) {
			CLErrGuard err(error);

			if(!(err = CLObject::ReleaseFunc(this->id)) || error == NULL) {
				this->id = NULL;
				delete static_cast<CLObject*>(this);
			}
		}
};

#endif //_CL_BASE
