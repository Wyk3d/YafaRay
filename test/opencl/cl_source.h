#ifndef _CL_SOURCE_H
#define _CL_SOURCE_H

#include <map>
#include <sstream>

#define CL_SRC(...) #__VA_ARGS__

class CLParam {
	public:
		virtual void addTypeTo(std::ostream &s) = 0;
		virtual void addValTo(std::ostream &s) = 0;
};

template<class T> struct CLParamType { };
template<> struct CLParamType<int> { 
	static void addTypeTo(std::ostream &s) { s << "int"; };
	static void addValTo(std::ostream &s, int val) { s << val; };
};
template<> struct CLParamType<float> { 
	static void addTypeTo(std::ostream &s) { s << "float"; };
	static void addValTo(std::ostream &s, int val) { s << val; };
};
template<> struct CLParamType<unsigned int> {
	static void addTypeTo(std::ostream &s) { s << "unsigned int"; };
	static void addValTo(std::ostream &s, int val) { s << val; };
};

template<class T>
class CLParamT : public CLParam {
	private:
		T val;
	public:
		CLParamT(const T& val) : val(val) {

		}

		void addTypeTo(std::ostream &s) {
			CLParamType<T>::addTypeTo(s);
		}
		
		void addValTo(std::ostream &s) {
			CLParamType<T>::addValTo(s, val);
		}
};

class CLSrcConst;

class CLParamHolder {
	private:
		typedef std::map<std::string, CLParam*> MapType;
		MapType &map;
		MapType::iterator itr;
		CLParamHolder(MapType &map, MapType::iterator itr) 
			: map(map), itr(itr) {
			
		}
	public:
		friend class CLSrcConst;

		template<class T>
		void operator = (T val) {
			if(itr->second) delete itr->second;
			itr->second = new CLParamT<T>(val);
		}

		~CLParamHolder() {
			if(!itr->second)
				map.erase(itr);
		}
};

class CLSrcConst {
	private:
		typedef std::map<std::string, CLParam*> MapType;
		MapType map;

	public:
		friend class CLParamHolder;
		CLParamHolder operator[](const std::string& str) {
			std::pair<MapType::iterator, bool> ib = map.insert(
				std::pair<std::string, CLParam*>(str, NULL)
			);
			return CLParamHolder(map, ib.first);
		}

		operator std::string() {
			std::ostringstream ss;
			ss << "typedef struct {\n";
			for(MapType::iterator itr = map.begin(); itr != map.end(); ++itr) {
				if(itr->second) {
					itr->second->addTypeTo(ss);
					ss << " " << itr->first << ";\n";
				}
			}
			ss << "} const_t;\n";
			ss << "__constant const_t ct = {\n";
			for(MapType::iterator itr = map.begin(); itr != map.end(); ++itr) {
				if(itr->second) {
					if(itr != map.begin())
						ss << ", ";
					itr->second->addValTo(ss);
				}
			}
			ss << "};\n";
			return ss.str();
		}
};
 

#endif //_CL_SOURCE_H