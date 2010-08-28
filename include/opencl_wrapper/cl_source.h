#ifndef _CL_SOURCE_H
#define _CL_SOURCE_H

#include <map>
#include <sstream>

#define CL_SRC(...) #__VA_ARGS__

template<class T>
class CLParamHolder {
	private:
		typedef std::map<std::string, T*> MapType;
		MapType &map;
		typename MapType::iterator itr;
		CLParamHolder(MapType &map, typename MapType::iterator itr) 
			: map(map), itr(itr) {
			
		}
	public:
		template<class U>
		friend class CLParamMap;

		template<class V>
		void operator = (V val) {
			if(itr->second) delete itr->second;
			itr->second = new typename T::template rebind<V>::other(val);
		}

		~CLParamHolder() {
			if(!itr->second)
				map.erase(itr);
		}
};

template<class T>
class CLParamMap {
	protected:
		typedef std::map<std::string, T*> MapType;
		MapType map;

	public:
		CLParamHolder<T> operator[](const std::string& str) {
			std::pair<typename MapType::iterator, bool> ib = map.insert(
				std::pair<std::string, T*>(str, NULL)
			);
			return CLParamHolder<T>(map, ib.first);
		}
};



template<class T> class CLSrcParamT;

class CLSrcParam {
	public:
		virtual void addTypeTo(std::ostream &s) = 0;
		virtual void addValTo(std::ostream &s) = 0;

		template<class T>
		struct rebind {
			typedef CLSrcParamT<T> other;
		};
};

template<typename T> struct CLSrcParamType { };
template<> struct CLSrcParamType<int> { 
	static void addTypeTo(std::ostream &s) { s << "int"; };
	static void addValTo(std::ostream &s, int val) { s << val; };
};
template<> struct CLSrcParamType<float> { 
	static void addTypeTo(std::ostream &s) { s << "float"; };
	static void addValTo(std::ostream &s, float val) { s << val; };
};
template<> struct CLSrcParamType<unsigned int> {
	static void addTypeTo(std::ostream &s) { s << "unsigned int"; };
	static void addValTo(std::ostream &s, unsigned int val) { s << val; };
};

template<class T>
class CLSrcParamT : public CLSrcParam {
	private:
		T val;
	public:
		CLSrcParamT(const T& val) : val(val) {

		}

		void addTypeTo(std::ostream &s) {
			CLSrcParamType<T>::addTypeTo(s);
		}

		void addValTo(std::ostream &s) {
			CLSrcParamType<T>::addValTo(s, val);
		}
};

class CLSrcParamMap
	: public CLParamMap<CLSrcParam>
{
	private:
		std::string name;
	public:
		CLSrcParamMap(const std::string &name) : name(name) {

		}

		void generate(std::ostringstream &ss) {
			ss << "#define PARAM_CONST_" << name << "(__param_var) ";
			ss << "DECL_PARAM_" << name << "(__param_var, __constant)\n";

			ss << "#define PARAM_PRIVATE_" << name << "(__param_var) ";
			ss << "DECL_PARAM_" << name << "(__param_var, __private)\n";

			ss << "#define PARAM_LOCAL_" << name << "(__param_var) ";
			ss << "DECL_PARAM_" << name << "(__param_var, __local)\n";

			ss << "#define DECL_PARAM_" << name << "(__param_var, __param_space)\\\n";
			ss << "typedef struct {\\\n";
			for(MapType::iterator itr = map.begin(); itr != map.end(); ++itr) {
				if(itr->second) {
					itr->second->addTypeTo(ss);
					ss << " " << itr->first << ";\\\n";
				}
			}
			ss << "} const_t;\\\n";

			ss << "__param_space const_t __param_var = {";
			for(MapType::iterator itr = map.begin(); itr != map.end(); ++itr) {
				if(itr->second) {
					if(itr != map.begin())
						ss << ", ";
					itr->second->addValTo(ss);
				}
			}
			ss << "}\n";
		}

		operator std::string() {
			std::ostringstream ss;
			generate(ss);
			return ss.str();
		}
};

class CLSrcGenerator
{
	private:
		std::ostringstream ss;
		std::string final_str;
		const char *program_source;

	public:
		CLSrcGenerator(const char *program_source)
			: program_source(program_source)
		{
			
		}

		void addDebug(bool debug) {
			if(debug) ss << "#define IF_DEBUG(...) __VA_ARGS__\n";
			else ss << "#define IF_DEBUG(...)\n";
		}

		void addParamMap(CLSrcParamMap &map) {
			map.generate(ss);
		}

		operator const char *() {
			final_str = ss.str() + program_source;
			ss.clear();
			return final_str.c_str();
		}
};

#endif //_CL_SOURCE_H