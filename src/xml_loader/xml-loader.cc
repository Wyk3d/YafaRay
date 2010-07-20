#include <yafray_config.h>
#include <cstdlib>
#include <iostream>
#include <cctype>
#include <algorithm>

#ifdef WIN32
	#ifndef __MINGW32__
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

#include <core_api/scene.h>
#include <core_api/environment.h>
#include <core_api/integrator.h>
#include <core_api/imagefilm.h>
#include <yafraycore/xmlparser.h>
#include <yaf_revision.h>
#include <utilities/console_utils.h>
#include <yafraycore/imageOutput.h>

#include "../gui/yafqtapi.h"

using namespace::yafaray;

#include <yafraycore/triangle.h>
#include <yafraycore/meshtypes.h>
#include <limits>

class DiskTessellator
{
	protected:
		int points_per_tri;
		std::vector<point3d_t> tri_points;
	public:
		DiskTessellator(int points_per_tri) : points_per_tri(points_per_tri) {
			tri_points.resize(points_per_tri);
		}

		void tessellate(point3d_t & p, float r, vector3d_t &n) {
			float nz_sq = n.z * n.z;
			int cmap[3] = {0,1,2};
			if(nz_sq < 1e-5) {
				if(n.x * n.x < 1e-5) {
					cmap[1] = 2;
					cmap[2] = 1;
				} else {
					cmap[0] = 2;
					cmap[2] = 0;
				}
				nz_sq = n[cmap[2]] * n[cmap[2]];
			}
			float nx = n[cmap[0]], ny = n[cmap[1]];

			for(int k = 0; k < points_per_tri; k++) {
				float beta_k = k * M_2PI / points_per_tri;
				float cos_beta_k = cos(beta_k);
				float sin_beta_k = sin(beta_k);

				float sak_p = (nx * cos_beta_k + ny * sin_beta_k);
				float sak_d = ( sak_p * sak_p + nz_sq);
				if(sak_d < 1e-5) {
					Y_ERROR << "division by zero warning";
				}
				float sinsq_alfa_k = nz_sq / sak_d;
				//float sinsq_alfa_k = nz_sq / ( sak_p * sak_p + nz_sq);

				float sin_alfa_k = sqrt(sinsq_alfa_k);
				float cos_alfa_k = sqrt(1 - sinsq_alfa_k);
				float r_sin_alfa_k = r * sin_alfa_k;

				point3d_t &q = tri_points[k];
				q[cmap[0]] = p[cmap[0]] + r_sin_alfa_k * cos_beta_k;
				q[cmap[1]] = p[cmap[1]] + r_sin_alfa_k * sin_beta_k;
				q[cmap[2]] = p[cmap[2]] + r * cos_alfa_k;

				float cos_n = (q-p) * n / r;
				if(abs(cos_n) > 0.01) {
					Y_ERROR << "cos_n is " << cos_n << yendl;
				}
			}
		}
};

class DiskSceneTessellator : public DiskTessellator
{
	protected:
		scene_t *scene;

	public:
		DiskSceneTessellator(scene_t *scene, int points_per_tri) : DiskTessellator(points_per_tri), scene(scene) {

		}

		void tessellate(const material_t *mat, point3d_t & p, float r, vector3d_t &n) {
			DiskTessellator::tessellate(p, r, n);

			bool hasOrco = false;
			bool hasUV = false;
			int type = 0;

			if(!scene->startGeometry()) exit(0);
			scene->startTriMesh(scene->getNextFreeID(), points_per_tri + 1, points_per_tri, hasOrco, hasUV, type);
			scene->addVertex(p);

			for(int k = 0; k < points_per_tri; ++k) {
				scene->addVertex(tri_points[k]);
			}

			for(int k = 0; k < points_per_tri; k++) {
				point3d_t &a = p, &b = tri_points[k], &c = tri_points[(k+1) % points_per_tri];
				vector3d_t gn = ((b-a)^(c-a)).normalize();
				float cos_n = gn * n;
				const float similarity_threshold = 1.0 - 1e-4;
				if(cos_n >= similarity_threshold) {
					scene->addTriangle(0, 1 + k, 1 + (k+1) % points_per_tri, mat); 
				} else if(-cos_n >= similarity_threshold) {
					scene->addTriangle(0, 1 + (k+1) % points_per_tri, 1 + k, mat);
				} else {
					Y_ERROR << "invalid triangle normal - similarity between "
						<< "(" << n.x << "," << n.y << "," << n.z << ") and "
						<< "(" << gn.y << "," << gn.z << "," << gn.z << ") "
						<< "is |" << cos_n << "| < " << similarity_threshold << yendl;
				}
			}

			if(!scene->endTriMesh()) exit(0);
			if(!scene->endGeometry()) exit(0);
		}
};

template <
	class MemberType, 
	class ContainerIteratorType, 
	class FuncHolder,
	MemberType &(FuncHolder::*GetMemberFunc)(typename ContainerIteratorType::reference)
>
class MemberIterator : public std::iterator_traits<const MemberType*>
{
	protected:
		ContainerIteratorType container_iter;
		FuncHolder funcHolder;
	public:
		MemberIterator(const FuncHolder &funcHolder, const ContainerIteratorType &iter) :
		  container_iter(iter), funcHolder(funcHolder) {}

		MemberIterator(const MemberIterator &iter) {
			container_iter = iter->container_iter;
		}

		MemberType &operator *() {
			return ((&funcHolder)->*GetMemberFunc)(typename ContainerIteratorType::reference(*container_iter));
		}

		MemberIterator& operator++() {
			++container_iter;
			return *this;
		}

		MemberIterator& operator--() {
			--container_iter;
			return *this;
		}

		bool operator == (const MemberIterator &iter) {
			return container_iter == iter.container_iter;
		}

		bool operator != (const MemberIterator &iter) {
			return !((*this)==iter);
		}
};

class BestCandidateSampler {
	protected:
		struct cand {
			point3d_t p;
			bool t;
			float min_dist;
		};

		typedef std::vector<cand> CandVecType;
		CandVecType cands;
		int nr_to_generate;
		int nr_to_keep;

		friend class SampleIterator;

		class SampleIterator : public std::iterator_traits<const cand*>
		{
			protected:
				BestCandidateSampler &sampler;
				CandVecType::iterator iter;
			public:

				SampleIterator(BestCandidateSampler &sampler, CandVecType::iterator &iter) 
					: sampler(sampler), iter(iter) {}

				SampleIterator(const SampleIterator &iter)
					: sampler(iter.sampler), iter(iter.iter) {}

				reference operator *() {
					return *iter;
				}

				SampleIterator& operator++() {
					do {
						++iter;
					} while(iter != sampler.cands.end() && !iter->t);
					return *this;
				}

				SampleIterator& operator--() {
					do {
						--iter;
					} while(!iter->t);
					return *this;
				}

				bool operator ==(const SampleIterator &iter) {
					return this->iter == iter.iter;
				}

				bool operator !=(const SampleIterator &iter) {
					return !((*this)==iter);
				}

				void operator = (SampleIterator &iter) {
					this->iter = iter.iter;
					this->sampler = iter.sampler;
				}
		};

		struct GetPointFuncHolder {
			const point3d_t &getPoint(const cand& c) {
				return c.p;
			}
		};

	public:

		typedef MemberIterator<const point3d_t, SampleIterator, 
			GetPointFuncHolder, &GetPointFuncHolder::getPoint> iterator;

		iterator begin() {
			CandVecType::iterator itr = cands.begin();
			while(itr != cands.end() && !itr->t)
				++itr;

			return iterator(GetPointFuncHolder(), 
				SampleIterator(*this, itr));
		}

		iterator end() {
			CandVecType::iterator itr = cands.end();
			return iterator(GetPointFuncHolder(), 
				SampleIterator(*this, itr));
		}

		BestCandidateSampler() : nr_to_generate(-1), nr_to_keep(-1) {
			
		}

		template<class PointGenFunc> 
		void gen_candidates(PointGenFunc gen_functor, int nr_to_keep, int nr_to_generate = -1) {
			this->nr_to_keep = nr_to_keep;
			if(nr_to_generate < 0) {
				nr_to_generate = nr_to_keep * 5;
			}
			this->nr_to_generate = nr_to_generate;

			cands.resize(nr_to_generate);

			printf("generating %d candiates\n", nr_to_generate);

			for(int j = 0; j < nr_to_generate; j++)
			{
				cands[j].p = gen_functor(j);
				cands[j].t = false;
				cands[j].min_dist = std::numeric_limits<float>::max();
			}
		}

		void gen_samples(int nr_to_keep = -1) {
			if(nr_to_keep == -1) {
				nr_to_keep = this->nr_to_keep;
			} else {
				if(nr_to_keep >= nr_to_generate) {
					nr_to_keep = nr_to_generate;
				}
				// TODO: check number already generated and
				// generate the difference
			}
								
			printf("generating %d samples of %d\n", nr_to_keep, nr_to_generate);
			
			
			int ppoz = 0;

			// the current minimum distance from the taken samples
			// to each non-taken candidate is stored in cand.min_dist
			for(int j = 0; j < nr_to_keep; j++)
			{
				point3d_t &p = cands[ppoz].p;
				cands[ppoz].t = true;

				// for each new sample, update the min_dist in the candidates
				// and find the next sample by taking that highest min_dist
				float max_mind = std::numeric_limits<float>::min();
				for(int k = 0; k < nr_to_generate; k++) {
					if(!cands[k].t) {
						float d = (cands[k].p - p).lengthSqr();
						float &mind = cands[k].min_dist;
						if(d < mind) 
							mind = d;

						if(mind > max_mind) {
							max_mind = mind;
							ppoz = k;
						}
					}
				}
			}
		}
};


struct disk
{
	point3d_t p;
	int n_idx;
	float r;
	const material_t *mat;
	disk(const material_t *mat, const point3d_t &p, int n_idx, float r) : mat(mat), p(p), n_idx(n_idx), r(r) {}
};

typedef std::vector<disk> DiskVectorType;

void build_disk_hierarchy(DiskVectorType &v, int s, int e) {
	if(s >= e - 1)		// no need to sort single point
		return;

	float max_c[3], min_c[3];
	for(int i = 0; i < 3; i++) {
		max_c[i] = std::numeric_limits<float>::min();
		min_c[i] = std::numeric_limits<float>::max();
	}

	for(int poz = s; poz < e; poz++) {
		point3d_t &p = v[poz].p;
		for(int i = 0; i < 3; i++) {
			if(p[i] < min_c[i]) min_c[i] = p[i];
			if(p[i] > max_c[i]) max_c[i] = p[i];
		}
	}

	float max_ex = std::numeric_limits<float>::min();
	int ex = 0;
	for(int i = 0; i < 3; i++) {
		float exi = max_c[i] - min_c[i];
		if(exi > max_ex) {
			max_ex = exi;
			ex = i;
		}
	}

	struct CoordinateComparator
	{
		int coord;
		CoordinateComparator(int coord) :
			coord(coord) {}
		bool operator()(const disk& a, const disk &b) {
			return a.p[coord] < b.p[coord];
		}
	} comp(ex);

	std::sort(v.begin() + s, v.begin() + e, comp);

	build_disk_hierarchy(v, s, (s+e)/2);
	build_disk_hierarchy(v, (s+e)/2 + 1, e);
}

void test(scene_t *scene) {


	std::vector<vector3d_t> normals;
	DiskVectorType disks;

	scene_t::objDataArray_t &meshes = scene->getMeshes();
	for(scene_t::objDataArray_t::iterator itr = meshes.begin(); itr != meshes.end(); ++itr)
	{
		triangleObject_t *obj = itr->second.obj;
		int nr_mesh_primitives = obj->numPrimitives();
		const triangle_t **tris = new const triangle_t*[nr_mesh_primitives];
		itr->second.obj->getPrimitives(tris);

		for(int i = 0; i < nr_mesh_primitives; ++i) {
			const triangle_t *t = tris[i];
			point3d_t a, b, c;
			t->getVertices(a, b, c);
			vector3d_t n = t->getNormal();
			if(n.lengthSqr() < 1.0 - 1e-5) {
				Y_ERROR << "found triangle " 
					<< "(" << a.x << "," << a.y << "," << a.z << "), "
					<< "(" << b.x << "," << b.y << "," << b.z << "), "
					<< "(" << c.x << "," << c.y << "," << c.z << ") "
					<< "with invalid normal "
					<< "(" << n.x << "," << n.y << "," << n.z << ")" << yendl;
				continue;
			}
			normals.push_back(n);
			
			vector3d_t ab = b - a, ac = c - a;
			float area = 0.5 * (ab ^ ac).length();

			float r = 0.1;
			//float area_multiplier = 2*sqrt(2.0);
			float area_multiplier = 1;
			int nr_to_keep = std::max((int)(area * area_multiplier / (r*r*M_PI)), 1);
			printf("radius: %f\n");

			BestCandidateSampler sampler;

			struct PointGenFunc {
				point3d_t &a;
				vector3d_t &ab, &ac;
				PointGenFunc(point3d_t &a, vector3d_t &ab, vector3d_t &ac)
					: a(a), ab(ab), ac(ac) 
				{
				}

				point3d_t operator()(int) {
					float u = ourRandom();
					float v = ourRandom();
					//printf("(%f,%f,%f)", points[j].p.x, points[j].p.y, points[j].p.z);
					return a + v * ac + (1-v) * u * ab;
				}
			} gen_func(a, ab, ac);
			
			sampler.gen_candidates(gen_func, nr_to_keep);
			sampler.gen_samples();

			for(BestCandidateSampler::iterator itr = sampler.begin(); itr != sampler.end(); ++itr)
				disks.push_back(disk(t->getMaterial(), (*itr), i, r));
		}
	}

	build_disk_hierarchy(disks, 0, (int)disks.size());

	// displace the original scene out of view to show only the current scene
	// TODO: create a new scene or remove the meshes from the original scene
	for(scene_t::objDataArray_t::iterator itr = meshes.begin(); itr != meshes.end(); ++itr) {
		for(std::vector<point3d_t>::iterator p_itr = itr->second.points.begin(); p_itr != itr->second.points.end(); ++p_itr)
			(*p_itr) += point3d_t(3,3,3);
	}

	DiskSceneTessellator tes(scene, 10);

	for(std::vector<disk>::iterator itr = disks.begin(); itr != disks.end(); ++itr) {
		tes.tessellate(itr->mat, itr->p, itr->r, normals[itr->n_idx]);
	}
}

int main(int argc, char *argv[])
{
	cliParser_t parse(argc, argv, 2, 1, "You need to set at least a yafaray's valid XML file.");
	
#ifdef RELEASE
	std::string version = std::string(VERSION);
#else
	std::string version = std::string(YAF_SVN_REV);
#endif

	std::string xmlLoaderVersion = "YafaRay XML loader version 0.2";

	renderEnvironment_t *env = new renderEnvironment_t();
	
	// Plugin load
	std::string ppath = "";
	
	if (env->getPluginPath(ppath))
	{
		Y_DEBUG(1) << "The plugin path is: " << ppath << yendl;
		env->loadPlugins(ppath);
	}
	else
	{
		Y_ERROR << "Getting plugin path from render environment failed!" << yendl;
		return 1;
	}
	
	std::vector<std::string> formats = env->listImageHandlers();
	
	std::string formatString = "";
	for(size_t i = 0; i < formats.size(); i++)
	{
		formatString.append("                                       " + formats[i]);
		if(i < formats.size() - 1) formatString.append("\n");
	}

	parse.setAppName(xmlLoaderVersion,
	"[OPTIONS]... <input xml file> [output filename]\n<input xml file> : A valid yafaray XML file\n[output filename] : The filename of the rendered image without extension.\n*Note: If output filename is ommited the name \"yafaray\" will be used instead.");
	// Configuration of valid flags
	
	parse.setOption("v","version", true, "Displays this program's version.");
	parse.setOption("h","help", true, "Displays this help text.");
	parse.setOption("op","output-path", false, "Uses the path in <value> as rendered image output path.");
	parse.setOption("f","format", false, "Sets the output image format, available formats are:\n\n" + formatString + "\n                                       Default: tga.\n");
	parse.setOption("t","threads", false, "Overrides threads setting on the XML file, for auto selection use -1.");
	parse.setOption("a","with-alpha", true, "Enables saving the image with alpha channel.");
	parse.setOption("dp","draw-params", true, "Enables saving the image with a settings badge.");
	parse.setOption("ndp","no-draw-params", true, "Disables saving the image with a settings badge (warning: this overrides --draw-params setting).");
	parse.setOption("cs","custom-string", false, "Sets the custom string to be used on the settings badge.");
	parse.setOption("z","z-buffer", true, "Enables the rendering of the depth map (Z-Buffer) (this flag overrides XML setting).");
	parse.setOption("nz","no-z-buffer", true, "Disables the rendering of the depth map (Z-Buffer) (this flag overrides XML setting).");
	
	bool parseOk = parse.parseCommandLine();
	
	if(parse.getFlag("h"))
	{
		parse.printUsage();
		return 0;
	}
	
	if(parse.getFlag("v"))
	{
		Y_INFO << xmlLoaderVersion << yendl << "Built with YafaRay version " << version << yendl;
		return 0;
	}
	
	if(!parseOk)
	{
		parse.printError();
		parse.printUsage();
		return 0;
	}
	
	bool alpha = parse.getFlag("a");
	std::string format = parse.getOptionString("f");
	std::string outputPath = parse.getOptionString("op");
	int threads = parse.getOptionInteger("t");
	bool drawparams = parse.getFlag("dp");
	bool nodrawparams = parse.getFlag("ndp");
	std::string customString = parse.getOptionString("cs");
	bool zbuf = parse.getFlag("z");
	bool nozbuf = parse.getFlag("nz");
	
	if(format.empty()) format = "tga";
	bool formatValid = false;
	
	for(size_t i = 0; i < formats.size(); i++)
	{
		if(formats[i].find(format) != std::string::npos) formatValid = true;
	}
	
	if(!formatValid)
	{
		Y_ERROR << "Couldn't find any valid image format, image handlers missing?" << yendl;
		return 1;
	}
	
	const std::vector<std::string> files = parse.getCleanArgs();
	
	if(files.size() == 0)
	{
		return 0;
	}
	
	std::string outName = "yafray." + format;
	
	if(files.size() > 1) outName = files[1] + "." + format;
	
	std::string xmlFile = files[0];
	
	//env->Debug = debug; //disabled until proper debugging messages are set throughout the core

	// Set the full output path with filename
	if (outputPath.empty())
	{
		outputPath = outName;
	}
	else if (outputPath.at(outputPath.length() - 1) == '/')
	{
		outputPath += outName;
	}
	else if (outputPath.at(outputPath.length() - 1) != '/')
	{
		outputPath += "/" + outName;
	}
	
	scene_t *scene = new scene_t();
	env->setScene(scene);
	paraMap_t render;
	
	bool success = parse_xml_file(xmlFile.c_str(), scene, env, render);
	if(!success) exit(1);

	test(scene);
	
	int width=320, height=240;
	render.getParam("width", width); // width of rendered image
	render.getParam("height", height); // height of rendered image
	
	if(threads >= -1) render["threads"] = threads;
	
	if(drawparams)
	{
		render["drawParams"] = true;
		if(!customString.empty()) render["customString"] = customString;
	}
	
	if(nodrawparams) render["drawParams"] = false;
	
	if(zbuf) render["z_channel"] = true;
	if(nozbuf) render["z_channel"] = false;
	
	bool use_zbuf = false;
	render.getParam("z_channel", use_zbuf);
	
	// create output
	colorOutput_t *out = NULL;

	paraMap_t ihParams;
	ihParams["type"] = format;
	ihParams["width"] = width;
	ihParams["height"] = height;
	ihParams["alpha_channel"] = alpha;
	ihParams["z_channel"] = use_zbuf;
	
	imageHandler_t *ih = env->createImageHandler("outFile", ihParams);

	if(ih)
	{
		out = new imageOutput_t(ih, outputPath);
		if(!out) return 1;				
	}
	else return 1;
	
	if(! env->setupScene(*scene, render, *out) ) return 1;
	
	scene->render();
	env->clearAll();

	imageFilm_t *film = scene->getImageFilm();

	delete film;
	delete out;
	
	return 0;
}
