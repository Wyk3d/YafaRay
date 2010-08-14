
#ifndef Y_PHOTONINTEGR_H
#define Y_PHOTONINTEGR_H

#include <yafray_config.h>
#include <core_api/environment.h>
#include <core_api/material.h>
#include <core_api/background.h>
#include <core_api/light.h>
#include <core_api/imagefilm.h>
#include <core_api/camera.h>
#include <yafraycore/tiledintegrator.h>
#include <yafraycore/photon.h>
#include <yafraycore/monitor.h>
#include <yafraycore/ccthreads.h>
#include <yafraycore/timer.h>
#include <yafraycore/spectrum.h>
#include <utilities/sample_utils.h>
#include <integrators/integr_utils.h>
#include <opencl_wrapper/cl_wrapper.h>

__BEGIN_YAFRAY

class YAFRAYPLUGIN_EXPORT photonIntegratorGPU_t: public tiledIntegrator_t
{
	public:
		photonIntegratorGPU_t(unsigned int dPhotons, unsigned int cPhotons, bool transpShad=false, int shadowDepth = 4, float dsRad = 0.1f, float cRad = 0.01f);
		~photonIntegratorGPU_t();
		virtual bool preprocess();
		virtual colorA_t integrate(renderState_t &state, diffRay_t &ray/*, sampler_t &sam*/) const;
		static integrator_t* factory(paraMap_t &params, renderEnvironment_t &render);
		void onSceneUpdate();

		struct PHLeaf
		{
			point3d_t c;	// disk center
			vector3d_t n;	// disk normal
			/*int mat_type;	// disk material type
			PHLeaf(const point3d_t &c, const vector3d_t &n, int mat_type)
				: c(c), n(n), mat_type(mat_type) {}*/
			/*const triangle_t *t;
			PHLeaf(const point3d_t &c, const vector3d_t &n, const triangle_t *t)
				: c(c), n(n), t(t) {}*/
			int tri_idx;
			PHLeaf(const point3d_t &c, const vector3d_t &n, int tri_idx)
				: c(c), n(n), tri_idx(tri_idx) {}
			PHLeaf() {}
		};

		struct PHInternalNode
		{
			point3d_t c;	// bounding sphere center
			float r;		// bounding sphere radius
			PHInternalNode() {}
			/*PHInternalNode(const point3d_t &c, float r) 
				: c(c), r(r) {}*/
			int coord;
			float M;
			PHInternalNode(const point3d_t &c, float r, int coord, float M) 
				: c(c), r(r), coord(coord), M(M) {}
		};

		struct PHTriangle
		{
			point3d_t a, b, c;
			PHTriangle(const point3d_t &a, const point3d_t &b, const point3d_t &c)
				: a(a), b(b), c(c) {}
			PHTriangle() {}
		};

		struct PHRay
		{
			point3d_t p;
			vector3d_t r;
			PHRay(const point3d_t &p, const vector3d_t &r)
				: p(p), r(r) {}
			PHRay() {}
		};

		struct Disk
		{
			point3d_t c;
			int tri_idx;
			Disk() {}
			Disk(point3d_t c, int tri_idx)
				: c(c), tri_idx(tri_idx) {}
		};

		typedef std::vector<Disk> DiskVectorType;
		typedef std::vector<vector3d_t> NormalVectorType;

		struct PHierarchy
		{
			std::vector<PHInternalNode> int_nodes;
			std::vector<PHLeaf> leaves;
			std::vector<PHTriangle> tris;
			float leaf_radius;
		};

		void build_disk_hierarchy(PHierarchy &ph, std::vector<const triangle_t *> &prims, int node_poz, DiskVectorType &v, int s, int e);
		void generate_points(DiskVectorType &disks, std::vector<const triangle_t *> &prims, PHierarchy &ph, scene_t *scene);

		bool renderTile(renderArea_t &a, int n_samples, int offset, bool adaptive, int threadID);

	protected:
		color_t finalGathering(renderState_t &state, const surfacePoint_t &sp, const vector3d_t &wo) const;
		color_t estimateOneDirect(renderState_t &state, const surfacePoint_t &sp, vector3d_t wo, const std::vector<light_t *>  &lights, int d1, int n)const;

		CLDevice getOpenCLDevice();
		CLPlatform getOpenCLPlatform();

		void upload_hierarchy(PHierarchy &ph);

		friend class RayTest;

		struct RayTest
		{
			photonIntegratorGPU_t &pi;
			std::vector<PHInternalNode> &int_nodes;
			std::vector<PHLeaf> &leaves;
			std::vector<PHTriangle> &tris;
			float leaf_radius;
			scene_t *scene;
			std::vector<const triangle_t*> &prims;

			bool set_test;
			bool hit_test;
			const triangle_t *tri_test;
			point3d_t p_test;
			float t_test;
			int tri_idx_test;

			std::vector<int> cand_leaves;
			std::vector<int> cand_tris;

			renderState_t *state;
			diffRay_t ray;

			surfacePoint_t sp_ref;
			float t_ref;
			bool hit_ref;

			RayTest(photonIntegratorGPU_t &pi, PHierarchy &ph)
				: pi(pi), int_nodes(ph.int_nodes), leaves(ph.leaves), tris(ph.tris),
				leaf_radius(ph.leaf_radius), scene(pi.scene), prims(pi.prims)
			{

			}

			void test_rays(renderState_t &state);
			void init_test(const diffRay_t &ray);

			void test_intersect_sh();
			void test_intersect_brute();
			void test_intersect_kd();
			void test_intersect_stored();

			void from_cand_leaves();
			void from_cand_tris();

			void from_tri_idx(int tri_idx);
			void from_leaf_idx(int leaf_idx);

			void list_ray_candidates(bool exact = false);
		};

		
		bool getSPforRay(const diffRay_t &ray, renderState_t &rstate, surfacePoint_t &sp) const;
		bool getSPfromHit(const diffRay_t &ray, int tri_idx, surfacePoint_t &sp) const;
		
		background_t *background;
		bool trShad;
		bool finalGather, showMap;
		bool prepass;
		unsigned int nPhotons;
		unsigned int nCausPhotons;
		int sDepth, rDepth, maxBounces, nSearch, nCausSearch;
		int nPaths, gatherBounces;
		float dsRadius; //!< diffuse search radius
		float cRadius; //!< caustic search radius
		float lookupRad; //!< square radius to lookup radiance photons, as infinity is no such good idea ;)
		float gatherDist; //!< minimum distance to terminate path tracing (unless gatherBounces is reached)
		photonMap_t diffuseMap, causticMap;
		photonMap_t radianceMap; //!< this map contains precomputed radiance "photons", not incoming photon marks
		pdf1D_t *lightPowerD;
		std::vector<light_t*> lights;
		BSDF_t allBSDFIntersect;
		friend class prepassWorker_t;
		bool hasBGLight;

		friend class RayStorer;
		DiskVectorType disks;
		PHierarchy pHierarchy;
		std::vector<const triangle_t*> prims;

		CLPlatform platform;
		CLDevice device;
		CLContext *context;
		CLCommandQueue *queue;

		std::string cl_build_options;

		CLBuffer *d_int_nodes, *d_leaves, *d_tris;
		CLProgram *program;
};

__END_YAFRAY

#endif // Y_PHOTONINTEGR_H
