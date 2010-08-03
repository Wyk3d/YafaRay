
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
			int mat_type;	// disk material type
			PHLeaf() {}
			PHLeaf(const point3d_t &c, const vector3d_t &n, int mat_type)
				: c(c), n(n), mat_type(mat_type) {}
		};

		struct PHInternalNode
		{
			point3d_t c;	// bounding sphere center
			float r;		// bounding sphere radius
			PHInternalNode() {}
			PHInternalNode(const point3d_t &c, float r) 
				: c(c), r(r) {}
		};

		struct Disk
		{
			point3d_t c;
			const triangle_t *t;
			Disk() {}
			Disk(point3d_t c, const triangle_t *t)
				: c(c), t(t) {}
		};

		typedef std::vector<Disk> DiskVectorType;
		typedef std::vector<vector3d_t> NormalVectorType;

		void build_disk_hierarchy(std::vector<PHInternalNode> &int_nodes, std::vector<PHLeaf> &leaves, int node_poz, DiskVectorType &v, int s, int e, float leaf_radius);
		void generate_points(NormalVectorType &normals, DiskVectorType &disks, scene_t *scene, float r);

		bool renderTile(renderArea_t &a, int n_samples, int offset, bool adaptive, int threadID);

	protected:
		color_t finalGathering(renderState_t &state, const surfacePoint_t &sp, const vector3d_t &wo) const;
		color_t estimateOneDirect(renderState_t &state, const surfacePoint_t &sp, vector3d_t wo, const std::vector<light_t *>  &lights, int d1, int n)const;

		CLDevice getOpenCLDevice();
		CLPlatform getOpenCLPlatform();
		
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
		std::vector<diffRay_t> c_rays;

		CLPlatform platform;
		CLDevice device;
		CLContext *context;
		CLCommandQueue *queue;
};

__END_YAFRAY

#endif // Y_PHOTONINTEGR_H
