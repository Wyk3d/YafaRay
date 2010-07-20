
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


		struct disk
		{
			point3d_t c;	// center of the disk
			int n_idx;		// normal index
			float r;		// disk radius
			float br;		// bounding sphere radius in the hierarchy
			point3d_t bc;   // center of the bounding sphere
			const material_t *mat;	
			disk(const material_t *mat, const point3d_t &c, int n_idx, float r) : mat(mat), c(c), n_idx(n_idx), r(r) {}
		};

		typedef std::vector<disk> DiskVectorType;
		typedef std::vector<vector3d_t> NormalVectorType;

		void build_disk_hierarchy(DiskVectorType &v, int s, int e);
		void generate_points(NormalVectorType &normals, DiskVectorType disks, scene_t *scene);

	protected:
		color_t finalGathering(renderState_t &state, const surfacePoint_t &sp, const vector3d_t &wo) const;
		color_t estimateOneDirect(renderState_t &state, const surfacePoint_t &sp, vector3d_t wo, const std::vector<light_t *>  &lights, int d1, int n)const;
		
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
};

__END_YAFRAY

#endif // Y_PHOTONINTEGR_H
