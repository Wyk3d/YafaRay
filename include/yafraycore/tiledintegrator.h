
#ifndef Y_TILEDINTEGRATOR_H
#define Y_TILEDINTEGRATOR_H

#include <core_api/integrator.h>
#include <core_api/imagesplitter.h>
#include <core_api/material.h>

__BEGIN_YAFRAY

class YAFRAYCORE_EXPORT tiledIntegrator_t: public surfaceIntegrator_t
{
	public:
		/*! Rendering prepasses to precalc suff in case needed */
		virtual void preRender(); //!< Called before the render starts and after the minDepth and maxDepth are calculated
		virtual void prePass(int samples, int offset, bool adaptive); //!< Called before the proper rendering of all the tiles starts
		virtual void preTile(renderArea_t &a, int n_samples, int offset, bool adaptive, int threadID); //!< Called brfore each tile is rendered
		
		/*! do whatever is required to render the image; default implementation renders image in passes
		dividing each pass into tiles for multithreading. */
		virtual bool render(imageFilm_t *imageFilm);
		/*! render a pass; only required by the default implementation of render() */
		virtual bool renderPass(int samples, int offset, bool adaptive);
		/*! render a tile; only required by default implementation of render() */
		virtual bool renderTile(renderArea_t &a, int n_samples, int offset, bool adaptive, int threadID);
		
		virtual void recursiveRaytrace(renderState_t &state, diffRay_t &ray, int rDepth, BSDF_t bsdfs, surfacePoint_t &sp, vector3d_t &wo, color_t &col, float &alpha) const;
		virtual void precalcDepths();

		friend class PrimaryRayGenerator;
		friend class RenderTile_PrimaryRayGenerator;

		class YAFRAYCORE_EXPORT PrimaryRayGenerator
		{
			protected:
				tiledIntegrator_t *integrator;
				renderArea_t &area;
				int n_samples, offset;
				scene_t *scene;
				const camera_t* camera;
				renderState_t rstate;
			public:
				PrimaryRayGenerator(
					renderArea_t &a, int n_samples, int offset, 
					tiledIntegrator_t *integrator, random_t &prng
				);
				virtual bool skipPixel(int i, int j) = 0;
				virtual void onCameraRayMissed(int i, int j, int dx, int dy) = 0;
				virtual void rays(diffRay_t &c_ray, ray_t &d_ray, int i, int j, int dx, int dy, float wt) = 0;
				void genRays();
		};

		class YAFRAYCORE_EXPORT RenderTile_PrimaryRayGenerator : public tiledIntegrator_t::PrimaryRayGenerator
		{
			private:
				bool adaptive, threadID;
				bool do_depth;
				imageFilm_t *imageFilm;
			public:
				RenderTile_PrimaryRayGenerator(
					renderArea_t &a, int n_samples, int offset,
					bool adaptive, int threadID, 
					tiledIntegrator_t *integrator, random_t &prng
				);
				bool skipPixel(int i, int j);
				void onCameraRayMissed(int i, int j, int dx, int dy);
				void rays(diffRay_t &c_ray, ray_t &d_ray, int i, int j, int dx, int dy, float wt);
		};
	
	protected:
		int AA_samples, AA_passes, AA_inc_samples;
		float iAA_passes; //!< Inverse of AA_passes used for depth map
		float AA_threshold;
		imageFilm_t *imageFilm;
		float maxDepth; //!< Inverse of max depth from camera within the scene boundaries
		float minDepth; //!< Distance between camera and the closest object on the scene
};

#ifdef USING_THREADS

struct threadControl_t
{
	threadControl_t() : finishedThreads(0) {}
	yafthreads::conditionVar_t countCV; //!< condition variable to signal main thread
	std::vector<renderArea_t> areas; //!< area to be output to e.g. blender, if any
	volatile int finishedThreads; //!< number of finished threads, lock countCV when increasing/reading!
};

#endif

__END_YAFRAY

#endif // Y_TILEDINTEGRATOR_H
