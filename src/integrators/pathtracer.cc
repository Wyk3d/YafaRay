/****************************************************************************
 * 			pathtracer.cc: a rather simple QMC path integrator
 *      This is part of the yafray package
 *      Copyright (C) 2006  Mathias Wein
 *
 *      This library is free software; you can redistribute it and/or
 *      modify it under the terms of the GNU Lesser General Public
 *      License as published by the Free Software Foundation; either
 *      version 2.1 of the License, or (at your option) any later version.
 *
 *      This library is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *      Lesser General Public License for more details.
 *
 *      You should have received a copy of the GNU Lesser General Public
 *      License along with this library; if not, write to the Free Software
 *      Foundation,Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
 
//#include <mcqmc.h>
#include <yafray_config.h>
#include <core_api/environment.h>
#include <core_api/material.h>
#include <core_api/volume.h>
#include <yafraycore/tiledintegrator.h>
#include <core_api/background.h>
#include <core_api/light.h>
#include <utilities/mcqmc.h>
#include <yafraycore/scr_halton.h>
#include <yafraycore/photon.h>
#include <yafraycore/spectrum.h>
#include <integrators/integr_utils.h>
#include <sstream>
#include <yafraycore/triangle.h>
#include <yafraycore/meshtypes.h>
#include <limits>

__BEGIN_YAFRAY

class YAFRAYPLUGIN_EXPORT pathIntegrator_t: public tiledIntegrator_t
{
	public:
		pathIntegrator_t(bool transpShad=false, int shadowDepth=4);
		virtual bool preprocess();
		virtual colorA_t integrate(renderState_t &state, diffRay_t &ray/*, sampler_t &sam*/) const;
		static integrator_t* factory(paraMap_t &params, renderEnvironment_t &render);
		enum { NONE, PATH, PHOTON, BOTH };
	protected:
		color_t estimateOneDirect(renderState_t &state, const surfacePoint_t &sp, vector3d_t wo, const std::vector<light_t *>  &lights, int d1, int n) const;
		background_t *background;
		bool trShad;
		bool traceCaustics; //!< use path tracing for caustics (determined by causticType)
		bool no_recursive;
		int sDepth, rDepth, bounces, nPaths;
		float invNPaths;
		int causticType, nPhotons, cDepth, nSearch;
		PFLOAT cRadius; //!< radius to search for caustic photons
		std::vector<light_t*> lights;
		photonMap_t causticMap;
};

pathIntegrator_t::pathIntegrator_t(bool transpShad, int shadowDepth):
	trShad(transpShad), sDepth(shadowDepth), causticType(PATH)
{
	type = SURFACE;
	rDepth = 6;
	bounces = 5;
	nPaths = 64;
	invNPaths = 1.f/64.f;
	no_recursive = false;
	integratorName = "PathTracer";
	integratorShortName = "PT";
}

bool pathIntegrator_t::preprocess()
{
	std::stringstream set;
	background = scene->getBackground();
	lights = scene->lights;
	
	if(trShad)
	{
		set << "ShadowDepth: [" << sDepth << "]";
	}
	if(!set.str().empty()) set << "+";
	set << "RayDepth: [" << rDepth << "]";


	bool success = true;
	traceCaustics = false;
	
	if(causticType == PHOTON || causticType == BOTH)
	{
		progressBar_t *pb;
		if(intpb) pb = intpb;
		else pb = new ConsoleProgressBar_t(80);
		success = createCausticMap(*scene, lights, causticMap, cDepth, nPhotons, pb, integratorName);
		if(!intpb) delete pb;
	}

	if(causticType == BOTH || causticType == PATH) traceCaustics = true;
	
	if(causticType == PATH)
	{
		if(!set.str().empty()) set << "+";
		set << "Caustics: Path";
	}
	else if(causticType == PHOTON)
	{
		if(!set.str().empty()) set << "+";
		set << "Caustics: Photon(" << nPhotons << ")";
	}
	else if(causticType == BOTH)
	{
		if(!set.str().empty()) set << "+";
		set << "Caustics: Path+Photon(" << nPhotons << ")";
	}
	
	settings = set.str();

	scene_t *new_scene = new scene_t;

	int max_to_generate = 32000;
	struct ptd {
		point3d_t p;
		bool t;
	};
	std::vector<ptd> points;
	points.resize(max_to_generate);

	struct disk
	{
		point3d_t p;
		int n_idx;
		float r;
		disk(const point3d_t &p, int n_idx, float r) : p(p), n_idx(n_idx), r(r) {}
	};

	std::vector<vector3d_t> normals;
	std::vector<disk> disks;

	int num_tri_points = 20;
	std::vector<point3d_t> tri_points;
	tri_points.resize(num_tri_points);

	scene_t::objDataArray_t &meshes = scene->getMeshes();
	for(scene_t::objDataArray_t::iterator itr = meshes.begin(); itr != meshes.end(); ++itr) {
		triangleObject_t *obj = itr->second.obj;
		int nt = obj->numPrimitives();
		const triangle_t **tris = new const triangle_t*[nt];
		itr->second.obj->getPrimitives(tris);
		for(int i = 0; i < nt; ++i) {
			const triangle_t *t = tris[i];
			point3d_t a, b, c;
			t->getVertices(a, b, c);
			normals.push_back(t->getNormal());
			
			vector3d_t ab = b - a, ac = c - a;
			float area = 0.5 * (ab ^ ac).length();

			// TODO: compute these from the area
			int nr_to_generate = 50;
			int nr_to_keep = 10;
			
			
			for(int j = 0; j < nr_to_generate; j++)
			{
				float u = ourRandom();
				float v = ourRandom();
				points[i].p = a + u * ab + v * ac; 
				points[i].t = false;
			}

			for(int j = 0; j < nr_to_keep; j++)
			{
				int ppoz = 0;
				float mind = std::numeric_limits<float>::max();

				for(int k = 0; k < nr_to_generate; k++)
				{
					if(!points[k].t)
					{
						// TODO: avoid computing the distances more than once
						for(int l = 0; l < j; l++) {
							float d = (points[k].p - disks[l].p).lengthSqr();
							if(d < mind) {
								mind = d;
								ppoz = k;
							}
						}
					}
				}

				point3d_t &p = points[ppoz].p;

				// TODO: determine this from the area
				float r = 10;
				disks.push_back(disk(p, i, r));

				vector3d_t &n = normals[i];
				for(int j = 0; j < num_tri_points; j++) {
					float beta_j = j * M_2PI / num_tri_points;
					float cos_beta_j = cos(beta_j);
					float sin_beta_j = sin(beta_j);

					float saj_p = (n.x * cos_beta_j + n.y * sin_beta_j);
					float nz_sq = n.z * n.z;
					float sinsq_alfa_j = nz_sq / ( saj_p * saj_p + nz_sq);
					float sin_alfa_j = sqrt(sinsq_alfa_j);
					float cos_alfa_j = sqrt(1 - sinsq_alfa_j);

					float r_sin_alfa_j = r * sin_alfa_j;
					
					tri_points[j] = point3d_t(
						p.x + r_sin_alfa_j * cos_beta_j,
						p.y + r_sin_alfa_j * sin_beta_j,
						p.z + r * cos_alfa_j
					);
				}
			}
		}
	}

	return success;
}

inline color_t pathIntegrator_t::estimateOneDirect(renderState_t &state, const surfacePoint_t &sp, vector3d_t wo,
												   const std::vector<light_t *>  &lights, int d1, int n)const
{
	color_t lcol(0.0), scol, col(0.0);
	ray_t lightRay;
	float lightPdf;
	bool shadowed;
	const material_t *oneMat = sp.material;
	lightRay.from = sp.P;
	int nLightsI = lights.size();
	if(nLightsI == 0) return color_t(0.f);
	float nLights = float(nLightsI);
	float s1;

	s1 = scrHalton(d1, n) * nLights;
	int lnum = std::min((int)(s1), nLightsI-1);
	const light_t *light = lights[lnum];
	s1 = s1 - (float)lnum;

	// handle lights with delta distribution, e.g. point and directional lights
	if( light->diracLight() )
	{
		if( light->illuminate(sp, lcol, lightRay) )
		{
			// ...shadowed...
			lightRay.tmin = YAF_SHADOW_BIAS; // < better add some _smart_ self-bias value...this is bad.
			shadowed = (trShad) ? scene->isShadowed(state, lightRay, sDepth, scol) : scene->isShadowed(state, lightRay);
			if(!shadowed)
			{
				if(trShad) lcol *= scol;
				color_t surfCol = oneMat->eval(state, sp, wo, lightRay.dir, BSDF_ALL);
				col = surfCol * lcol * std::fabs(sp.N*lightRay.dir);
			}
		}
	}
	else // area light and suchlike
	{
		// ...get sample val...
		lSample_t ls;
		ls.s1 = s1;
		ls.s2 = scrHalton(d1+1, n);

		bool canIntersect=light->canIntersect();
		
		if( light->illumSample (sp, ls, lightRay) )
		{
			// ...shadowed...
			lightRay.tmin = YAF_SHADOW_BIAS; // < better add some _smart_ self-bias value...this is bad.
			shadowed = (trShad) ? scene->isShadowed(state, lightRay, sDepth, scol) : scene->isShadowed(state, lightRay);
			if(!shadowed && ls.pdf > 1e-6f)
			{
				if(trShad) ls.col *= scol;

				color_t surfCol = oneMat->eval(state, sp, wo, lightRay.dir, BSDF_ALL);
				
				if( canIntersect )
				{
					float mPdf = oneMat->pdf(state, sp, wo, lightRay.dir, BSDF_GLOSSY | BSDF_DIFFUSE | BSDF_DISPERSIVE | BSDF_REFLECT | BSDF_TRANSMIT);
					float l2 = ls.pdf * ls.pdf;
					float m2 = mPdf * mPdf + 0.1f;
					float w = l2 / (l2 + m2);
					//test! limit lightPdf...
					if(ls.pdf < 1e-5f) ls.pdf = 1e-5f;
					col = surfCol * ls.col * std::fabs(sp.N*lightRay.dir) * w / ls.pdf;
				}
				else
				{
					//test! limit lightPdf...
					if(ls.pdf < 1e-5f) ls.pdf = 1e-5f;
					col = surfCol * ls.col * std::fabs(sp.N*lightRay.dir) / ls.pdf;
				}
			}
		}
		if(canIntersect) // sample from BSDF to complete MIS
		{
			ray_t bRay;
			bRay.tmin = MIN_RAYDIST;
			bRay.from = sp.P;

			sample_t s(ls.s1, ls.s2, BSDF_GLOSSY | BSDF_DIFFUSE | BSDF_DISPERSIVE | BSDF_REFLECT | BSDF_TRANSMIT);
			color_t surfCol = oneMat->sample(state, sp, wo, bRay.dir, s);

			if( s.pdf>1e-6f && light->intersect(bRay, bRay.tmax, lcol, lightPdf) )
			{
				shadowed = (trShad) ? scene->isShadowed(state, bRay, sDepth, scol) : scene->isShadowed(state, bRay);
				if(!shadowed)
				{
					if(trShad) lcol *= scol;
					float lPdf = 1.f/lightPdf;
					float l2 = lPdf * lPdf;
					float m2 = s.pdf * s.pdf + 0.1f;
					float w = m2 / (l2 + m2);
					float cos2 = std::fabs(sp.N*bRay.dir);
					col += surfCol * lcol * cos2 * w / s.pdf;
				}
			}
		}
	}
	return col*nLights;
}

colorA_t pathIntegrator_t::integrate(renderState_t &state, diffRay_t &ray/*, sampler_t &sam*/) const
{
	static int calls=0;
	++calls;
	color_t col(0.0);
	CFLOAT alpha=0.0;
	surfacePoint_t sp;
	void *o_udat = state.userdata;
	//shoot ray into scene
	if(scene->intersect(ray, sp))
	{
		// if camera ray initialize sampling offset:
		if(state.raylevel == 0)
		{
			state.includeLights = true;
			//...
		}
		unsigned char userdata[USER_DATA_SIZE+7];
		userdata[0] = 0;
		state.userdata = (void *)( &userdata[7] - ( ((size_t)&userdata[7])&7 ) ); // pad userdata to 8 bytes
		BSDF_t bsdfs;
		const material_t *material = sp.material;
		material->initBSDF(state, sp, bsdfs);
		vector3d_t wo = -ray.dir;
		const volumeHandler_t *vol;
		color_t vcol(0.f);

		// contribution of light emitting surfaces		
		if(bsdfs & BSDF_EMIT) col += material->emit(state, sp, wo);
		
		if(bsdfs & BSDF_DIFFUSE) col += estimateDirect_PH(state, sp, lights, scene, wo, trShad, sDepth);
		
		if((bsdfs & BSDF_DIFFUSE) && (causticType == PHOTON || causticType == BOTH)) col += estimatePhotons(state, sp, causticMap, wo, nSearch, cRadius);
				
		// path tracing:
		// the first path segment is "unrolled" from the loop because for the spot the camera hit
		// we do things slightly differently (e.g. may not sample specular, need not to init BSDF anymore,
		// have more efficient ways to compute samples...)
		
		bool was_chromatic = state.chromatic;
		BSDF_t path_flags = no_recursive ? BSDF_ALL : (BSDF_DIFFUSE);
		
		if(bsdfs & path_flags)
		{
			color_t pathCol(0.0), wl_col;
			path_flags |= (BSDF_DIFFUSE | BSDF_REFLECT | BSDF_TRANSMIT);
			int nSamples = std::max(1, nPaths/state.rayDivision);
			for(int i=0; i<nSamples; ++i)
			{
				void *first_udat = state.userdata;
				unsigned char userdata[USER_DATA_SIZE+7];
				void *n_udat = (void *)( &userdata[7] - ( ((size_t)&userdata[7])&7 ) ); // pad userdata to 8 bytes
				unsigned int offs = nPaths * state.pixelSample + state.samplingOffs + i; // some redunancy here...
				color_t throughput( 1.0 );
				color_t lcol, scol;
				surfacePoint_t sp1=sp, sp2;
				surfacePoint_t *hit=&sp1, *hit2=&sp2;
				vector3d_t pwo = wo;
				ray_t pRay;

				state.chromatic = was_chromatic;
				if(was_chromatic) state.wavelength = RI_S(offs);
				//this mat already is initialized, just sample (diffuse...non-specular?)
				float s1 = RI_vdC(offs);
				float s2 = scrHalton(2, offs);
				if(state.rayDivision > 1)
				{
					s1 = addMod1(s1, state.dc1);
					s2 = addMod1(s2, state.dc2);
				}
				// do proper sampling now...
				sample_t s(s1, s2, path_flags);
				scol = material->sample(state, sp, pwo, pRay.dir, s);
				
				if(s.pdf <= 1e-6f) continue;
				
				scol *= (std::fabs(pRay.dir*sp.N)/s.pdf);
				throughput = scol;
				state.includeLights = false;

				pRay.tmin = MIN_RAYDIST;
				pRay.tmax = -1.0;
				pRay.from = sp.P;
				
				if(!scene->intersect(pRay, *hit)) continue; //hit background

				state.userdata = n_udat;
				const material_t *p_mat = hit->material;
				BSDF_t matBSDFs;
				p_mat->initBSDF(state, *hit, matBSDFs);
				pwo = -pRay.dir;
				lcol = estimateOneDirect(state, *hit, pwo, lights, 3, offs);
				if(matBSDFs & BSDF_EMIT) lcol += p_mat->emit(state, *hit, pwo);

				pathCol += lcol*throughput;
				
				bool caustic = false;
				
				for(int depth=1; depth<bounces; ++depth)
				{
					int d4 = 4*depth;
					s.s1 = scrHalton(d4+3, offs); //ourRandom();//
					s.s2 = scrHalton(d4+4, offs); //ourRandom();//

					if(state.rayDivision > 1)
					{
						s1 = addMod1(s1, state.dc1);
						s2 = addMod1(s2, state.dc2);
					}

					s.flags = BSDF_ALL;
					
					scol = p_mat->sample(state, *hit, pwo, pRay.dir, s);
					if(s.pdf <= 1.0e-6f) break;

					scol *= (std::fabs(pRay.dir*hit->N)/s.pdf);
					
					if(scol.isBlack()) break;
					
					throughput *= scol;
					caustic = traceCaustics && (s.sampledFlags & (BSDF_SPECULAR | BSDF_GLOSSY | BSDF_FILTER));
					state.includeLights = caustic;

					pRay.tmin = MIN_RAYDIST;
					pRay.tmax = -1.0;
					pRay.from = hit->P;

					if(!scene->intersect(pRay, *hit2)) //hit background
					{
						if((caustic && background))
						{
							pathCol += throughput * (*background)(pRay, state);
						}
						break;
					}
					
					std::swap(hit, hit2);
					p_mat = hit->material;
					p_mat->initBSDF(state, *hit, matBSDFs);
					pwo = -pRay.dir;

					if(matBSDFs & BSDF_DIFFUSE) lcol = estimateOneDirect(state, *hit, pwo, lights, d4+3, offs);
					else lcol = color_t(0.f);

					if((matBSDFs & BSDF_VOLUMETRIC) && (vol=p_mat->getVolumeHandler(hit->N * pwo < 0)))
					{
						if(vol->transmittance(state, pRay, vcol)) throughput *= vcol;
					}
					
					if (matBSDFs & BSDF_EMIT && caustic) lcol += p_mat->emit(state, *hit, pwo);
					
					pathCol += lcol*throughput;
				}
				state.userdata = first_udat;
				
			}
			col += pathCol / nSamples;
		}
		//reset chromatic state:
		state.chromatic = was_chromatic;

		recursiveRaytrace(state, ray, rDepth, bsdfs, sp, wo, col, alpha);

		CFLOAT m_alpha = material->getAlpha(state, sp, wo);
		alpha = m_alpha + (1.f-m_alpha)*alpha;
	}
	else //nothing hit, return background
	{
		if(background)
		{
			col += (*background)(ray, state, false);
		}
	}
	//dbg = false;
	state.userdata = o_udat;
	return colorA_t(col, alpha);
}

integrator_t* pathIntegrator_t::factory(paraMap_t &params, renderEnvironment_t &render)
{
	bool transpShad=false, noRec=false;
	int shadowDepth = 5;
	int path_samples = 32;
	int bounces = 3;
	int raydepth = 5;
	const std::string *cMethod=0;
	
	params.getParam("raydepth", raydepth);
	params.getParam("transpShad", transpShad);
	params.getParam("shadowDepth", shadowDepth);
	params.getParam("path_samples", path_samples);
	params.getParam("bounces", bounces);
	params.getParam("no_recursive", noRec);
	
	pathIntegrator_t* inte = new pathIntegrator_t(transpShad, shadowDepth);
	if(params.getParam("caustic_type", cMethod))
	{
		bool usePhotons=false;
		if(*cMethod == "photon"){ inte->causticType = PHOTON; usePhotons=true; }
		else if(*cMethod == "both"){ inte->causticType = BOTH; usePhotons=true; }
		else if(*cMethod == "none") inte->causticType = NONE;
		if(usePhotons)
		{
			double cRad = 0.25;
			int cDepth=10, search=100, photons=500000;
			params.getParam("photons", photons);
			params.getParam("caustic_mix", search);
			params.getParam("caustic_depth", cDepth);
			params.getParam("caustic_radius", cRad);
			inte->nPhotons = photons;
			inte->nSearch = search;
			inte->cDepth = cDepth;
			inte->cRadius = cRad;
		}
	}
	inte->rDepth = raydepth;
	inte->nPaths = path_samples;
	inte->invNPaths = 1.f / (float)path_samples;
	inte->bounces = bounces;
	inte->no_recursive = noRec;
	return inte;
}

extern "C"
{

	YAFRAYPLUGIN_EXPORT void registerPlugin(renderEnvironment_t &render)
	{
		render.registerFactory("pathtracing",pathIntegrator_t::factory);
	}

}

__END_YAFRAY
