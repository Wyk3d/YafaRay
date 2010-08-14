/****************************************************************************
 *      photonintegr.cc: integrator for photon mapping and final gather
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

#include <integrators/photonintegrGPU.h>
#include <yafraycore/triangle.h>
#include <yafraycore/meshtypes.h>
#include <yafraycore/best_candidate.h>
#include <sstream>
#include <limits>
#include <opencl_wrapper/cl_util.h>
#include <yafraycore/kdtree.h>

__BEGIN_YAFRAY

photonIntegratorGPU_t::photonIntegratorGPU_t(unsigned int dPhotons, unsigned int cPhotons, bool transpShad, int shadowDepth, float dsRad, float cRad):
	trShad(transpShad), finalGather(true), nPhotons(dPhotons), nCausPhotons(cPhotons), sDepth(shadowDepth), dsRadius(dsRad), cRadius(cRad),
	platform(getOpenCLPlatform()), device(getOpenCLDevice())
{
	type = SURFACE;
	rDepth = 6;
	maxBounces = 5;
	allBSDFIntersect = BSDF_GLOSSY | BSDF_DIFFUSE | BSDF_DISPERSIVE | BSDF_REFLECT | BSDF_TRANSMIT;
	intpb = 0;
	integratorName = "PhotonMapGPU";
	integratorShortName = "PMG";
	hasBGLight = false;

	CLError err;
	context = platform.createContext(device, &err);
	checkErr(err || context == NULL, "failed to get a context for the chosen device\n");

	queue = context->createCommandQueue(device, &err);
	checkErr(err || queue == NULL, "failed to create command queue");

	cl_uint vendor_id = device.getVendorId(&err);
	checkErr(err, "failed to get device vendor");

	if(vendor_id == CL_VENDOR_NVIDIA) {
		cl_build_options += " -cl-nv-verbose";
	}

	d_int_nodes = NULL;
	d_leaves = NULL;
	d_tris = NULL;
	program = NULL;
}

struct preGatherData_t
{
	preGatherData_t(photonMap_t *dm): diffuseMap(dm), fetched(0) {}
	photonMap_t *diffuseMap;
	
	std::vector<radData_t> rad_points;
	std::vector<photon_t> radianceVec;
	progressBar_t *pbar;
	volatile int fetched;
	yafthreads::mutex_t mutex;
};

class preGatherWorker_t: public yafthreads::thread_t
{
	public:
		preGatherWorker_t(preGatherData_t *dat, float dsRad, int search):
			gdata(dat), dsRadius_2(dsRad*dsRad), nSearch(search) {};
		virtual void body();
	protected:
		preGatherData_t *gdata;
		float dsRadius_2;
		int nSearch;
};

void preGatherWorker_t::body()
{
	unsigned int start, end, total;
	
	gdata->mutex.lock();
	start = gdata->fetched;
	total = gdata->rad_points.size();
	end = gdata->fetched = std::min(total, start + 32);
	gdata->mutex.unlock();
	
	foundPhoton_t *gathered = new foundPhoton_t[nSearch];

	float radius = 0.f;
	float iScale = 1.f / ((float)gdata->diffuseMap->nPaths() * M_PI);
	float scale = 0.f;
	
	while(start < total)
	{
		for(unsigned int n=start; n<end; ++n)
		{
			radius = dsRadius_2;//actually the square radius...
			int nGathered = gdata->diffuseMap->gather(gdata->rad_points[n].pos, gathered, nSearch, radius);
			
			vector3d_t rnorm = gdata->rad_points[n].normal;
			
			color_t sum(0.0);
			
			if(nGathered > 0)
			{
				scale = iScale / radius;
				
				for(int i=0; i<nGathered; ++i)
				{
					vector3d_t pdir = gathered[i].photon->direction();
					
					if( rnorm * pdir > 0.f ) sum += gdata->rad_points[n].refl * scale * gathered[i].photon->color();
					else sum += gdata->rad_points[n].transm * scale * gathered[i].photon->color();
				}
			}
			
			gdata->radianceVec[n] = photon_t(rnorm, gdata->rad_points[n].pos, sum);
		}
		gdata->mutex.lock();
		start = gdata->fetched;
		end = gdata->fetched = std::min(total, start + 32);
		gdata->pbar->update(32);
		gdata->mutex.unlock();
	}
	delete[] gathered;
}

photonIntegratorGPU_t::~photonIntegratorGPU_t()
{
	CLError err;

	if(program) {
		program->free(&err);
		checkErr("failed to free program");
	}
	if(d_int_nodes) {
		d_int_nodes->free(&err);
		checkErr(err, "failed to free d_int_nodes");
	}
	if(d_leaves) {
		d_leaves->free(&err);
		checkErr(err, "failed to free d_leaves");
	}
	if(d_tris) {
		d_tris->free(&err);
		checkErr(err, "failed to free tris");
	}

	queue->free(&err);
	checkErr(err, "failed to free queue");
	context->free(&err);
	checkErr(err, "failed to free context");
}

inline color_t photonIntegratorGPU_t::estimateOneDirect(renderState_t &state, const surfacePoint_t &sp, vector3d_t wo, const std::vector<light_t *>  &lights, int d1, int n)const
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
	
	int lnum = std::min((int)(s1), nLightsI - 1);
	
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
		
		bool canIntersect = light->canIntersect();
		
		if( light->illumSample (sp, ls, lightRay) )
		{
			// ...shadowed...
			lightRay.tmin = YAF_SHADOW_BIAS; // < better add some _smart_ self-bias value...this is bad.
			shadowed = (trShad) ? scene->isShadowed(state, lightRay, sDepth, scol) : scene->isShadowed(state, lightRay);
			if(!shadowed && ls.pdf > 1e-6f)
			{
				if(trShad) ls.col *= scol;
				color_t surfCol = oneMat->eval(state, sp, wo, lightRay.dir, BSDF_ALL);
				
				if( canIntersect ) // bound samples and compensate by sampling from BSDF later
				{
					float mPdf = oneMat->pdf(state, sp, wo, lightRay.dir, allBSDFIntersect);
					if(mPdf > 1e-6f)
					{
						float l2 = ls.pdf * ls.pdf;
						float m2 = mPdf * mPdf;
						float w = l2 / (l2 + m2);
						col = surfCol * ls.col * std::fabs(sp.N*lightRay.dir) * w / ls.pdf;
					}
					else col = surfCol * ls.col * std::fabs(sp.N*lightRay.dir) / ls.pdf;
				}
				else
				{
					col = surfCol * ls.col * std::fabs(sp.N*lightRay.dir) / ls.pdf;
				}
			}
		}
		if(canIntersect) // sample from BSDF to complete MIS
		{
			ray_t bRay;
			bRay.tmin = YAF_SHADOW_BIAS;
			bRay.from = sp.P;
			sample_t s(ls.s1, ls.s2, allBSDFIntersect);
			color_t surfCol = oneMat->sample(state, sp, wo, bRay.dir, s);
			if( s.pdf>1e-6f && light->intersect(bRay, bRay.tmax, lcol, lightPdf) )
			{
				shadowed = (trShad) ? scene->isShadowed(state, bRay, sDepth, scol) : scene->isShadowed(state, bRay);
				if(!shadowed)
				{
					if(trShad) lcol *= scol;
					color_t transmitCol = scene->volIntegrator->transmittance(state, lightRay);
					ls.col *= transmitCol;
					float cos2 = std::fabs(sp.N*bRay.dir);
					if(lightPdf > 1e-6f)
					{
						float lPdf = 1.f/lightPdf;
						float l2 = lPdf * lPdf;
						float m2 = s.pdf * s.pdf;
						float w = m2 / (l2 + m2);
						col += surfCol * lcol * cos2 * w / s.pdf;
					}
					else col += surfCol * lcol * cos2 / s.pdf;
				}
			}
		}
	}
	return col*nLights;
}

bool photonIntegratorGPU_t::preprocess()
{
	std::stringstream set;
	gTimer.addEvent("prepass");
	gTimer.start("prepass");

	Y_INFO << integratorName << ": Starting preprocess..." << yendl;

	if(trShad)
	{
		set << "ShadowDepth [" << sDepth << "]";
	}
	if(!set.str().empty()) set << "+";
	set << "RayDepth [" << rDepth << "]";

	diffuseMap.clear();
	causticMap.clear();
	background = scene->getBackground();
	lights = scene->lights;
	std::vector<light_t*> tmplights;

	if(!set.str().empty()) set << "+";
	
	set << "DiffPhotons [" << nPhotons << "]+CausPhotons[" << nCausPhotons << "]";
	
	if(finalGather)
	{
		set << "+FG[" << nPaths << ", " << gatherBounces << "]";
	}
	
	settings = set.str();
	
	ray_t ray;
	float lightNumPdf, lightPdf, s1, s2, s3, s4, s5, s6, s7, sL;
	int numCLights = 0;
	int numDLights = 0;
	float fNumLights = 0.f;
	float *energies = NULL;
	color_t pcol;

	tmplights.clear();

	for(int i=0;i<(int)lights.size();++i)
	{
		if(lights[i]->shootsDiffuseP())
		{
			numDLights++;
			tmplights.push_back(lights[i]);
		}
	}
	
	fNumLights = (float)numDLights;
	energies = new float[numDLights];

	for(int i=0;i<numDLights;++i) energies[i] = tmplights[i]->totalEnergy().energy();

	lightPowerD = new pdf1D_t(energies, numDLights);
	
	Y_INFO << integratorName << ": Light(s) photon color testing for diffuse map:" << yendl;
	for(int i=0;i<numDLights;++i)
	{
		pcol = tmplights[i]->emitPhoton(.5, .5, .5, .5, ray, lightPdf);
		lightNumPdf = lightPowerD->func[i] * lightPowerD->invIntegral;
		pcol *= fNumLights*lightPdf/lightNumPdf; //remember that lightPdf is the inverse of the pdf, hence *=...
		Y_INFO << integratorName << ": Light [" << i+1 << "] Photon col:" << pcol << " | lnpdf: " << lightNumPdf << yendl;
	}
	
	delete[] energies;
	
	//shoot photons
	bool done=false;
	unsigned int curr=0;
	// for radiance map:
	preGatherData_t pgdat(&diffuseMap);
	
	surfacePoint_t sp;
	renderState_t state;
	unsigned char userdata[USER_DATA_SIZE+7];
	state.userdata = (void *)( &userdata[7] - ( ((size_t)&userdata[7])&7 ) ); // pad userdata to 8 bytes
	state.cam = scene->getCamera();
	progressBar_t *pb;
	int pbStep;
	if(intpb) pb = intpb;
	else pb = new ConsoleProgressBar_t(80);

	pHierarchy.leaf_radius = 0.5;
	disks.clear();
	generate_points(disks, prims, pHierarchy, scene);
	int node_size = std::max(sizeof(PHInternalNode), sizeof(PHLeaf));
	// should be max leaves = 2^h > nr disks ..
	int h = (int)(ceil(log((double)disks.size()) / log(2.0)));
	int nr_leaves = (1<<h);
	int nr_internal_nodes = nr_leaves; // -1 (from the formula) +1 from position 0 not used
	
	pHierarchy.int_nodes.resize(nr_internal_nodes);
	pHierarchy.leaves.resize(nr_leaves);
	build_disk_hierarchy(pHierarchy, prims, 1, disks, 0, (int)disks.size());

	upload_hierarchy(pHierarchy);

	Y_INFO << integratorName << ": Building diffuse photon map..." << yendl;
	
	pb->init(128);
	pbStep = std::max(1U, nPhotons/128);
	pb->setTag("Building diffuse photon map...");
	//Pregather diffuse photons

	float invDiffPhotons = 1.f / (float)nPhotons;
	
	while(!done)
	{
		if(scene->getSignals() & Y_SIG_ABORT) {  pb->done(); if(!intpb) delete pb; return false; }

		s1 = RI_vdC(curr);
		s2 = scrHalton(2, curr);
		s3 = scrHalton(3, curr);
		s4 = scrHalton(4, curr);

		sL = float(curr) * invDiffPhotons;
		int lightNum = lightPowerD->DSample(sL, &lightNumPdf);
		if(lightNum >= numDLights)
		{
			Y_ERROR << integratorName << ": lightPDF sample error! " << sL << "/" << lightNum << "... stopping now." << yendl;
			delete lightPowerD;
			return false;
		}

		pcol = tmplights[lightNum]->emitPhoton(s1, s2, s3, s4, ray, lightPdf);
		ray.tmin = MIN_RAYDIST;
		ray.tmax = -1.0;
		pcol *= fNumLights*lightPdf/lightNumPdf; //remember that lightPdf is the inverse of th pdf, hence *=...
		
		if(pcol.isBlack())
		{
			++curr;
			done = (curr >= nPhotons);
			continue;
		}

		int nBounces=0;
		bool causticPhoton = false;
		bool directPhoton = true;
		const material_t *material = NULL;
		BSDF_t bsdfs;

		while( scene->intersect(ray, sp) )
		{
			if(isnan(pcol.R) || isnan(pcol.G) || isnan(pcol.B))
			{
				Y_WARNING << integratorName << ": NaN  on photon color for light" << lightNum + 1 << "." << yendl;
				continue;
			}
			
			color_t transm(1.f);
			color_t vcol(0.f);
			const volumeHandler_t* vol;
			
			if(material)
			{
				if((bsdfs&BSDF_VOLUMETRIC) && (vol=material->getVolumeHandler(sp.Ng * -ray.dir < 0)))
				{
					if(vol->transmittance(state, ray, vcol)) transm = vcol;
				}
			}
			
			vector3d_t wi = -ray.dir, wo;
			material = sp.material;
			material->initBSDF(state, sp, bsdfs);
			
			if(bsdfs & (BSDF_DIFFUSE))
			{
				//deposit photon on surface
				if(!causticPhoton)
				{
					photon_t np(wi, sp.P, pcol);
					diffuseMap.pushPhoton(np);
					diffuseMap.setNumPaths(curr);
				}
				// create entry for radiance photon:
				// don't forget to choose subset only, face normal forward; geometric vs. smooth normal?
				if(finalGather && ourRandom() < 0.125 && !causticPhoton )
				{
					vector3d_t N = FACE_FORWARD(sp.Ng, sp.N, wi);
					radData_t rd(sp.P, N);
					rd.refl = material->getReflectivity(state, sp, BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_REFLECT);
					rd.transm = material->getReflectivity(state, sp, BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_TRANSMIT);
					pgdat.rad_points.push_back(rd);
				}
			}
			// need to break in the middle otherwise we scatter the photon and then discard it => redundant
			if(nBounces == maxBounces) break;
			// scatter photon
			int d5 = 3*nBounces + 5;

			s5 = scrHalton(d5, curr);
			s6 = scrHalton(d5+1, curr);
			s7 = scrHalton(d5+2, curr);
			
			pSample_t sample(s5, s6, s7, BSDF_ALL, pcol, transm);

			bool scattered = material->scatterPhoton(state, sp, wi, wo, sample);
			if(!scattered) break; //photon was absorped.

			pcol = sample.color;

			causticPhoton = ((sample.sampledFlags & (BSDF_GLOSSY | BSDF_SPECULAR | BSDF_DISPERSIVE)) && directPhoton) ||
							((sample.sampledFlags & (BSDF_GLOSSY | BSDF_SPECULAR | BSDF_FILTER | BSDF_DISPERSIVE)) && causticPhoton);
			directPhoton = (sample.sampledFlags & BSDF_FILTER) && directPhoton;

			ray.from = sp.P;
			ray.dir = wo;
			ray.tmin = MIN_RAYDIST;
			ray.tmax = -1.0;
			++nBounces;
		}
		++curr;
		if(curr % pbStep == 0) pb->update();
		done = (curr >= nPhotons);
	}
	pb->done();
	pb->setTag("Diffuse photon map built.");
	Y_INFO << integratorName << ": Diffuse photon map built." << yendl;
	Y_INFO << integratorName << ": Shot "<<curr<<" photons from " << numDLights << " light(s)" << yendl;

	delete lightPowerD;

	tmplights.clear();

	for(int i=0;i<(int)lights.size();++i)
	{
		if(lights[i]->shootsCausticP())
		{
			numCLights++;
			tmplights.push_back(lights[i]);
		}
	}

	if(numCLights > 0)
	{
		
		done = false;
		curr=0;

		fNumLights = (float)numCLights;
		energies = new float[numCLights];

		for(int i=0;i<numCLights;++i) energies[i] = tmplights[i]->totalEnergy().energy();

		lightPowerD = new pdf1D_t(energies, numCLights);
		
		Y_INFO << integratorName << ": Light(s) photon color testing for caustics map:" << yendl;
		for(int i=0;i<numCLights;++i)
		{
			pcol = tmplights[i]->emitPhoton(.5, .5, .5, .5, ray, lightPdf);
			lightNumPdf = lightPowerD->func[i] * lightPowerD->invIntegral;
			pcol *= fNumLights*lightPdf/lightNumPdf; //remember that lightPdf is the inverse of the pdf, hence *=...
			Y_INFO << integratorName << ": Light [" << i+1 << "] Photon col:" << pcol << " | lnpdf: " << lightNumPdf << yendl;
		}
		
		delete[] energies;

		Y_INFO << integratorName << ": Building caustics photon map..." << yendl;
		pb->init(128);
		pbStep = std::max(1U, nCausPhotons / 128);
		pb->setTag("Building caustics photon map...");
		//Pregather caustic photons
		
		float invCaustPhotons = 1.f / (float)nCausPhotons;
		
		while(!done)
		{
			if(scene->getSignals() & Y_SIG_ABORT) { pb->done(); if(!intpb) delete pb; return false; }
			state.chromatic = true;
			state.wavelength = scrHalton(5,curr);

			s1 = RI_vdC(curr);
			s2 = scrHalton(2, curr);
			s3 = scrHalton(3, curr);
			s4 = scrHalton(4, curr);

			sL = float(curr) * invCaustPhotons;
			int lightNum = lightPowerD->DSample(sL, &lightNumPdf);
			
			if(lightNum >= numCLights)
			{
				Y_ERROR << integratorName << ": lightPDF sample error! "<<sL<<"/"<<lightNum<<"... stopping now." << yendl;
				delete lightPowerD;
				return false;
			}

			pcol = tmplights[lightNum]->emitPhoton(s1, s2, s3, s4, ray, lightPdf);
			ray.tmin = MIN_RAYDIST;
			ray.tmax = -1.0;
			pcol *= fNumLights*lightPdf/lightNumPdf; //remember that lightPdf is the inverse of th pdf, hence *=...
			if(pcol.isBlack())
			{
				++curr;
				done = (curr >= nCausPhotons);
				continue;
			}
			int nBounces=0;
			bool causticPhoton = false;
			bool directPhoton = true;
			const material_t *material = NULL;
			BSDF_t bsdfs;

			while( scene->intersect(ray, sp) )
			{
				if(isnan(pcol.R) || isnan(pcol.G) || isnan(pcol.B))
				{
					Y_WARNING << integratorName << ": NaN  on photon color for light" << lightNum + 1 << "." << yendl;
					continue;
				}
				
				color_t transm(1.f);
				color_t vcol(0.f);
				const volumeHandler_t* vol;
				
				if(material)
				{
					if((bsdfs&BSDF_VOLUMETRIC) && (vol=material->getVolumeHandler(sp.Ng * -ray.dir < 0)))
					{
						if(vol->transmittance(state, ray, vcol)) transm = vcol;
					}
				}
				
				vector3d_t wi = -ray.dir, wo;
				material = sp.material;
				material->initBSDF(state, sp, bsdfs);

				if(bsdfs & (BSDF_DIFFUSE | BSDF_GLOSSY))
				{
					if(causticPhoton)
					{
						photon_t np(wi, sp.P, pcol);
						causticMap.pushPhoton(np);
						causticMap.setNumPaths(curr);
					}
				}
				
				// need to break in the middle otherwise we scatter the photon and then discard it => redundant
				if(nBounces == maxBounces) break;
				// scatter photon
				int d5 = 3*nBounces + 5;

				s5 = scrHalton(d5, curr);
				s6 = scrHalton(d5+1, curr);
				s7 = scrHalton(d5+2, curr);

				pSample_t sample(s5, s6, s7, BSDF_ALL, pcol, transm);

				bool scattered = material->scatterPhoton(state, sp, wi, wo, sample);
				if(!scattered) break; //photon was absorped.

				pcol = sample.color;

				causticPhoton = ((sample.sampledFlags & (BSDF_GLOSSY | BSDF_SPECULAR | BSDF_DISPERSIVE)) && directPhoton) ||
								((sample.sampledFlags & (BSDF_GLOSSY | BSDF_SPECULAR | BSDF_FILTER | BSDF_DISPERSIVE)) && causticPhoton);
				directPhoton = (sample.sampledFlags & BSDF_FILTER) && directPhoton;
				
				if(state.chromatic && (sample.sampledFlags & BSDF_DISPERSIVE))
				{
					state.chromatic=false;
					color_t wl_col;
					wl2rgb(state.wavelength, wl_col);
					pcol *= wl_col;
				}
				
				ray.from = sp.P;
				ray.dir = wo;
				ray.tmin = MIN_RAYDIST;
				ray.tmax = -1.0;
				++nBounces;
			}
			++curr;
			if(curr % pbStep == 0) pb->update();
			done = (curr >= nCausPhotons);
		}
		
		pb->done();
		pb->setTag("Caustics photon map built.");
		delete lightPowerD;
	}
	else
	{
		Y_INFO << integratorName << ": No caustic source lights found, skiping caustic gathering..." << yendl;
	}
	
	Y_INFO << integratorName << ": Shot "<<curr<<" caustic photons from " << numCLights <<" light(s)." << yendl;
	Y_INFO << integratorName << ": Stored caustic photons: " << causticMap.nPhotons() << yendl;
	Y_INFO << integratorName << ": Stored diffuse photons: " << diffuseMap.nPhotons() << yendl;
	
	if(diffuseMap.nPhotons() > 0)
	{
		Y_INFO << integratorName << ": Building diffuse photons kd-tree:" << yendl;
		pb->setTag("Building diffuse photons kd-tree...");
		diffuseMap.updateTree();
		Y_INFO << integratorName << ": Done." << yendl;
	}

	if(causticMap.nPhotons() > 0)
	{
		Y_INFO << integratorName << ": Building caustic photons kd-tree:" << yendl;
		pb->setTag("Building caustic photons kd-tree...");
		causticMap.updateTree();
		Y_INFO << integratorName << ": Done." << yendl;
	}

	if(diffuseMap.nPhotons() < 50)
	{
		Y_ERROR << integratorName << ": Too few diffuse photons, stopping now." << yendl;
		return false;
	}
	
	lookupRad = 4*dsRadius*dsRadius;
	
	tmplights.clear();

	if(!intpb) delete pb;
	
	if(finalGather) //create radiance map:
	{
#ifdef USING_THREADS
		// == remove too close radiance points ==//
		kdtree::pointKdTree< radData_t > *rTree = new kdtree::pointKdTree< radData_t >(pgdat.rad_points);
		std::vector< radData_t > cleaned;
		for(unsigned int i=0; i<pgdat.rad_points.size(); ++i)
		{
			if(pgdat.rad_points[i].use)
			{
				cleaned.push_back(pgdat.rad_points[i]);
				eliminatePhoton_t elimProc(pgdat.rad_points[i].normal);
				PFLOAT maxrad = 0.01f*dsRadius; // 10% of diffuse search radius
				rTree->lookup(pgdat.rad_points[i].pos, elimProc, maxrad);
			}
		}
		pgdat.rad_points.swap(cleaned);
		// ================ //
		int nThreads = scene->getNumThreads();
		pgdat.radianceVec.resize(pgdat.rad_points.size());
		if(intpb) pgdat.pbar = intpb;
		else pgdat.pbar = new ConsoleProgressBar_t(80);
		pgdat.pbar->init(pgdat.rad_points.size());
		pgdat.pbar->setTag("Pregathering radiance data for final gathering...");
		std::vector<preGatherWorker_t *> workers;
		for(int i=0; i<nThreads; ++i) workers.push_back(new preGatherWorker_t(&pgdat, dsRadius, nSearch));
		
		for(int i=0;i<nThreads;++i) workers[i]->run();
		for(int i=0;i<nThreads;++i)	workers[i]->wait();
		for(int i=0;i<nThreads;++i)	delete workers[i];
		
		radianceMap.swapVector(pgdat.radianceVec);
		pgdat.pbar->done();
		pgdat.pbar->setTag("Pregathering radiance data done...");
		if(!intpb) delete pgdat.pbar;
#else
		if(radianceMap.nPhotons() != 0)
		{
			Y_WARNING << integratorName << ": radianceMap not empty!" << yendl;
			radianceMap.clear();
		}
		
		Y_INFO << integratorName << ": Creating radiance map..." << yendl;
		progressBar_t *pbar;
		if(intpb) pbar = intpb;
		else pbar = new ConsoleProgressBar_t(80);
		pbar->init(pgdat.rad_points.size());
		foundPhoton_t *gathered = (foundPhoton_t *)malloc(nSearch * sizeof(foundPhoton_t));
		PFLOAT dsRadius_2 = dsRadius*dsRadius;
		for(unsigned int n=0; n< pgdat.rad_points.size(); ++n)
		{
			PFLOAT radius = dsRadius_2; //actually the square radius...
			int nGathered = diffuseMap.gather(pgdat.rad_points[n].pos, gathered, nSearch, radius);
			color_t sum(0.0);
			if(nGathered > 0)
			{
				color_t surfCol = pgdat.rad_points[n].refl;
				vector3d_t rnorm = pgdat.rad_points[n].normal;
				float scale = 1.f / ( float(diffuseMap.nPaths()) * radius * M_PI);
				
				if(isnan(scale))
				{
					Y_WARNING << integratorName << ": NaN on (scale)" << yendl;
					break;
				}
				
				for(int i=0; i<nGathered; ++i)
				{
					vector3d_t pdir = gathered[i].photon->direction();
					
					if( rnorm * pdir > 0.f ) sum += surfCol * scale * gathered[i].photon->color();
					else sum += pgdat.rad_points[n].transm * scale * gathered[i].photon->color();
				}
			}
			photon_t radP(pgdat.rad_points[n].normal, pgdat.rad_points[n].pos, sum);
			radianceMap.pushPhoton(radP);
			if(n && !(n&7)) pbar->update(8);
		}
		pbar->done();
		if(!pbar) delete pbar;
		free(gathered);
#endif
		Y_INFO << integratorName << ": Radiance tree built... Updating the tree..." << yendl;
		radianceMap.updateTree();
		Y_INFO << integratorName << ": Done." << yendl;
	}

	gTimer.stop("prepass");
	Y_INFO << integratorName << ": Photonmap building time: " << gTimer.getTime("prepass") << yendl;

	return true;
}

// final gathering: this is basically a full path tracer only that it uses the radiance map only
// at the path end. I.e. paths longer than 1 are only generated to overcome lack of local radiance detail.
// precondition: initBSDF of current spot has been called!
color_t photonIntegratorGPU_t::finalGathering(renderState_t &state, const surfacePoint_t &sp, const vector3d_t &wo) const
{
	color_t pathCol(0.0);
	void *first_udat = state.userdata;
	unsigned char userdata[USER_DATA_SIZE+7];
	void *n_udat = (void *)( &userdata[7] - ( ((size_t)&userdata[7])&7 ) ); // pad userdata to 8 bytes
	const volumeHandler_t *vol;
	color_t vcol(0.f);
	
	int nSampl = std::max(1, nPaths/state.rayDivision);
	for(int i=0; i<nSampl; ++i)
	{
		color_t throughput( 1.0 );
		PFLOAT length=0;
		surfacePoint_t hit=sp;
		vector3d_t pwo = wo;
		ray_t pRay;
		BSDF_t matBSDFs;
		bool did_hit;
		const material_t *p_mat = sp.material;
		unsigned int offs = nPaths * state.pixelSample + state.samplingOffs + i; // some redundancy here...
		color_t lcol, scol;
		// "zero'th" FG bounce:
		float s1 = RI_vdC(offs);
		float s2 = scrHalton(2, offs);
		if(state.rayDivision > 1)
		{
			s1 = addMod1(s1, state.dc1);
			s2 = addMod1(s2, state.dc2);
		}

		sample_t s(s1, s2, BSDF_DIFFUSE|BSDF_REFLECT|BSDF_TRANSMIT); // glossy/dispersion/specular done via recursive raytracing
		scol = p_mat->sample(state, hit, pwo, pRay.dir, s);

		if(s.pdf <= 1.0e-6f) continue;
		scol *= (std::fabs(pRay.dir*sp.N)/s.pdf);
		if(scol.isBlack()) continue;

		pRay.tmin = MIN_RAYDIST;
		pRay.tmax = -1.0;
		pRay.from = hit.P;
		throughput = scol;
		
		if( !(did_hit = scene->intersect(pRay, hit)) ) continue; //hit background
		
		p_mat = hit.material;
		length = pRay.tmax;
		state.userdata = n_udat;
		matBSDFs = p_mat->getFlags();
		bool has_spec = matBSDFs & BSDF_SPECULAR;
		bool caustic = false;
		bool close = length < gatherDist;
		bool do_bounce = close || has_spec;
		// further bounces construct a path just as with path tracing:
		for(int depth=0; depth<gatherBounces && do_bounce; ++depth)
		{
			int d4 = 4*depth;
			pwo = -pRay.dir;
			p_mat->initBSDF(state, hit, matBSDFs);
			
			if((matBSDFs & BSDF_VOLUMETRIC) && (vol=p_mat->getVolumeHandler(hit.N * pwo < 0)))
			{
				if(vol->transmittance(state, pRay, vcol)) throughput *= vcol;
			}
	
			if(matBSDFs & (BSDF_DIFFUSE))
			{
				if(close)
				{
					lcol = estimateOneDirect(state, hit, pwo, lights, d4+5, offs);
				}
				else if(caustic)
				{
					vector3d_t sf = FACE_FORWARD(hit.Ng, hit.N, pwo);
					const photon_t *nearest = radianceMap.findNearest(hit.P, sf, lookupRad);
					if(nearest) lcol = nearest->color();
				}
				
				if(close || caustic)
				{
					if(matBSDFs & BSDF_EMIT) lcol += p_mat->emit(state, hit, pwo);
					pathCol += lcol*throughput;
				}
			}
			
			s1 = scrHalton(d4+3, offs);
			s2 = scrHalton(d4+4, offs);

			if(state.rayDivision > 1)
			{
				s1 = addMod1(s1, state.dc1);
				s2 = addMod1(s2, state.dc2);
			}
			
			sample_t sb(s1, s2, (close) ? BSDF_ALL : BSDF_ALL_SPECULAR | BSDF_FILTER);
			scol = p_mat->sample(state, hit, pwo, pRay.dir, sb);
			
			if( sb.pdf <= 1.0e-6f)
			{
				did_hit=false;
				break;
			}

			scol *= (std::fabs(pRay.dir*hit.N)/sb.pdf);

			pRay.tmin = MIN_RAYDIST;
			pRay.tmax = -1.0;
			pRay.from = hit.P;
			throughput *= scol;
			did_hit = scene->intersect(pRay, hit);
			
			if(!did_hit) //hit background
			{
				 if(caustic && hasBGLight)
				 {
					pathCol += throughput * (*background)(pRay, state);
				 }
				 break;
			}
			
			p_mat = hit.material;
			length += pRay.tmax;
			caustic = (caustic || !depth) && (sb.sampledFlags & (BSDF_SPECULAR | BSDF_FILTER));
			close =  length < gatherDist;
			do_bounce = caustic || close;
		}
		
		if(did_hit)
		{
			p_mat->initBSDF(state, hit, matBSDFs);
			if(matBSDFs & (BSDF_DIFFUSE | BSDF_GLOSSY))
			{
				vector3d_t sf = FACE_FORWARD(hit.Ng, hit.N, -pRay.dir);
				const photon_t *nearest = radianceMap.findNearest(hit.P, sf, lookupRad);
				if(nearest) lcol = nearest->color();
				if(matBSDFs & BSDF_EMIT) lcol += p_mat->emit(state, hit, -pRay.dir);
				pathCol += lcol * throughput;
			}
		}
		state.userdata = first_udat;
	}
	return pathCol / (float)nSampl;
}

bool photonIntegratorGPU_t::getSPfromHit(const diffRay_t &ray, int tri_idx, surfacePoint_t &sp) const
{
	if(tri_idx == -1)
		return false;

	unsigned char udat[PRIM_DAT_SIZE];

	const triangle_t *hitt = prims[tri_idx];
	
	PFLOAT Z;
	bool hit = hitt->intersect(ray, &Z, (void*)&udat[0]);
	if(!hit) {
		Y_ERROR << "hit was not computed correctly" << yendl;
		return false;
	}

	point3d_t h=ray.from + Z*ray.dir;
	hitt->getSurface(sp, h, (void*)&udat[0]);
	sp.origin = (triangle_t*)hitt;
	return true;
}

bool photonIntegratorGPU_t::getSPforRay(const diffRay_t &ray, renderState_t &rstate, surfacePoint_t &sp) const
{
	if(ray.idx < 0)
		return scene->intersect(ray, sp);

	int tri_idx = rstate.inter_tris[ray.idx];
	return getSPfromHit(ray, tri_idx, sp);
}

colorA_t photonIntegratorGPU_t::integrate(renderState_t &state, diffRay_t &ray) const
{
	if(ray.idx >= 0) {
		const diffRay_t &orig_ray = state.c_rays[ray.idx];
		if(ray.from.x != orig_ray.from.x ||
			ray.from.y != orig_ray.from.y ||
			ray.from.z != orig_ray.from.z ||
			ray.xfrom.x != orig_ray.xfrom.x ||
			ray.xfrom.y != orig_ray.xfrom.y ||
			ray.xfrom.z != orig_ray.xfrom.z ||
			ray.yfrom.x != orig_ray.yfrom.x ||
			ray.yfrom.y != orig_ray.yfrom.y ||
			ray.yfrom.z != orig_ray.yfrom.z ||
			ray.dir.x != orig_ray.dir.x ||
			ray.dir.y != orig_ray.dir.y ||
			ray.dir.z != orig_ray.dir.z ||
			ray.xdir.x != orig_ray.xdir.x ||
			ray.xdir.y != orig_ray.xdir.y ||
			ray.xdir.z != orig_ray.xdir.z ||
			ray.ydir.x != orig_ray.ydir.x ||
			ray.ydir.y != orig_ray.ydir.y ||
			ray.ydir.z != orig_ray.ydir.z ||
			ray.time != orig_ray.time ||
			ray.hasDifferentials != orig_ray.hasDifferentials)
		{
			Y_ERROR << "different ray parameters" << yendl;
		}
	}

	static int _nMax=0;
	static int calls=0;
	++calls;
	color_t col(0.0);
	CFLOAT alpha=0.0;

	surfacePoint_t sp;

	void *o_udat = state.userdata;
	bool oldIncludeLights = state.includeLights;

	if(getSPforRay(ray, state, sp))
	{
		unsigned char userdata[USER_DATA_SIZE+7];
		state.userdata = (void *)( &userdata[7] - ( ((size_t)&userdata[7])&7 ) ); // pad userdata to 8 bytes
		if(state.raylevel == 0)
		{
			state.chromatic = true;
			state.includeLights = true;
		}
		BSDF_t bsdfs;
		vector3d_t N_nobump = sp.N;
		vector3d_t wo = -ray.dir;
		const material_t *material = sp.material;
		material->initBSDF(state, sp, bsdfs);
		col += material->emit(state, sp, wo);
		state.includeLights = false;
		spDifferentials_t spDiff(sp, ray);
		
		if(finalGather)
		{
			if(showMap)
			{
				vector3d_t N = FACE_FORWARD(sp.Ng, sp.N, wo);
				const photon_t *nearest = radianceMap.findNearest(sp.P, N, lookupRad);
				if(nearest) col += nearest->color();
			}
			else
			{
				// contribution of light emitting surfaces
				if(bsdfs & BSDF_EMIT) col += material->emit(state, sp, wo);
				
				if(bsdfs & BSDF_DIFFUSE) col += estimateDirect_PH(state, sp, lights, scene, wo, trShad, sDepth);
				
				if(bsdfs & BSDF_DIFFUSE) col += finalGathering(state, sp, wo);
			}
		}
		else
		{
			foundPhoton_t *gathered = (foundPhoton_t *)alloca(nSearch * sizeof(foundPhoton_t));
			PFLOAT radius = dsRadius; //actually the square radius...

			int nGathered=0;
			
			if(diffuseMap.nPhotons() > 0) nGathered = diffuseMap.gather(sp.P, gathered, nSearch, radius);
			color_t sum(0.0);
			if(nGathered > 0)
			{
				if(nGathered > _nMax) _nMax = nGathered;

				float scale = 1.f / ( float(diffuseMap.nPaths()) * radius * M_PI);
				for(int i=0; i<nGathered; ++i)
				{
					vector3d_t pdir = gathered[i].photon->direction();
					color_t surfCol = material->eval(state, sp, wo, pdir, BSDF_DIFFUSE);
					col += surfCol * scale * gathered[i].photon->color();
				}
			}
		}
		
		// add caustics
		if(bsdfs & (BSDF_DIFFUSE)) col += estimatePhotons(state, sp, causticMap, wo, nCausSearch, cRadius);
		
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
	state.userdata = o_udat;
	state.includeLights = oldIncludeLights;
	return colorA_t(col, alpha);
}

integrator_t* photonIntegratorGPU_t::factory(paraMap_t &params, renderEnvironment_t &render)
{
	bool transpShad=false;
	bool finalGather=true;
	bool show_map=false;
	int shadowDepth=5;
	int raydepth=5;
	int numPhotons = 100000;
	int numCPhotons = 500000;
	int search = 50;
	int caustic_mix = 50;
	int bounces = 5;
	int fgPaths = 32;
	int fgBounces = 2;
	float dsRad=0.1;
	float cRad=0.01;
	float gatherDist=0.2;
	
	params.getParam("transpShad", transpShad);
	params.getParam("shadowDepth", shadowDepth);
	params.getParam("raydepth", raydepth);
	params.getParam("photons", numPhotons);
	params.getParam("cPhotons", numCPhotons);
	params.getParam("diffuseRadius", dsRad);
	params.getParam("causticRadius", cRad);
	params.getParam("search", search);
	caustic_mix = search;
	params.getParam("caustic_mix", caustic_mix);
	params.getParam("bounces", bounces);
	params.getParam("finalGather", finalGather);
	params.getParam("fg_samples", fgPaths);
	params.getParam("fg_bounces", fgBounces);
	gatherDist = dsRad;
	params.getParam("fg_min_pathlen", gatherDist);
	params.getParam("show_map", show_map);
	
	photonIntegratorGPU_t* ite = new photonIntegratorGPU_t(numPhotons, numCPhotons, transpShad, shadowDepth, dsRad, cRad);
	ite->rDepth = raydepth;
	ite->nSearch = search;
	ite->nCausSearch = caustic_mix;
	ite->finalGather = finalGather;
	ite->maxBounces = bounces;
	ite->nPaths = fgPaths;
	ite->gatherBounces = fgBounces;
	ite->showMap = show_map;
	ite->gatherDist = gatherDist;
	return ite;
}

struct BuildDiskHierarchy_CoordinateComparator
{
	int coord;
	BuildDiskHierarchy_CoordinateComparator(int coord) :
	coord(coord) {}
	bool operator()(const photonIntegratorGPU_t::Disk& a, const photonIntegratorGPU_t::Disk &b) {
		return a.c[coord] < b.c[coord];
	}
};

void photonIntegratorGPU_t::build_disk_hierarchy(PHierarchy &ph, std::vector<const triangle_t *> &prims, int node_poz, DiskVectorType &v, int s, int e)
{
	if(s >= e - 1) {
		assert(s == e-1);

		while(1) {
			int leaf_poz = node_poz - ph.int_nodes.size();
			if(leaf_poz >= 0) {
				assert(leaf_poz < ph.leaves.size());
				ph.leaves[leaf_poz] = PHLeaf(v[s].c, prims[v[s].tri_idx]->getNormal(), v[s].tri_idx);
				//leaves[leaf_poz] = PHLeaf(v[s].c, v[s].t->getNormal(), v[s].t);
				//leaf_tris[leaf_poz] = v[s].t;
				return;
			} else {
				assert(false);
				//int_nodes[node_poz] = PHInternalNode(v[s].c, leaf_radius);
				node_poz *= 2;
			}
		}
	}

	float max_c[3], min_c[3];
	for(int i = 0; i < 3; i++) {
		max_c[i] = -std::numeric_limits<float>::max();
		min_c[i] = std::numeric_limits<float>::max();
	}

	for(int poz = s; poz < e; poz++) {
		point3d_t &p = v[poz].c;
		for(int i = 0; i < 3; i++) {
			if(p[i] < min_c[i]) min_c[i] = p[i];
			if(p[i] > max_c[i]) max_c[i] = p[i];
		}
	}

	float max_ex = -std::numeric_limits<float>::max();
	int ex = 0;
	for(int i = 0; i < 3; i++) {
		float exi = max_c[i] - min_c[i];
		if(exi > max_ex) {
			max_ex = exi;
			ex = i;
		}
	}

	BuildDiskHierarchy_CoordinateComparator comp(ex);

	int m = (s+e)/2;

	//std::sort(v.begin() + s, v.begin() + e, comp);									// O(n log n)
	//NOTE: the following works iff only partitioning is required
	std::nth_element(v.begin() + s, v.begin() + m, v.begin() + e, comp);	// O(n)

	// a range [s,e) denotes a certain subtree
	// the radius is computed as the bound of the subtrees
	// [0,m) and [m, e)

	int left_poz = 2*node_poz;
	int right_poz = 2*node_poz+1;

	build_disk_hierarchy(ph, prims, left_poz, v, s, m);
	build_disk_hierarchy(ph, prims, right_poz, v, m, e);

	float r1, r2;
	point3d_t c1, c2;

	int left_leaf_poz = left_poz - ph.int_nodes.size();
	if(left_leaf_poz >= 0) {	// right child is a leaf
		PHLeaf &l = ph.leaves[left_leaf_poz];
		c1 = l.c;
		r1 = ph.leaf_radius;
	} else {			// right child is an internal node
		PHInternalNode &n = ph.int_nodes[left_poz];
		c1 = n.c;
		r1 = n.r;
	}

	int right_leaf_poz = right_poz - ph.int_nodes.size();
	if(right_leaf_poz >= 0) {		// left child is a leaf
		PHLeaf &l = ph.leaves[right_leaf_poz];
		c2 = l.c;
		r2 = ph.leaf_radius;
	} else {			// left child is an internal node
		PHInternalNode &n = ph.int_nodes[right_poz];
		c2 = n.c;
		r2 = n.r;
	}

	vector3d_t c21 = c2-c1;
	float c21m = c21.length();

	PHInternalNode &n =  ph.int_nodes[node_poz];
	n.r = (c21m+r1+r2)/2;
	assert(n.r > 0);
	n.c = c1 + c21 *( (n.r-r1)/c21m );
	n.coord = ex;
	n.M = v[m].c[ex];
}

struct GeneratePoints_PointGenFunc {
    point3d_t &a;
    vector3d_t ab, ac;

    GeneratePoints_PointGenFunc(photonIntegratorGPU_t::PHTriangle &t) : a(t.a)
    {
        ab = t.b-t.a, ac = t.c-t.a;
    }

    point3d_t operator()(int) {
        float u = ourRandom();
        float v = ourRandom();
        //printf("(%f,%f,%f)", points[j].p.x, points[j].p.y, points[j].p.z);
        return a + v * ac + (1-v) * u * ab;
    }
};

void photonIntegratorGPU_t::generate_points(DiskVectorType &disks, std::vector<const triangle_t*> &prims, PHierarchy &ph, scene_t *scene)
{
	scene_t::objDataArray_t &meshes = scene->getMeshes();

	int total_prims = 0;
	for(scene_t::objDataArray_t::iterator itr = meshes.begin(); itr != meshes.end(); ++itr) {
		total_prims += itr->second.obj->numPrimitives();
	}

	prims.resize(total_prims);
	ph.tris.resize(total_prims);

	std::vector<int> nr_to_keep;
	nr_to_keep.resize(total_prims);
	int tri_idx = 0;

	//float area_multiplier = 2 * sqrt(2.0);
	float area_multiplier = 3.5;
	//float area_multiplier = 1;

	int total_nr_tmp = 0;
	int prims_remaining = total_prims;

	for(scene_t::objDataArray_t::iterator itr = meshes.begin(); itr != meshes.end(); ++itr)
	{
		triangleObject_t *obj = itr->second.obj;
		int nr_mesh_primitives = obj->numPrimitives();
		itr->second.obj->getPrimitives(&prims[tri_idx]);

		for(int i = 0; i < nr_mesh_primitives; ++i) {
			const triangle_t *t = prims[tri_idx];

			vector3d_t n = t->getNormal();
			if(n.lengthSqr() < 1.0 - 1e-5) {
				point3d_t a,b,c;
				t->getVertices(a, b, c);
				Y_ERROR << "found triangle " 
					<< "(" << a.x << "," << a.y << "," << a.z << "), "
					<< "(" << b.x << "," << b.y << "," << b.z << "), "
					<< "(" << c.x << "," << c.y << "," << c.z << ") "
					<< "with invalid normal "
					<< "(" << n.x << "," << n.y << "," << n.z << ")" << yendl;
				nr_to_keep[tri_idx] = 0;
				++tri_idx;
				--prims_remaining;
				continue;
			}

			float area = t->surfaceArea();
			float cur_nr = std::max((int)(area * area_multiplier / (ph.leaf_radius*ph.leaf_radius*M_PI)), 1);
			total_nr_tmp += cur_nr;
			nr_to_keep[tri_idx] = cur_nr;
			++tri_idx;
		}
	}

	int total_nr = 1;
	while(total_nr < total_nr_tmp) total_nr *= 2;

	int diff = (total_nr - total_nr_tmp);
	int diff_per_tri = diff / prims_remaining;
	int diff_mod = diff % prims_remaining;
		
	for(int i = 0; i < total_prims; ++i)
	{
		if(nr_to_keep[i] == 0)
			continue;

		const triangle_t *t = prims[i];
		int cur_nr = nr_to_keep[i] + diff_per_tri;
		if(diff_mod > 0) {
			diff_mod --;
			cur_nr ++;
		}

		PHTriangle &pht = ph.tris[i];
		t->getVertices(pht.a, pht.b, pht.c);

		BestCandidateSampler sampler;
		GeneratePoints_PointGenFunc gen_func(pht);

		sampler.gen_candidates(gen_func, cur_nr);
		sampler.gen_samples();

		for(BestCandidateSampler::iterator itr = sampler.begin(); itr != sampler.end(); ++itr) {
				disks.push_back(Disk((*itr), i));
		}
	}
}

bool photonIntegratorGPU_t::renderTile(renderArea_t &a, int n_samples, int offset, bool adaptive, int threadID)
{
	class RayStorer : public tiledIntegrator_t::PrimaryRayGenerator {
		protected:
			std::vector<diffRay_t> &c_rays;
		public:
			RayStorer(
				renderArea_t &a, int n_samples, int offset, 
				tiledIntegrator_t *integrator, random_t &prng,
				std::vector<diffRay_t> &c_rays
			) : PrimaryRayGenerator(a, n_samples, offset, integrator, prng),
				c_rays(c_rays)
			{

			}
			void rays(diffRay_t &c_ray, int i, int j, int dx, int dy, float wt)
			{
				c_rays.push_back(c_ray);
			}
	};

	random_t prng_rt(offset * (scene->getCamera()->resX() * a.Y + a.X) + 123);
	RenderTile_PrimaryRayGenerator raygen_rt(a, n_samples, offset, adaptive, threadID, this, prng_rt);

	renderState_t &state = raygen_rt.getRenderState();
	std::vector<diffRay_t> &c_rays = state.c_rays;
	std::vector<int> &inter_tris = state.inter_tris;

	random_t prng_rs(offset * (scene->getCamera()->resX() * a.Y + a.X) + 123);
	RayStorer raygen_rs(a, n_samples, offset, this, prng_rs, c_rays);
	
	raygen_rs.genRays();

	std::vector<PHRay> rays;
	rays.resize(c_rays.size());
	for(int i = 0; i < (int)c_rays.size(); ++i)
	{
		rays[i].p = c_rays[i].from;
		rays[i].r = c_rays[i].dir;
	}

	CLError err;
	CLBuffer *d_rays = context->createBuffer(rays, &err);
	checkErr(err || !d_rays, "failed to create internal node buffer");
	queue->writeBuffer(d_rays, rays, &err);
	checkErr(err, "failed to write d_rays");

	inter_tris.resize(rays.size());

	CLBuffer *d_inter_tris = context->createBuffer(inter_tris, &err);
	checkErr(err || !d_inter_tris, "failed to create internal node buffer");
	// initialize
	queue->writeBuffer(d_inter_tris, inter_tris, &err); 
	checkErr(err, "failed to initialize d_inter_tris");

	CLKernel *kernel = program->createKernel("intersect_rays_test", &err);
	checkErr(err || !kernel, "failed to create kernel");

	CLVectorBuffer<float> dbg;

	static int use_debug_buffer = false;
	if(use_debug_buffer) {
		dbg.resize(8*rays.size());
		queue->writeBuffer(dbg, &err);
		checkErr(err, "failed to init dbg");
	}

	/*
	__kernel void intersect_rays(
		__global PHInternalNode *int_nodes, 
		int nr_int_nodes,
		__global PHLeaf *leaves,
		float leaf_radius,
		__global PHTriangle *tris,
		int nr_tris,
		__global PHRay *rays,
		int nr_rays,
		__global int *inter_tris,
		__global float *dbg,
		int do_debug
	){
	*/
	kernel->setArgs(
		d_int_nodes,
		(int)pHierarchy.int_nodes.size(),
		d_leaves,
		pHierarchy.leaf_radius,
		d_tris,
		(int)prims.size(),
		d_rays,
		(int)rays.size(),
		d_inter_tris,
		dbg,
		use_debug_buffer,
		&err
	);
	checkErr(err, "failed to set kernel args");

	int local_size = 32;
	int global_size = rays.size();
	if(global_size % local_size != 0) 
		global_size = (global_size / local_size + 1) * (local_size);

	queue->runKernel(kernel, Range1D(global_size, local_size), &err);
	checkErr(err, "failed to run kernel");

	queue->readBuffer(d_inter_tris, inter_tris, &err);
	checkErr(err, "failed to read inter_tris");

	if(use_debug_buffer) {
		queue->readBuffer(dbg, &err);
		checkErr(err, "failed to read dbg_nr");
	}

	kernel->free(&err);
	checkErr(err, "failed to free kernel");
	d_inter_tris->free(&err);
	checkErr(err, "failed to free inter_tris");
	d_rays->free(&err);
	checkErr(err, "failed to free d_rays");

	//RayTest test(*this, this->pHierarchy);
	//test.test_rays(state);

	raygen_rt.genRays();
	return true;
}

void photonIntegratorGPU_t::upload_hierarchy(PHierarchy &ph)
{
	CLError err;
	d_int_nodes = context->createBuffer(ph.int_nodes, &err);
	checkErr(err || !d_int_nodes, "failed to create internal node buffer");
	queue->writeBuffer(d_int_nodes, ph.int_nodes, &err);
	checkErr(err, "failed to write d_int_nodes");

	d_leaves = context->createBuffer(ph.leaves, &err);
	checkErr(err || !d_leaves, "failed to create leaf buffer");
	queue->writeBuffer(d_leaves, ph.leaves, &err);
	checkErr(err, "failed to write d_leaves");

	d_tris = context->createBuffer(ph.tris, &err);
	checkErr(err || !d_tris, "failed to create leaf buffer");
	queue->writeBuffer(d_tris, ph.tris, &err);
	checkErr(err, "failed to write d_tris");

	char * kernel_src = CL_SRC(
		typedef struct
		{
			float m[3];	// disk center
			float n[3];	// disk normal
			int tri_idx; // index of the triangle the disk belongs to
		} PHLeaf;

		typedef struct
		{
			float c[3];	// bounding sphere center
			float r;		// bounding sphere radius
			int coord;
			float M;
		} PHInternalNode;

		typedef struct 
		{
			float p[3]; // ray origin
			float r[3]; // ray direction
		} PHRay;

		typedef struct
		{
			float a[3], b[3], c[3];
		} PHTriangle;
		

		__kernel void test(
			__global PHInternalNode *int_nodes, 
			__global PHLeaf *leaves,
			int nr_int_nodes,
			__global float *ret
		){
			float total = 0;
			for(int i = 0; i < nr_int_nodes; ++i) {
				total += int_nodes[i].r / nr_int_nodes;
			}

			ret[0] = total;
		}

		void sub(float a[3], float b[3], float c[3]) {	// c = a - b
			c[0] = a[0]-b[0];
			c[1] = a[1]-b[1];
			c[2] = a[2]-b[2];
		}

		void add(float a[3], float b[3], float c[3]) {	// c = a + b
			c[0] = a[0]+b[0];
			c[1] = a[1]+b[1];
			c[2] = a[2]+b[2];
		}

		void _cross(float a[3], float b[3], float c[3]) { // c = a ^ b
			c[0] = a[1]*b[2]-a[2]*b[1];
			c[1] = a[2]*b[0]-a[0]*b[2];
			c[2] = a[0]*b[1]-a[1]*b[0];
		}

		void mul(float a[3], float t, float b[3]) {	// b = t * a
			b[0] = t * a[0];
			b[1] = t * a[1];
			b[2] = t * a[2];
		}

		float _dot(float a[3], float b[3]) {	// ret = a * b
			return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
		}

		float normSqr(float a[3]) {	// ret = |a|^2
			return a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
		}
\n
		__kernel void intersect_rays_test2(
			 __global PHInternalNode *int_nodes, 
			 int nr_int_nodes,
			 __global PHLeaf *leaves,
			 float leaf_radius,
			 __global PHTriangle *tris,
			 int nr_tris,
			 __global PHRay *rays,
			 int nr_rays,
			 __global int *inter_tris,
			 __global float *dbg,
			 int do_debug
		){
			if(get_global_id(0) >= nr_rays)
				return;

			PHRay ray = rays[get_global_id(0)];
			int cand = -1;
			float t_cand = 100000000.0f;

			for(int i = 0; i < nr_tris; ++i)
			{
				PHTriangle tri = tris[i];
				float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
				float det, inv_det, u, v;
				sub(tri.b, tri.a, edge1);
				sub(tri.c, tri.a, edge2);
				_cross(ray.r, edge2, pvec);
				det = _dot(edge1, pvec);
				//if (/*(det>-0.000001) && (det<0.000001))
				if (det == 0.0f)
					continue;
				inv_det = 1.0f / det;
				sub(ray.p, tri.a, tvec);
				u = _dot(tvec,pvec) * inv_det;
				if (u < 0.0f || u > 1.0f)
					continue;
				_cross(tvec,edge1,qvec);
				v = _dot(ray.r,qvec) * inv_det;
				if ((v<0.0f) || ((u+v)>1.0f) )
					continue;
				float t = _dot(edge2,qvec) * inv_det;
				if(t < t_cand) {
					t_cand = t;
					cand = i;
				}
			}

			inter_tris[get_global_id(0)] = cand;
		}
\n
		__kernel void intersect_rays_test(
			 __global PHInternalNode *int_nodes, 
			 int nr_int_nodes,
			 __global PHLeaf *leaves,
			 float leaf_radius,
			 __global PHTriangle *tris,
			 int nr_tris,
			 __global PHRay *rays,
			 int nr_rays,
			 __global int *inter_tris,
		     __global float *dbg,
			 int do_debug
		){
			if(get_global_id(0) >= nr_rays)
				return;

			PHRay ray = rays[get_global_id(0)];
			int cand = -1;
			float t_cand = 100000000.0f;

			for(int i = 0; i < nr_int_nodes; ++i)
			{
				PHLeaf l = leaves[i];
				float nr = _dot(l.n, ray.r);
				if(nr >= 0) {
					continue;
				}
\n
				float pm[3];
				sub(l.m, ray.p, pm);
				float t = _dot(l.n, pm) / nr;
				float rt[3];
				mul(ray.r, t, rt);
				float q[3];
				add(ray.p, rt, q);
				float mq[3];
				sub(q, l.m, mq);
				float d2 = normSqr(mq);
				float r2 = leaf_radius * leaf_radius;
				if(d2 < r2) {
					PHTriangle tri = tris[l.tri_idx];
					float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
					float det, inv_det, u, v;
					sub(tri.b, tri.a, edge1);
					sub(tri.c, tri.a, edge2);
					_cross(ray.r, edge2, pvec);
					det = _dot(edge1, pvec);
					//if (/*(det>-0.000001) && (det<0.000001))
					if (det == 0.0f)
						continue;
					inv_det = 1.0f / det;
					sub(ray.p, tri.a, tvec);
					u = _dot(tvec,pvec) * inv_det;
					if (u < 0.0f || u > 1.0f)
						continue;
					_cross(tvec,edge1,qvec);
					v = _dot(ray.r,qvec) * inv_det;
					if ((v<0.0f) || ((u+v)>1.0f) )
						continue;
					t = _dot(edge2,qvec) * inv_det;
					if(t < t_cand) {
						t_cand = t;
						cand = l.tri_idx;

						if(do_debug && get_global_id(0) == 290) {
							int idx = get_global_id(0);
							/*dbg[8*idx] = nr;
							dbg[8*idx+1] = l.n[0];
							dbg[8*idx+2] = l.n[1];
							dbg[8*idx+3] = l.n[2];
							dbg[8*idx+4] = ray.r[0];
							dbg[8*idx+5] = ray.r[1];
							dbg[8*idx+6] = ray.r[2];
							dbg[8*idx+7] = (float)i;*/
							dbg[0] = (float)i;
							dbg[1] = nr;
							dbg[2] = det;
							dbg[3] = inv_det;
							dbg[4] = u;
							dbg[5] = v;
							dbg[6] = t;
							dbg[7] = d2;
							dbg[8] = r2;
						}
					}
				}
			}

			inter_tris[get_global_id(0)] = cand;
		}
\n
		__kernel void intersect_rays(
			__global PHInternalNode *int_nodes, 
			int nr_int_nodes,
			__global PHLeaf *leaves,
			float leaf_radius,
			__global PHTriangle *tris,
			int nr_tris,
			__global PHRay *rays,
			int nr_rays,
			__global int *inter_tris,
			__global float *dbg,
			int do_debug
		){
			if(get_global_id(0) >= nr_rays)
				return;

			if(do_debug && get_global_id(0) == 883) dbg[0] = 555.1234f;

			PHRay ray = rays[get_global_id(0)];
			float t_cand = FLT_MAX;
			int cand_tri = -1;
\n
			unsigned int stack = 0;
			unsigned int mask = 1;
			unsigned int poz = 1;
			while(1) {
				while(1) { // going down
					if(poz < nr_int_nodes) { // check internal node
						PHInternalNode n = int_nodes[poz];
						// d(line, center) <= radius ?
						float pc[3];
						sub(n.c, ray.p, pc);
						float pcxr[3];
						_cross(pc, ray.r, pcxr);
						float d2 = normSqr(pcxr);
						float r2 = n.r * n.r;
						if(d2 > r2) {
							break;
						}
\n
						// is p inside sphere ?
						if(normSqr(pc) > r2) {
							// exists q on line such that q in sphere
							// is q in ray's direction ?
							// if p outside sphere and the line intersects the sphere
							// then c must lie roughly in the ray's direction
							float pcr = _dot(pc, ray.r);
							if(pcr <= 0) {
								break;
							}
							// exists t >= pcr >= 0 => found
						}
\n
					} else { // check leaf
						PHLeaf l = leaves[poz - nr_int_nodes];
						float nr = _dot(l.n, ray.r);
						if(nr >= 0) {
							break;
						}
\n
						float pm[3];
						sub(l.m, ray.p, pm);
						float t = _dot(l.n, pm) / nr;
						float rt[3];
						mul(ray.r, t, rt);
						float q[3];
						add(ray.p, rt, q);
						float mq[3];
						sub(q, l.m, mq);
						float d2 = normSqr(mq);
						float r2 = leaf_radius * leaf_radius;
						if(d2 < r2) {
							PHTriangle tri = tris[l.tri_idx];
							float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
							float det, inv_det, u, v;
							sub(tri.b, tri.a, edge1);
							sub(tri.c, tri.a, edge2);
							_cross(ray.r, edge2, pvec);
							det = _dot(edge1, pvec);
							//if (/*(det>-0.000001) && (det<0.000001))
							if (det == 0.0f)
								break;
							inv_det = 1.0f / det;
							sub(ray.p, tri.a, tvec);
							u = _dot(tvec,pvec) * inv_det;
							if (u < 0.0f || u > 1.0f)
								break;
							_cross(tvec,edge1,qvec);
							v = _dot(ray.r,qvec) * inv_det;
							if ((v<0.0f) || ((u+v)>1.0f) )
								break;
							t = _dot(edge2,qvec) * inv_det;
							if(t < t_cand) {
								t_cand = t;
								cand_tri = l.tri_idx;
								dbg[0] = 666.0f;

								if(do_debug && get_global_id(0) == 883) {
									int idx = get_global_id(0);
									dbg[0] = (float)(poz-nr_int_nodes);
									dbg[1] = nr;
									dbg[2] = det;
									dbg[3] = inv_det;
									dbg[4] = u;
									dbg[5] = v;
									dbg[6] = t;
									dbg[7] = d2;
									dbg[8] = r2;
								}
							}
						}
						break;
					}
\n
					stack |= mask;
					mask <<= 1;
					poz *= 2;
				}
\n
				while(1) // going up
				{
					mask >>= 1;
					if(!mask) {
						if(do_debug && get_global_id(0) == 883 && cand_tri == 8)
							dbg[1] = 1234.6542f;
						inter_tris[get_global_id(0)] = cand_tri;
						return;
					}
					if(stack & mask) {
						// traverse sibling
						stack &= ~mask;
						mask <<= 1;
						poz++;
						break;
					}
					poz /= 2;
				}
			}
\n
		}
	);

	program = buildCLProgram(kernel_src, context, device, cl_build_options.c_str());
	checkErr(err || !program, "failed to build program");

	{
		CLKernel *kernel = program->createKernel("test", &err);
		checkErr(err || !kernel, "failed to create kernel");

		float ret = 0;
		CLBuffer *d_ret = context->createBuffer(CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
		checkErr(err || !d_ret, "failed to create ret buf");

		queue->writeBuffer(d_ret, 0, sizeof(float), &ret, &err);
		checkErr(err, "failed to init ret");

		ret = -1;

		float comp = 0;
		for(int i = 0; i < (int)ph.int_nodes.size(); ++i) {
			comp += ph.int_nodes[i].r / ph.int_nodes.size();
		}

		kernel->setArgs(d_int_nodes, d_leaves, (int)ph.int_nodes.size(), d_ret, &err);
		checkErr(err, "failed to set kernel arguments");

		queue->runKernel(kernel, Range1D(1,1), &err);
		checkErr(err, "failed to run kernel");

		queue->readBuffer(d_ret, 0, sizeof(float), &ret, &err);
		checkErr(err, "failed to read buffer");

		d_ret->free(&err);
		checkErr(err, "failed to free d_ret");

		kernel->free(&err);
		checkErr(err, "failed to free kernel");

		if(abs(ret - comp) > 1e-5) {
			Y_ERROR << "uploaded data mismatch" << yendl; 
		}
	}
}

void photonIntegratorGPU_t::onSceneUpdate() {

}

CLPlatform photonIntegratorGPU_t::getOpenCLPlatform() {
	CLError err;
	CLMain cl;
	std::list<CLPlatform> platforms = cl.getPlatforms(&err);
	checkErr(err || platforms.empty(), "failed to find platforms\n");
	return *platforms.begin();
}

CLDevice photonIntegratorGPU_t::getOpenCLDevice() {
	CLError err;
	std::list<CLDevice> devices = platform.getDevices(&err);
	checkErr(err || devices.empty(), "failed to find devices for the chosen platform");
	return *devices.begin();
}

extern "C"
{

	YAFRAYPLUGIN_EXPORT void registerPlugin(renderEnvironment_t &render)
	{
		render.registerFactory("photonmappingGPU", photonIntegratorGPU_t::factory);
	}

}

__END_YAFRAY
