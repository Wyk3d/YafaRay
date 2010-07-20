/****************************************************************************
 * 			scene.cc: scene_t controls the rendering of a scene
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
 

#include <core_api/scene.h>
#include <core_api/object3d.h>
#include <core_api/camera.h>
#include <core_api/light.h>
#include <core_api/background.h>
#include <core_api/integrator.h>
#include <core_api/imagefilm.h>
#include <yafraycore/triangle.h>
#include <yafraycore/kdtree.h>
#include <yafraycore/ray_kdtree.h>
#include <yafraycore/timer.h>
#include <yafraycore/scr_halton.h>
#include <utilities/mcqmc.h>
#include <utilities/sample_utils.h>
#ifdef __APPLE__
	#include <sys/sysctl.h>
#endif
#include <iostream>
#include <limits>
#include <sstream>

__BEGIN_YAFRAY

scene_t::scene_t():  volIntegrator(0), camera(0), imageFilm(0), tree(0), vtree(0), background(0), surfIntegrator(0),
					AA_samples(1), AA_passes(1), AA_threshold(0.05), nthreads(1), mode(1), do_depth(false), signals(0)
{
	state.changes = C_ALL;
	state.stack.push_front(READY);
	state.nextFreeID = 1;
	state.curObj = 0;
}

scene_t::~scene_t()
{
	if(tree) delete tree;
	if(vtree) delete vtree;
	std::map<objID_t, objData_t>::iterator i;
	for(i = meshes.begin(); i != meshes.end(); ++i)
	{
		if(i->second.type == TRIM)
			delete i->second.obj;
		else
			delete i->second.mobj;
	}
}

void scene_t::abort()
{
	sig_mutex.lock();
	signals |= Y_SIG_ABORT;
	sig_mutex.unlock();
}

int scene_t::getSignals() const
{
	int sig;
	sig_mutex.lock();
	sig = signals;
	sig_mutex.unlock();
	return sig;
}

void scene_t::getAAParameters(int &samples, int &passes, int &inc_samples, CFLOAT &threshold) const
{
	samples = AA_samples;
	passes = AA_passes;
	inc_samples = AA_inc_samples;
	threshold = AA_threshold;
}

bool scene_t::startGeometry()
{
	if(state.stack.front() != READY) return false;
	state.stack.push_front(GEOMETRY);
	return true;
}

bool scene_t::endGeometry()
{
	if(state.stack.front() != GEOMETRY) return false;
	// in case objects share arrays, so they all need to be updated
	// after each object change, uncomment the below block again:
	// don't forget to update the mesh object iterators!
/*	for(std::map<objID_t, objData_t>::iterator i=meshes.begin();
		 i!=meshes.end(); ++i)
	{
		objData_t &dat = (*i).second;
		dat.obj->setContext(dat.points.begin(), dat.normals.begin() );
	}*/
	state.stack.pop_front();
	return true;
}

bool scene_t::startCurveMesh(objID_t id, int vertices)
{
	if(state.stack.front() != GEOMETRY) return false;
	int ptype = 0 & 0xFF;

	objData_t &nObj = meshes[id];

	//TODO: switch?
	// Allocate triangles to render the curve
	nObj.obj = new triangleObject_t( 2 * (vertices-1) , true, false);
	nObj.type = ptype;
	state.stack.push_front(OBJECT);
	state.changes |= C_GEOM;
	state.orco=false;
	state.curObj = &nObj;

	nObj.points.reserve(2*vertices);
	return true;
}

bool scene_t::endCurveMesh(const material_t *mat, float strandStart, float strandEnd, float strandShape)
{
	if(state.stack.front() != OBJECT) return false;

	// TODO: Check if we have at least 2 vertex...
	// TODO: math optimizations
	
	// extrude vertices and create faces
	std::vector<point3d_t> &points = state.curObj->points;
	float r;	//current radius
	int i;
	point3d_t o,a,b;
	vector3d_t N(0),u(0),v(0);
	int n = points.size();
	// Vertex extruding
	for (i=0;i<n;i++){
		o = points[i];
		if (strandShape < 0)
		{
			r = strandStart + pow((float)i/(n-1) ,1+strandShape) * ( strandEnd - strandStart );
		}
		else
		{
			r = strandStart + (1 - pow(((float)(n-i-1))/(n-1) ,1-strandShape)) * ( strandEnd - strandStart );
		}
		// Last point keep previous tangent plane
		if (i<n-1)
		{
			N = points[i+1]-points[i];
			N.normalize();
			createCS(N,u,v);
		}
		// TODO: thikness?
		a = o - (0.5 * r *v) - 1.5 * r / sqrt(3.f) * u;
		b = o - (0.5 * r *v) + 1.5 * r / sqrt(3.f) * u;
		
		state.curObj->points.push_back(a);
		state.curObj->points.push_back(b);
	}

	// Face fill
	triangle_t tri;
	int a1,a2,a3,b1,b2,b3;
	float su,sv;
	int iu,iv;
	for (i=0;i<n-1;i++){
		// 1D particles UV mapping
		su = (float)i / (n-1);
		sv = su + 1. / (n-1);
		iu = addUV(su,su);
		iv = addUV(sv,sv);
		a1 = i;
		a2 = 2*i+n;
		a3 = a2 +1;
		b1 = i+1;
		b2 = a2 +2;
		b3 = b2 +1;
		// Close bottom
		if (i == 0)
		{
			tri = triangle_t(a1, a3, a2, state.curObj->obj);
			tri.setMaterial(mat);
			state.curTri = state.curObj->obj->addTriangle(tri);
			state.curObj->obj->uv_offsets.push_back(iu);
			state.curObj->obj->uv_offsets.push_back(iu);
			state.curObj->obj->uv_offsets.push_back(iu);
		}
		
		// Fill
		tri = triangle_t(a1, b2, b1, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
		// StrandUV
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iv);
		state.curObj->obj->uv_offsets.push_back(iv);

		tri = triangle_t(a1, a2, b2, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iv);
		
		tri = triangle_t(a2, b3, b2, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iv);
		state.curObj->obj->uv_offsets.push_back(iv);

		tri = triangle_t(a2, a3, b3, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iv);
		
		tri = triangle_t(b3, a3, a1, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
		state.curObj->obj->uv_offsets.push_back(iv);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iu);

		tri = triangle_t(b3, a1, b1, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
		state.curObj->obj->uv_offsets.push_back(iv);
		state.curObj->obj->uv_offsets.push_back(iu);
		state.curObj->obj->uv_offsets.push_back(iv);
		
	}
	// Close top
	tri = triangle_t(i, 2*i+n, 2*i+n+1, state.curObj->obj);
	tri.setMaterial(mat);
	state.curTri = state.curObj->obj->addTriangle(tri);
	state.curObj->obj->uv_offsets.push_back(iv);
	state.curObj->obj->uv_offsets.push_back(iv);
	state.curObj->obj->uv_offsets.push_back(iv);

	state.curObj->obj->setContext(state.curObj->points.begin(), state.curObj->normals.begin() );
	state.curObj->obj->finish();

	state.stack.pop_front();
	return true;
}

bool scene_t::startTriMesh(objID_t id, int vertices, int triangles, bool hasOrco, bool hasUV, int type)
{
	if(state.stack.front() != GEOMETRY) return false;
	int ptype = type & 0xFF;
	if(ptype != TRIM && type != VTRIM && type != MTRIM) return false;
	
	objData_t &nObj = meshes[id];
	switch(ptype)
	{
		case TRIM:	nObj.obj = new triangleObject_t(triangles, hasUV, hasOrco); 
					nObj.obj->setVisibility( !(type & INVISIBLEM) );
					break;
		case VTRIM:
		case MTRIM:	nObj.mobj = new meshObject_t(triangles, hasUV, hasOrco);
					nObj.mobj->setVisibility( !(type & INVISIBLEM) );
					break;
		default: return false;
	}
	nObj.type = ptype;
	state.stack.push_front(OBJECT);
	state.changes |= C_GEOM;
	state.orco=hasOrco;
	state.curObj = &nObj;
	
	//reserve points
	if(hasOrco)
	{
		//decided against shared point vector...
		nObj.points.reserve(2*vertices);
	}
	else
	{
		nObj.points.reserve(vertices);
	}
	return true;
}

bool scene_t::endTriMesh()
{
	if(state.stack.front() != OBJECT) return false;
	
	if(state.curObj->type == TRIM)
	{
		if(state.curObj->obj->has_uv)
		{
			if( state.curObj->obj->uv_offsets.size() != 3*state.curObj->obj->triangles.size() )
			{
				Y_ERROR << "Scene: UV-offsets mismatch!" << yendl;
				return false;
			}
		}
		
		//update mesh context (iterators for points and normals)
		state.curObj->obj->setContext(state.curObj->points.begin(), state.curObj->normals.begin() );
		
		//calculate geometric normals of tris
		state.curObj->obj->finish();
	}
	else
	{
		state.curObj->mobj->setContext(state.curObj->points.begin(), state.curObj->normals.begin() );
		state.curObj->mobj->finish();
	}
	
	state.stack.pop_front();
	return true;
}

void scene_t::setNumThreads(int threads)
{
	nthreads = threads;
	
	if(nthreads == -1) //Automatic detection of number of threads supported by this system, taken from Blender. (DT)
	{
		Y_INFO << "Automatic Detection of Threads: Active." << yendl;

#ifdef WIN32
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		nthreads = (int) info.dwNumberOfProcessors;
#else 
	#	ifdef __APPLE__
		int mib[2];
		size_t len;
		
		mib[0] = CTL_HW;
		mib[1] = HW_NCPU;
		len = sizeof(int);
		sysctl(mib, 2, &nthreads, &len, NULL, 0);
	#	elif defined(__sgi)
		nthreads = sysconf(_SC_NPROC_ONLN);
	#	else
		nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	#	endif
#endif

		Y_INFO << "Number of Threads supported: [" << nthreads << "]." << yendl;
	}
	else
	{
		Y_INFO << "Automatic Detection of Threads: Inactive." << yendl;
	}
	
	Y_INFO << "Using [" << nthreads << "] Threads." << yendl;
}	

/* currently not fully implemented! id=0 means state.curObj, other than 0 not supported yet. */
bool scene_t::smoothMesh(objID_t id, PFLOAT angle)
{
	if( state.stack.front() != GEOMETRY ) return false;
	objData_t *odat;
	if(id)
	{
		std::map<objID_t, objData_t>::iterator it = meshes.find(id);
		if(it == meshes.end() ) return false;
		odat = &(it->second);
	}
	else
	{
		odat = state.curObj;
		if(!odat) return false;
	}
	// cannot smooth other mesh types yet...
	if(odat->type > 0) return false;
	unsigned int i1, i2, i3;
	std::vector<normal_t> &normals = odat->normals;
	std::vector<triangle_t> &triangles = odat->obj->triangles;
	std::vector<point3d_t> &points = odat->points;
	std::vector<triangle_t>::iterator tri;
	if (angle>=180)
	{
		normals.reserve(points.size());
		normals.resize(points.size(), normal_t(0,0,0));
		for (tri=triangles.begin(); tri!=triangles.end(); ++tri)
		{
			i1 = tri->pa;
			i2 = tri->pb;
			i3 = tri->pc;
			// hm somehow it was a stupid idea not allowing calculation with normal_t class
			vector3d_t normal = (vector3d_t)tri->normal;
			normals[i1] = normal_t( (vector3d_t)normals[i1] + normal );
			normals[i2] = normal_t( (vector3d_t)normals[i2] + normal );
			normals[i3] = normal_t( (vector3d_t)normals[i3] + normal );
			tri->setNormals(i1, i2, i3);
		}
		for (i1=0;i1<normals.size();i1++)
			normals[i1] = normal_t( vector3d_t(normals[i1]).normalize() );
		//don't forget to refresh context as the normals memory may be reallocated...or rather will be.
		odat->obj->setContext(odat->points.begin(), odat->normals.begin() );
		odat->obj->is_smooth = true;
	}
	else if(angle>0.1)// angle dependant smoothing
	{
		PFLOAT thresh = fCos(degToRad(angle));
		std::vector<vector3d_t> vnormals;
		std::vector<int> vn_index;
		// create list of faces that include given vertex
		std::vector<std::vector<triangle_t*> > vface(points.size());
		for (tri=triangles.begin(); tri!=triangles.end(); ++tri)
		{
			vface[tri->pa].push_back(&(*tri));
			vface[tri->pb].push_back(&(*tri));
			vface[tri->pc].push_back(&(*tri));
		}
		for(int i=0; i<(int)vface.size(); ++i)
		{
			std::vector<triangle_t*> &tris = vface[i];
			for(std::vector<triangle_t*>::iterator fi=tris.begin(); fi!=tris.end(); ++fi)
			{
				triangle_t* f = *fi;
				bool smooth = false;
				// calculate vertex normal for face
				vector3d_t vnorm(f->normal), fnorm(f->normal);
				for(std::vector<triangle_t*>::iterator f2=tris.begin(); f2!=tris.end(); ++f2)
				{
					if(fi == f2) continue;
					vector3d_t f2norm((*f2)->normal);
					if((fnorm * f2norm) > thresh)
					{
						smooth = true;
						vnorm += f2norm;
					}
				}
				int n_idx = -1;
				if(smooth)
				{
					vnorm.normalize();
					//search for existing normal
					for(unsigned int j=0; j<vnormals.size(); ++j)
					{
						if(vnorm*vnormals[j] > 0.999){ n_idx = vn_index[j]; break; }
					}
					// create new if none found
					if(n_idx == -1)
					{
						n_idx = normals.size();
						vnormals.push_back(vnorm);
						vn_index.push_back(n_idx);
						normals.push_back( normal_t(vnorm) );
					}
				}
				// set vertex normal to idx
				if	   (f->pa == i) f->na = n_idx;
				else if(f->pb == i) f->nb = n_idx;
				else if(f->pc == i) f->nc = n_idx;
				else
				{
					Y_ERROR << "Scene: Mesh smoothing error!" << yendl;
					return false;
				}
			}
			vnormals.clear();
			vn_index.clear();
		}

		odat->obj->setContext(odat->points.begin(), odat->normals.begin() );
		odat->obj->is_smooth = true;
	}
	return true;
}

int scene_t::addVertex(const point3d_t &p)
{
	if(state.stack.front() != OBJECT) return -1;
	state.curObj->points.push_back(p);
	if(state.curObj->type == MTRIM)
	{
		std::vector<point3d_t> &points = state.curObj->points;
		int n = points.size();
		if(n%3==0)
		{
			//convert point 2 to quadratic bezier control point
			points[n-2] = 2.f*points[n-2] - 0.5f*(points[n-3] + points[n-1]);
		}
		return (n-1)/3;
	}
	return state.curObj->points.size()-1;
}

int scene_t::addVertex(const point3d_t &p, const point3d_t &orco)
{
	if(state.stack.front() != OBJECT) return -1;
	state.curObj->points.push_back(p);
	state.curObj->points.push_back(orco);
	return (state.curObj->points.size()-1)/2;
}

bool scene_t::addTriangle(int a, int b, int c, const material_t *mat)
{
	if(state.stack.front() != OBJECT) return false;
	if(state.curObj->type == MTRIM)
	{
		bsTriangle_t tri(3*a, 3*b, 3*c, state.curObj->mobj);
		tri.setMaterial(mat);
		state.curObj->mobj->addBsTriangle(tri);
	}
	else if(state.curObj->type == VTRIM)
	{
		if(state.orco) a*=2, b*=2, c*=2;
		vTriangle_t tri(a, b, c, state.curObj->mobj);
		tri.setMaterial(mat);
		state.curObj->mobj->addTriangle(tri);
	}
	else
	{
		if(state.orco) a*=2, b*=2, c*=2;
		triangle_t tri(a, b, c, state.curObj->obj);
		tri.setMaterial(mat);
		state.curTri = state.curObj->obj->addTriangle(tri);
	}
	return true;
}

bool scene_t::addTriangle(int a, int b, int c, int uv_a, int uv_b, int uv_c, const material_t *mat)
{
	if(!addTriangle(a, b, c, mat)) return false;
	
	if(state.curObj->type == TRIM)
	{
		state.curObj->obj->uv_offsets.push_back(uv_a);
		state.curObj->obj->uv_offsets.push_back(uv_b);
		state.curObj->obj->uv_offsets.push_back(uv_c);
	}
	else
	{
		state.curObj->mobj->uv_offsets.push_back(uv_a);
		state.curObj->mobj->uv_offsets.push_back(uv_b);
		state.curObj->mobj->uv_offsets.push_back(uv_c);
	}
	
	return true;
}

int scene_t::addUV(GFLOAT u, GFLOAT v)
{
	if(state.stack.front() != OBJECT) return false;
	if(state.curObj->type == TRIM)
	{
		state.curObj->obj->uv_values.push_back(uv_t(u, v));
		return (int)state.curObj->obj->uv_values.size()-1;
	}
	else
	{
		state.curObj->mobj->uv_values.push_back(uv_t(u, v));
		return (int)state.curObj->mobj->uv_values.size()-1;
	}
	return -1;
}

bool scene_t::addLight(light_t *l)
{
	if(l != 0)
	{
		lights.push_back(l);
		state.changes |= C_LIGHT;
		return true;
	}
	return false;
}

void scene_t::setCamera(camera_t *cam)
{
	camera = cam;
}

void scene_t::setImageFilm(imageFilm_t *film)
{
	imageFilm = film;
}

void scene_t::setBackground(background_t *bg)
{
	background = bg;
}

void scene_t::setSurfIntegrator(surfaceIntegrator_t *s)
{
	surfIntegrator = s;
	surfIntegrator->setScene(this);
	state.changes |= C_OTHER;
}

void scene_t::setVolIntegrator(volumeIntegrator_t *v)
{
	volIntegrator = v;
	volIntegrator->setScene(this);
	state.changes |= C_OTHER;
}

background_t* scene_t::getBackground() const
{
	return background;
}

triangleObject_t* scene_t::getMesh(objID_t id) const
{
	std::map<objID_t, objData_t>::const_iterator i = meshes.find(id);
	return (i==meshes.end()) ? 0 : i->second.obj;
}

object3d_t* scene_t::getObject(objID_t id) const
{
	std::map<objID_t, objData_t>::const_iterator i = meshes.find(id);
	if(i != meshes.end())
	{
		if(i->second.type == TRIM) return i->second.obj;
		else return i->second.mobj;
	}
	else
	{
		std::map<objID_t, object3d_t *>::const_iterator oi = objects.find(id);
		if(oi != objects.end() ) return oi->second;
	}
	return 0;
}

bound_t scene_t::getSceneBound() const
{
	return sceneBound;
}

void scene_t::setAntialiasing(int numSamples, int numPasses, int incSamples, double threshold)
{
	AA_samples = std::max(1, numSamples);
	AA_passes = numPasses;
	AA_inc_samples = (incSamples > 0) ? incSamples : AA_samples;
	AA_threshold = (CFLOAT)threshold;
}

/*! update scene state to prepare for rendering.
	\return false if something vital to render the scene is missing
			true otherwise
*/
bool scene_t::update()
{
	Y_INFO << "Scene: Mode \"" << ((mode == 0) ? "Triangle" : "Universal" ) << "\"" << yendl;
	if(!camera || !imageFilm) return false;

	if(surfIntegrator)
		surfIntegrator->onSceneUpdate();

	if(state.changes & C_GEOM)
	{
		if(tree) delete tree;
		if(vtree) delete vtree;
		tree = 0, vtree = 0;
		int nprims=0;
		if(mode==0)
		{
			for(std::map<objID_t, objData_t>::iterator i=meshes.begin(); i!=meshes.end(); ++i)
			{
				objData_t &dat = (*i).second;
				if(!dat.obj->isVisible()) continue;
				if(dat.type == TRIM) nprims += dat.obj->numPrimitives();
			}
			if(nprims > 0)
			{
				const triangle_t **tris = new const triangle_t*[nprims];
				const triangle_t **insert = tris;
				for(std::map<objID_t, objData_t>::iterator i=meshes.begin(); i!=meshes.end(); ++i)
				{
					objData_t &dat = (*i).second;
					if(!dat.obj->isVisible()) continue;
					if(dat.type == TRIM) insert += dat.obj->getPrimitives(insert);
				}
				tree = new triKdTree_t(tris, nprims, -1, 1, 0.8, 0.33 /* -1, 1.2, 0.40 */ );
				delete [] tris;
				sceneBound = tree->getBound();
				Y_INFO << "Scene: New scene bound is:" << 
				"(" << sceneBound.a.x << ", " << sceneBound.a.y << ", " << sceneBound.a.z << "), (" <<
				sceneBound.g.x << ", " << sceneBound.g.y << ", " << sceneBound.g.z << ")" << yendl;
			}
			else Y_ERROR << "Scene: Scene is empty..." << yendl;
		}
		else
		{
			for(std::map<objID_t, objData_t>::iterator i=meshes.begin(); i!=meshes.end(); ++i)
			{
				objData_t &dat = (*i).second;
				if(dat.type != TRIM) nprims += dat.mobj->numPrimitives();
			}
			// include all non-mesh objects; eventually make a common map...
			for(std::map<objID_t, object3d_t *>::iterator i=objects.begin(); i!=objects.end(); ++i)
			{
				nprims += i->second->numPrimitives();
			}
			if(nprims > 0)
			{
				const primitive_t **tris = new const primitive_t*[nprims];
				const primitive_t **insert = tris;
				for(std::map<objID_t, objData_t>::iterator i=meshes.begin(); i!=meshes.end(); ++i)
				{
					objData_t &dat = (*i).second;
					if(dat.type != TRIM) insert += dat.mobj->getPrimitives(insert);
				}
				for(std::map<objID_t, object3d_t *>::iterator i=objects.begin(); i!=objects.end(); ++i)
				{
					insert += i->second->getPrimitives(insert);
				}
				vtree = new kdTree_t<primitive_t>(tris, nprims, -1, 1, 0.8, 0.33 /* -1, 1.2, 0.40 */ );
				delete [] tris;
				sceneBound = vtree->getBound();
				Y_INFO << "Scene: New scene bound is:" << yendl <<
				"(" << sceneBound.a.x << ", " << sceneBound.a.y << ", " << sceneBound.a.z << "), (" <<
				sceneBound.g.x << ", " << sceneBound.g.y << ", " << sceneBound.g.z << ")" << yendl;
			}
			else Y_ERROR << "Scene: Scene is empty..." << yendl;
		}
	}
	
	for(unsigned int i=0; i<lights.size(); ++i) lights[i]->init(*this);
	
	if(!surfIntegrator)
	{
		Y_ERROR << "Scene: No surface integrator, bailing out..." << yendl;
		return false;
	}
	
	if(state.changes != C_NONE)
	{
		std::stringstream inteSettings;

		bool success = (surfIntegrator->preprocess() && volIntegrator->preprocess());
		
		inteSettings << surfIntegrator->getName() << " (" << surfIntegrator->getSettings() << ")";
		imageFilm->setIntegParams(inteSettings.str());

		if(!success) return false;
	}
	
	state.changes = C_NONE;
	
	return true;
}

bool scene_t::intersect(const ray_t &ray, surfacePoint_t &sp) const
{
	PFLOAT dis, Z;
	unsigned char udat[PRIM_DAT_SIZE];
	if(ray.tmax<0) dis=std::numeric_limits<PFLOAT>::infinity();
	else dis=ray.tmax;
	// intersect with tree:
	if(mode == 0)
	{
		if(!tree) return false;
		triangle_t *hitt=0;
		if( ! tree->Intersect(ray, dis, &hitt, Z, (void*)&udat[0]) ){ return false; }
		point3d_t h=ray.from + Z*ray.dir;
		hitt->getSurface(sp, h, (void*)&udat[0]);
		sp.origin = hitt;
	}
	else
	{
		if(!vtree) return false;
		primitive_t *hitprim=0;
		if( ! vtree->Intersect(ray, dis, &hitprim, Z, (void*)&udat[0]) ){ return false; }
		point3d_t h=ray.from + Z*ray.dir;
		hitprim->getSurface(sp, h, (void*)&udat[0]);
		sp.origin = hitprim;
	}
	ray.tmax = Z;
	return true;
}

bool scene_t::isShadowed(renderState_t &state, const ray_t &ray) const
{
	
	ray_t sray(ray);
	sray.from += sray.dir * sray.tmin; //argh...kill that!
	sray.time = state.time;
	PFLOAT dis;
	if(ray.tmax<0)	dis=std::numeric_limits<PFLOAT>::infinity();
	else  dis = sray.tmax - 2*sray.tmin;
	if(mode==0)
	{
		triangle_t *hitt=0;
		if(!tree) return false;
		return tree->IntersectS(sray, dis, &hitt);
	}
	else
	{
		primitive_t *hitt=0;
		if(!vtree) return false;
		return vtree->IntersectS(sray, dis, &hitt);
	}
}

bool scene_t::isShadowed(renderState_t &state, const ray_t &ray, int maxDepth, color_t &filt) const
{
	ray_t sray(ray);
	sray.from += sray.dir * sray.tmin; //argh...kill that!
	PFLOAT dis;
	if(ray.tmax<0)	dis=std::numeric_limits<PFLOAT>::infinity();
	else  dis = sray.tmax - 2*sray.tmin;
	filt = color_t(1.0);
	void *odat = state.userdata;
	unsigned char userdata[USER_DATA_SIZE+7];
	state.userdata = (void *)( ((size_t)&userdata[7])&(~7 ) ); // pad userdata to 8 bytes
	bool isect=false;
	if(mode==0)
	{
		triangle_t *hitt=0;
		if(tree) isect = tree->IntersectTS(state, sray, maxDepth, dis, &hitt, filt);
	}
	else
	{
		primitive_t *hitt=0;
		if(vtree) isect = vtree->IntersectTS(state, sray, maxDepth, dis, &hitt, filt);
	}
	state.userdata = odat;
	return isect;
}


bool scene_t::render()
{
	sig_mutex.lock();
	signals = 0;
	sig_mutex.unlock();

	if(!update()) return false;

	bool success = surfIntegrator->render(imageFilm);

	surfIntegrator->cleanup();
	imageFilm->flush();

	return success;
}

//! does not do anything yet...maybe never will
bool scene_t::addMaterial(material_t *m, const char* name) { return false; }

objID_t scene_t::getNextFreeID()
{
	objID_t id;
	id = state.nextFreeID;
	
	//create new entry for object, assert that no ID collision happens:
	if(meshes.find(id) != meshes.end())
	{
		Y_ERROR << "Scene: Object ID already in use!" << yendl;
		return 0;
	}
	
	++state.nextFreeID;
	
	return id;
}

bool scene_t::addObject(object3d_t *obj, objID_t &id)
{
	id = getNextFreeID();
	if( id > 0 )
	{
		//create new triangle object:
		objects[id] = obj;
		return true;
	} 
	else
	{
		return false;
	}
}

__END_YAFRAY
