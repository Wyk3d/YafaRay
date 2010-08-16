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
#include <stdio.h>

__BEGIN_YAFRAY

void photonIntegratorGPU_t::RayTest::init_test(const diffRay_t &diff_ray)
{
	ray = diff_ray;
	set_test = false;
	tri_test = NULL;
	hit_test = false;
	tri_idx_test = -1;
	t_test = -1.0;
	cand_leaves.clear();
	cand_tris.clear();
}

void photonIntegratorGPU_t::RayTest::test_rays(phRenderState_t &r_state)
{
	state = &r_state;
	std::vector<diffRay_t> &c_rays = state->c_rays;

	for(std::vector<diffRay_t>::iterator itr = c_rays.begin(), next; itr != c_rays.end(); itr = next)
	{
		next = itr;
		++next;

		// init test
		init_test((*itr));

		hit_ref = scene->intersect(ray, sp_ref);

		point3d_t & p_ref = sp_ref.P;
		if(hit_ref) {
			t_ref = (p_ref.x - ray.from.x) / ray.dir.x;
		}

		//test_intersect_sh();			from_cand_leaves();
		//test_intersect_kd();			from_cand_leaves();
		//test_intersect_brute();		from_cand_leaves();
		test_intersect_stored();

		if(!set_test) {
			Y_ERROR << "test was not set" << yendl;
			continue;
		}

		bool failed = false;

		if(!hit_ref)
		{
			if(!hit_test) {
				if(tri_idx_test != -1) {
					Y_ERROR << "ray should not hit but tri_idx " << tri_idx_test << " was returned" << yendl;
					failed = true;
				} else
					continue;
			} else {
				Y_ERROR << "test hit but no ref hit" << yendl; 
				continue;
			}
		} else {
			if(hit_test) {
				float d = (p_ref - p_test).length();
				if(d > 1e-5) {
					Y_ERROR << "d > 1e-5" << yendl;
					failed = true;
				}
				if(sp_ref.origin != tri_test) {
					Y_ERROR << "tri test != tri ref" << yendl;
					failed = true;
				}
			} else
				failed = true;
		}

		if(failed) 
		{
			static bool on = true;
			if(on) list_ray_candidates(true);
		}
	}
	return;
}

void photonIntegratorGPU_t::RayTest::list_ray_candidates(bool exact)
{
	bool found = false;
	for(int i = 0; i < (int)leaves.size(); ++i) {
		PHLeaf &l = leaves[i];
		const triangle_t *tri = prims[l.tri_idx];
		if(tri == sp_ref.origin) {
			float nr = l.n * ray.dir;
			if(nr >= 0) {
				Y_ERROR << "disk on hit triangle not facing ray";
				break;
			}

			vector3d_t pm = l.c - ray.from;
			float t = l.n * pm / nr;
			point3d_t q = ray.from + ray.dir * t;

			float d2 = (q - l.c).lengthSqr();
			float r2 = leaf_radius * leaf_radius;
			if(d2 < r2) {
				if(exact) {
					PFLOAT t;
					unsigned char udat2[PRIM_DAT_SIZE];
					bool good = tri->intersect(ray, &t, udat2);
					if(!good)
						continue;
				}
				Y_ERROR << "ray should have hit disk " << i << yendl;
				found = true;
			}
		}
	}
	if(!found) {
		Y_ERROR << "no candidate disks for the ray" << yendl;
	}
}

void photonIntegratorGPU_t::RayTest::test_intersect_brute()
{
	for(int i = 0; i < (int)leaves.size(); ++i) {
		PHLeaf &l = leaves[i];
		float nr = l.n * ray.dir;
		if(nr >= 0)
			continue;

		vector3d_t pm = l.c - ray.from;
		float t = l.n * pm / nr;
		point3d_t q = ray.from + ray.dir * t;

		float d2 = (q - l.c).lengthSqr();
		float r2 = leaf_radius * leaf_radius;
		if(d2 < r2)
			cand_leaves.push_back(i);
	}
}

void photonIntegratorGPU_t::RayTest::test_intersect_sh()
{
	static int nr_leaves = 0;
	static int nr_int_nodes = 0;
	static int nr_inside = 0;
	static int nr_missed = 0;
	static int nr_invdir = 0;
	static int nr_leaf_cull = 0;

	nr_leaves = 0;
	nr_int_nodes = 0;
	nr_inside = 0;
	nr_missed = 0;
	nr_invdir = 0;
	nr_leaf_cull = 0;

	float t_cand = std::numeric_limits<float>::max();

	unsigned int stack = 0;
	unsigned int mask = 1;
	unsigned int poz = 1;
	while(1) {
		while(1) { // going down
			if(poz < (unsigned int)int_nodes.size()) { // check internal node
				nr_int_nodes++;

				PHInternalNode &n = int_nodes[poz];
				// d(line, center) <= radius ?
				vector3d_t pc = n.c - ray.from;				
				float d2 = (pc ^ ray.dir).lengthSqr();
				float r2 = n.r * n.r;
				if(d2 > r2) {
					nr_missed++;
					break;
				}

				// is p inside sphere ?
				if(pc.lengthSqr() > r2) {
					// exists q on line such that q in sphere
					// is q in ray's direction ?
					// if p outside sphere and the line intersects the sphere
					// then c must lie roughly in the ray's direction
					float pcr = pc * ray.dir;
					if(pcr <= 0) {
						nr_invdir++;
						break;
					}
					// exists t >= pcr >= 0 => found
				} else
					nr_inside++;
				
			} else { // check leaf
				nr_leaves++;

				PHLeaf &l = leaves[poz - int_nodes.size()];
				float nr = l.n * ray.dir;
				if(nr >= 0) {
					nr_leaf_cull++;
					break;
				}

				vector3d_t pm = l.c - ray.from;
				float t = l.n * pm / nr;
				point3d_t q = ray.from + ray.dir * t;
				
				float d2 = (q - l.c).lengthSqr();
				float r2 = leaf_radius * leaf_radius;
				if(d2 < r2) {
					PHTriangle &tri = tris[l.tri_idx];
					vector3d_t edge1, edge2, tvec, pvec, qvec;
					PFLOAT det, inv_det, u, v;
					edge1 = tri.b - tri.a;
					edge2 = tri.c - tri.a;
					pvec = ray.dir ^ edge2;
					det = edge1 * pvec;
					if (/*(det>-0.000001) && (det<0.000001)*/ det == 0.0)
						break;
					inv_det = 1.0 / det;
					tvec = ray.from - tri.a;
					u = (tvec*pvec) * inv_det;
					if (u < 0.0 || u > 1.0)
						break;
					qvec = tvec^edge1;
					v = (ray.dir*qvec) * inv_det;
					if ((v<0.0) || ((u+v)>1.0) )
						break;
					t = edge2 * qvec * inv_det;
					if(t < t_cand) {
						t_cand = t;
						cand_leaves.clear();
						cand_leaves.push_back(poz - int_nodes.size());
					}
				}
				break;
			}

			stack |= mask;
			mask <<= 1;
			poz *= 2;
		}

		while(1) // going up
		{
			mask >>= 1;
			if(!mask)
				return;
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
}

void photonIntegratorGPU_t::RayTest::test_intersect_kd()
{
	static int nr_leaves = 0;
	static int nr_int_nodes = 0;
	static int nr_leaf_cull = 0;
	static int nr_inside = 0;
	static int nr_missed = 0;
	static int nr_invdir = 0;

	nr_leaves = 0;
	nr_int_nodes = 0;
	nr_leaf_cull = 0;
	nr_inside = 0;
	nr_missed = 0;
	nr_invdir = 0;

	unsigned int stack = 0;
	unsigned char depth = 0;
	unsigned int poz = 1;
	float tmax = std::numeric_limits<float>::max();
	float tmin = -std::numeric_limits<float>::max();
	bool found = false;
	bool do_report = false;

	float t_cand = std::numeric_limits<float>::max();

#define REPORT { if(do_report) if(poz < (int)int_nodes.size()) printf("int %d %f %f\n", poz, tmin, tmax); \
		else printf("leaf %d %f %f", poz-int_nodes.size(), tmin, tmax); }

#define GO_LEFT  { \
	poz *= 2; \
	depth++; \
	REPORT \
	continue; }
#define GO_RIGHT { \
	poz = poz * 2 + 1; \
	depth++; \
	REPORT \
	continue; }

	REPORT
	while(1) {
		while(1) // going down
		{
			if(poz < (int)int_nodes.size()) { // check internal node
				nr_int_nodes++;

				PHInternalNode &n = int_nodes[poz];

				// d(line, center) <= radius ?
				vector3d_t pc = n.c - ray.from;				
				float d2 = (pc ^ ray.dir).lengthSqr();
				float r2 = n.r * n.r;
				if(d2 > r2) {
					nr_missed++;
					break;
				}

				float P = ray.from[n.coord];
				float M = n.M;
				float R = ray.dir[n.coord];
				
				if(P < M) {
					if(R <= 0)			//  <-P   M
						GO_LEFT
					float t = (M-P)/R;	//    P-> M
					if(t >= tmax)
						GO_LEFT
					/*if(t <= tmin)
						GO_RIGHT*/
					stack |= (1<<depth);
					tmax = t;
					GO_LEFT
					/* tmax' = tmax
					 if(!found) {
						tmin' = t;
						GO_RIGHT
					}
					*/
				} else {
					if(R >= 0)			//	      M   P->
						GO_RIGHT
					float t = (P-M)/-R; //		  M <-P
					if(t >= tmax)
						GO_RIGHT
					/*if(t <= tmin)
						GO_LEFT*/
					stack |= (1<<depth);
					tmax = t;
					if(tmax < t_ref) {
						//Y_ERROR << "tmax < t_comp" << yendl;
					}
					GO_RIGHT
					/*
					tmax' = tmax
					if(!found) {
						tmin' = t
						GO_LEFT
					}
					*/
				}
			} else { // check leaf
				nr_leaves++;

				PHLeaf &l = leaves[poz - int_nodes.size()];
				float nr = l.n * ray.dir;
				if(nr >= 0) {
					nr_leaf_cull++;
					break;
				}

				vector3d_t pm = l.c - ray.from;
				float t = l.n * pm / nr;
				point3d_t q = ray.from + ray.dir * t;

				float d2 = (q - l.c).lengthSqr();
				float r2 = leaf_radius * leaf_radius;
				if(d2 < r2) {
					PHTriangle &tri = tris[l.tri_idx];
					vector3d_t edge1, edge2, tvec, pvec, qvec;
					PFLOAT det, inv_det, u, v;
					edge1 = tri.b - tri.a;
					edge2 = tri.c - tri.a;
					pvec = ray.dir ^ edge2;
					det = edge1 * pvec;
					if (/*(det>-0.000001) && (det<0.000001)*/ det == 0.0)
						break;
					inv_det = 1.0 / det;
					tvec = ray.from - tri.a;
					u = (tvec*pvec) * inv_det;
					if (u < 0.0 || u > 1.0)
						break;
					qvec = tvec^edge1;
					v = (ray.dir*qvec) * inv_det;
					if ((v<0.0) || ((u+v)>1.0) )
						break;
					t = edge2 * qvec * inv_det;
					if(t < t_cand) {
						t_cand = t;
						cand_leaves.clear();
						cand_leaves.push_back(poz - int_nodes.size());
					}
				}
				break;
			}
		}

		while(1) // going up
		{
			if(depth == 0)
				return;
			depth--;
			
			int mask = (1<<depth);
			if(stack & mask) {
				
				stack &= ~mask;	
				
				int p = poz / 2;
				// move tmin forward
				PHInternalNode &n = int_nodes[p];
				float P = ray.from[n.coord];
				float M = n.M;
				float R = ray.dir[n.coord];
				float t = (M-P)/R;
				if(t < tmin) {
					static bool r = false;
					if(r) Y_ERROR << "t < tmin" << yendl;
				}
				tmin = t;
				if(tmin > t_ref) {
					static bool r = false;
					if(r) Y_ERROR << "tmin > t_comp" << yendl;
				}

				// revert tmax
				// when was the last time tmax was set ?
				while(1)
				{
					mask >>= 1;
					p /= 2;
					if(!mask) {
						tmax = std::numeric_limits<float>::max();
						break;
					} else if(stack & mask) {
						PHInternalNode &n = int_nodes[p];
						float P = ray.from[n.coord];
						float M = n.M;
						float R = ray.dir[n.coord];
						float t = (M-P)/R;
						if(t < tmax) {
							static bool r = false;
							if(r) Y_ERROR << "t < tmax" << yendl;
						}
						tmax = t;
						break;
					}
				}
				
				// traverse sibling
				depth++;
				poz += -(poz % 2) * 2 + 1;
				REPORT
				break;
			}
			
			poz /= 2;
			REPORT
		}
	}
}

void photonIntegratorGPU_t::RayTest::test_intersect_stored()
{
	if(ray.idx == -1) {
		set_test = true;
		hit_test = false;
		return;
	}
	if(ray.idx < -1 || ray.idx >= (int)state->inter_tris.size())
	{
		Y_ERROR << "ray_idx = " << ray.idx << yendl;
		return;
	}
	from_tri_idx(state->inter_tris[ray.idx]);
}

void photonIntegratorGPU_t::RayTest::from_cand_leaves()
{
	if(cand_leaves.empty()) {
		set_test = true;
		hit_test = false;
		return;
	}

	float tmin = std::numeric_limits<float>::max();
	int idx_min = 0;
	for(int i = 0; i < (int)cand_leaves.size(); ++i) {
		int idx = cand_leaves[i];
		PHLeaf &l = leaves[idx];
		const triangle_t *tri = prims[l.tri_idx];

		PFLOAT t;
		unsigned char udat2[PRIM_DAT_SIZE];
		bool good = tri->intersect(ray, &t, udat2);
		if(good) {
			if(t < tmin) {
				tmin = t;
				idx_min = idx;
			}
		}
	}

	from_leaf_idx(idx_min);
}


void photonIntegratorGPU_t::RayTest::from_cand_tris()
{

}

void photonIntegratorGPU_t::RayTest::from_leaf_idx(int leaf_idx)
{
	if(leaf_idx < 0 || leaf_idx >= (int)leaves.size()) {
		Y_ERROR << "leaf_idx = " << leaf_idx << yendl;
		return;
	}

	from_tri_idx(leaves[leaf_idx].tri_idx);
}

void photonIntegratorGPU_t::RayTest::from_tri_idx(int tri_idx)
{
	if(tri_idx < -1) {
		Y_ERROR << "tri_idx = " << tri_idx << yendl;
		return;
	}
	if(tri_idx >= (int)prims.size()) {
		Y_ERROR << "tri_idx = " << tri_idx << yendl;
		return;
	}
	if(tri_idx == -1)
	{
		hit_test = false;
		set_test = true;
		return;
	}

	tri_idx_test = tri_idx;

	tri_test = prims[tri_idx];
	unsigned char udat2[PRIM_DAT_SIZE];
	hit_test = tri_test->intersect(ray, &t_test, udat2);
	set_test = true;
	if(hit_test) {
		p_test = ray.from + t_test * ray.dir;
	}
}

void photonIntegratorGPU_t::RayTest::benchmark_ray_count(phRenderState_t &r_state)
{
	state = &r_state;
	std::vector<diffRay_t> &c_rays = state->c_rays;
	int nr_rays = c_rays.size();

	int AAsamples, AApasses, AAinc_samples;
	CFLOAT AAthreshold;
	scene->getAAParameters(AAsamples, AApasses, AAinc_samples, AAthreshold);

	int width = scene->getCamera()->resX();
	int height = scene->getCamera()->resY();

	if(nr_rays != width * height * AAsamples) {
		Y_ERROR << "tile_size * AAsamples != width * height * AAsamples" << yendl;
		return;
	}

	int max_size = std::max(width, height);

	Y_INFO << yendl << "benchmarking image of size " << width << " * " << height << " * " << AAsamples << " samples" << yendl; 

	surfacePoint_t sp;
	clock_t start, end;

	int max_tile_size = 1;
	while(max_tile_size <= max_size) max_tile_size *= 2;

	//for(int tile_size = pi.ph_benchmark_min_tile_size; tile_size <= max_tile_size; tile_size *= 2) 
	for(int tile_size = max_tile_size; tile_size >= pi.ph_benchmark_min_tile_size; tile_size /= 2)
	{
		int ts2 = tile_size * tile_size * AAsamples;

		// TODO: reorder state->ph_rays/c_rays for locality

		progressBar_t *pb = new ConsoleProgressBar_t(80);
		pb->init(128);
		int pbStep = std::max(1, nr_rays/128); 
		int nextStep = pbStep;

		start = clock();
		for(int i = 0; i < nr_rays; i+=ts2)
		{
			int j = std::min(i+ts2,nr_rays);
			pi.intersect_rays(*state, state->ph_rays, i, j, state->inter_tris);
			for(int k = i; k < j; k++) {
				diffRay_t &pRay = state->c_rays[k];
				pRay.idx = k;
				pi.getSPforRay(pRay, *state, sp);
			}

			if(i >= nextStep) {
				pb->update();
				nextStep += pbStep;
			}
		}
		end = clock();
		pb->done();
		delete pb;
		float dt1 = ((float)(end-start)/CLOCKS_PER_SEC);
		
		pb = new ConsoleProgressBar_t(80);
		pb->init(128);
		nextStep = pbStep;
		start = clock();
		for(int i = 0; i < nr_rays; i+=ts2)
		{
			int j = std::min(i+ts2,nr_rays);
			for(int k = i; k < j; k++)
				scene->intersect(state->c_rays[k], sp);

			if(i >= nextStep) {
				pb->update();
				nextStep += pbStep;
			}
		}
		end = clock();
		pb->done();
		delete pb;
		float dt2 = ((float)(end-start)/CLOCKS_PER_SEC);

		Y_INFO << "tile size " << tile_size << ": ";
		Y_INFO << dt1 << "s (" << nr_rays / dt1 << " rays/sec) / ";
		Y_INFO << dt2 << "s (" << nr_rays / dt2 << " rays/sec)" << yendl;
	}
}

__END_YAFRAY
