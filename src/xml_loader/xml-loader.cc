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

/*void test(scene_t *scene) {
	photonIntegratorGPU_t::DiskVectorType disks;
	photonIntegratorGPU_t::NormalVectorType normals;

	//photonIntegratorGPU_t::generate_points(normals, disks, scene);
	//photonIntegratorGPU_t::build_disk_hierarchy(disks, 0, (int)disks.size());

	// displace the original scene out of view to show only the current scene
	// TODO: create a new scene or remove the meshes from the original scene
	scene_t::objDataArray_t &meshes = scene->getMeshes();
	for(scene_t::objDataArray_t::iterator itr = meshes.begin(); itr != meshes.end(); ++itr) {
		for(std::vector<point3d_t>::iterator p_itr = itr->second.points.begin(); p_itr != itr->second.points.end(); ++p_itr)
			(*p_itr) += point3d_t(3,3,3);
	}

	DiskSceneTessellator tes(scene, 10);

	for(photonIntegratorGPU_t::DiskVectorType::iterator itr = disks.begin(); itr != disks.end(); ++itr) {
		tes.tessellate(itr->mat, itr->c, itr->r, normals[itr->n_idx]);
	}
}*/

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

	//test(scene);
	
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
