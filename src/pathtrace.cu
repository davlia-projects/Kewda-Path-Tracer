#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define CACHE 0
#define GROUPED 0
#define MOTION 0
#define BIAS 0
#define ANTIALIASING 0
#define TIMING 0
#define BLOCKSIZE 8

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static PathSegment * dev_cache;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if CACHE == 1
	cudaMalloc(&dev_cache, pixelcount * sizeof(PathSegment));
#endif
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_cache);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];
		
		auto rng = makeSeededRandomEngine(iter, x, 0);
		thrust::uniform_real_distribution<float> uAA(-0.5, 0.5);
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::uniform_int_distribution<int> coin(0, 1);

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.insideObject = false; // for refraction
#if BIAS == 0 && MOTION == 1
		segment.time = u01(rng);
#elif BIAS == 1 && MOTION == 1
		segment.time = u01(rng) > 0.5 ? 1 - pow(u01(rng), 5) : pow(u01(rng), 5); // temporal AA
#elif BIAS == 2 && MOTION == 1
		segment.time = 1 - pow(u01(rng), 3);
#endif
#if ANTIALIASING == 1
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + uAA(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + uAA(rng) - (float)cam.resolution.y * 0.5f)
			);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
#if MOTION == 1
			Geom geom = geoms[i];
			geom.transform = glm::translate(geom.transform, geom.animeT * pathSegment.time);
			geom.transform = glm::rotate(geom.transform, pathSegment.time * geom.animeR.x * (float)PI / 180, glm::vec3(1, 0, 0));
			geom.transform = glm::rotate(geom.transform, pathSegment.time * geom.animeR.y * (float)PI / 180, glm::vec3(0, 1, 0));
			geom.transform = glm::rotate(geom.transform, pathSegment.time * geom.animeR.z * (float)PI / 180, glm::vec3(0, 0, 1));
			geom.inverseTransform = glm::inverse(geom.transform);
			geom.invTranspose = glm::inverseTranspose(geom.transform);
#else
			Geom & geom = geoms[i];
#endif
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shader(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths)
	{
		return;
	}
	
	ShadeableIntersection intersection = shadeableIntersections[idx];
	if (pathSegments[idx].remainingBounces <= 0) {
		pathSegments[idx].color = glm::vec3(0.0f);
		return;
	}
	if (intersection.t > 0.0005f) { // if the intersection exists...
		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			pathSegments[idx].color *= (materialColor * material.emittance);
			pathSegments[idx].remainingBounces = 0; // terminate ray
		}
		else {
			scatterRay(
				pathSegments[idx]
				, pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction
				, intersection.surfaceNormal, material
				, makeSeededRandomEngine(iter, idx, 0)
				);
			pathSegments[idx].remainingBounces--;
		}
	}
	else {
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = 0; // hit nothing
	}
	
}

struct partitionRays
{
	__host__ __device__
	bool operator()(const PathSegment &path)
	{
		return path.remainingBounces > 0;
	}
};
struct material
{
	__host__ __device__
		bool operator()(const ShadeableIntersection &itx1, const ShadeableIntersection &itx2)
	{
		return itx1.materialId < itx2.materialId;
	}
};

// Add the current iteration's output to the overall image
__global__ void gather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (!iterationPath.remainingBounces) {
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
	//const int traceDepth = 3;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = BLOCKSIZE;
#if TIMING == 1
	float total = 0.0f;
	float milliseconds = 0.0f;
	cudaEvent_t start, end;
#endif
#if CACHE == 1
	if (iter > 1) {
		cudaMemcpy(dev_paths, dev_cache, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
	}
#else
	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	const int total_num_paths = num_paths;

	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

#if GROUPED == 1
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
	thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
#endif

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if GROUPED == 1
#if TIMING == 1
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif
		thrust::sort_by_key(
			dev_thrust_intersections
			, dev_thrust_intersections + num_paths
			, dev_thrust_paths
			, material()
			);
#if TIMING == 1
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
#endif
#endif


		// tracing


#if CACHE == 1
		if (iter == 1 && depth == 0) { // cache first ray shot out of each iteration
			cudaMemcpy(dev_cache, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		}
		else if (iter > 1 || depth > 0) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
		}
#else
#if TIMING == 1
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
#if TIMING == 1
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
#endif
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
#if TIMING == 1
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif

		shader << <numblocksPathSegmentTracing, blockSize1d >> > (
		iter,
		num_paths,
		dev_intersections,
		dev_paths,
		dev_materials
		);
#if TIMING == 1
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
#endif

		// Stream compact dev_paths
		auto paths_end = thrust::partition(
			thrust::device
			, dev_paths
			, dev_paths + num_paths
			, partitionRays()
			);
		num_paths = paths_end - dev_paths;
		iterationComplete = depth >= traceDepth || num_paths <= 0;
#if TIMING == 1
		//printf("iteration %d: %d\n", iter, num_paths); // print out the number of rays remaining
		printf("iteration %d: %f\n", iter, total);
#endif
	}

  // Assemble this iteration and apply it to the image
  
	finalGather<<<numBlocksPixels, blockSize1d>>>(total_num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
