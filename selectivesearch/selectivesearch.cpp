// Original code implementation (c) Copyright (c) 2014 The MatConvNet team.
// https://github.com/jamt9000/matconvnet/blob/selectivesearch/COPYING

#include "selectivesearch.hpp"
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <limits>
#include <vector>
#include <iostream>
#include <time.h>
#include <stdio.h>

static int const nchannels = 3;
static int const timing = false; // print times

// temporary stuff for debugging
template <typename T>
void savePPM(T& data, int width, int height, const char * fn, float minv=0.f, float maxv=1.f)
{
  FILE* f = fopen(fn, "w");
  fprintf(f, "P6\n%d %d\n255\n", width, height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float v = data[height*j + i];
      v = (v - minv)/(maxv - minv);
      unsigned char b = (unsigned char) std::max(0.f, std::min(255.f, 255.f * v));
      fwrite(&b,1,1,f);
      fwrite(&b,1,1,f);
      fwrite(&b,1,1,f);
    }
  }
  fclose(f);
}

template <typename T>
void saveCSV(T& data, int width, int height, const char * fn)
{
  FILE* f = fopen(fn, "w");
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      float v = data[height*j + i];
      fprintf(f, "%f", v);
      if (j != width - 1) fprintf(f, ",");
    }
    fprintf(f, "\n");

  }
  fclose(f);
}


/* Selective search from
* J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, and A.W.M. Smeulders.
* Selective Search for Object Recognition
* IJCV, 2013.
* */

/* Initial segmentation based on
 * Felzenszwalb, P. F., & Huttenlocher, D. P. (2004).
 * Efficient graph-based image segmentation. IJCV
 * */

// See https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Disjoint-set_forests
static int findHead(int vert, std::vector<int>& connectionMap)
{
  assert(vert < connectionMap.size());
  int h = vert;
  // Traverse graph until we find vertex connected to itself
  while (connectionMap[h] != h) {
    h = connectionMap[h];
  }
  return h;
}


static int joinRegion(int vertexAHead, int vertexBHead, std::vector<int>& connectionMap,
               std::vector<int>& sizeMap, std::vector<int>& rankMap)
{
  int newHead = vertexBHead;
  int oldHead = vertexAHead;
  if (rankMap[vertexAHead] >= rankMap[vertexBHead]) {
    newHead = vertexAHead;
    oldHead = vertexBHead;
  }
  connectionMap[oldHead] = newHead;
  sizeMap[newHead] += sizeMap[oldHead];
  rankMap[newHead] += (int) (rankMap[vertexAHead] == rankMap[vertexBHead]);

  return newHead;
}

static void compressPath(int vert, std::vector<int>& connectionMap)
{
  int h = vert;
  std::vector<int> path;
  while (connectionMap[h] != h) {
    path.push_back(h);
    h = connectionMap[h];
  }
  for (int i = 0; i < path.size(); ++i) {
    connectionMap[path[i]] = h;
  }
}

static bool isBoundary(float edgeWeight, int a, int b, std::vector<float>& thresholds)
{
  return edgeWeight > (std::min)(thresholds[a], thresholds[b]);
}

class SortIndices
{
  std::vector<float>& v;
public:
  SortIndices(std::vector<float>& v) : v(v) {}
  bool operator() (const int& a, const int& b) const { return v[a] < v[b]; }
};

/* --------- PATCHED ------------------ */

template <typename Out, typename In>
static void filter1DSymmetric(Out& output, In& data, std::vector<float>& filter,
                       int height, int width, int channel, bool filterRows)
{
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
        // height*width*channel + height*j + i
        // channel + 3*j + 3*width*i
      float s = filter[0] * data[channel + 3*j + 3*width*i];
      for (int f = 1; f < filter.size(); ++f) {
        int iprev = filterRows ? (std::max)(0, i - f) : i;
        int inext = filterRows ? (std::min)((int)height - 1, i + f) : i;
        int jprev = filterRows ? j : (std::max)(0, j - f);
        int jnext = filterRows ? j : (std::min)((int)width - 1, j + f);
        s += data[channel + 3*jprev + 3*width*iprev] * filter[f];
        s += data[channel + 3*jnext + 3*width*inext] * filter[f];
      }
      output[channel + 3*j + 3*width*i] = s;
    }
  }
}

template <typename Out, typename In>
static void filter1D(Out& output, In& data, std::vector<float>& filter,
              int height, int width, int channel, bool filterRows)
{
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      float s = 0.0f;
      for (int f = 0; f < filter.size(); ++f) {
        int offs = f - filter.size()/2;
        int ioff = filterRows ? (std::min)((int)height - 1, (std::max)(0, i + offs)) : i;
        int joff = filterRows ? j : (std::min)((int)width - 1, (std::max)(0, j + offs));

        // conv2 style boundary handling
        //int ioff = filterRows ? i + offs : i;
        //int joff = filterRows ? j : j + offs;
        //if (ioff >= height || ioff < 0 || joff >= width || joff < 0) continue;
        s += data[channel + 3*joff + 3*width*ioff] * filter[f];
      }
      output[channel + 3*j + 3*width*i] = s;
    }
  }
}

static void angleGrad(std::vector<float>& out, float const *smoothed, int height, int width, int channel, float phi)
{
  // Angled derivative
  // First smooth with gaussian then take derivative with [-1,0,1]
  float sinp = 0.5f * sinf(phi);
  float cosp = 0.5f * cosf(phi);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      // channel + 3*j + 3*width*i
      float top = smoothed[channel + 3*width*((std::max)(i-1, 0)) + j * 3];
      float bot = smoothed[channel + 3*width*((std::min)(i+1, height-1)) + j * 3];
      float left = smoothed[channel + 3*width*i + (std::max)(j-1, 0) * 3];
      float right = smoothed[channel + 3*width*i + (std::min)(j+1,width-1) * 3];
      out[i + j * height] = sinp*(top-bot)+cosp*(left-right);
    }
  }
}

static void gaussianBlur(float *output, float const *data, int height, int width)
{
  // we only compute centre and positive side of gaussian, since it is symmetric
  // so effectively a 9x9 filter
  int const filterVectorSize = 5;
  float const sigma = 0.8f;

  // 1D gaussian filter (non-negative half)
  std::vector<float> filter(filterVectorSize, 0);
  float total = 0.0f;
  for (int f = 0; f < filterVectorSize; ++f) {
    filter[f] = expf(-0.5f*(f/sigma)*(f/sigma));
    total += filter[f];
  }
  total = total * 2 - filter[0];
  // Normalise
  for (int f = 0; f < filterVectorSize; ++f) {
    filter[f] /= total;
  }

  std::vector<float> firstPass(width*height*3, 0);

  // Gaussian blur (2 passes for rows & cols)
  for (int c = 0; c < nchannels; ++c) {
    filter1DSymmetric(firstPass, data, filter, height, width, c, /*filterRows=*/true);
    filter1DSymmetric(output, firstPass, filter, height, width, c, /*filterRows=*/false);
  }

}

static void findNeighboursAndRectangles(std::vector<bool>& neighbours, int& nRegions, std::vector<int>& rects, int height, int width, std::vector<int>& reindexed)
{
  // Find neighbour matrix
  // And store [mini minj maxi maxj] for each region
  neighbours.clear();
  neighbours.resize(nRegions * nRegions, false);
  rects.clear();
  rects.resize(nRegions * 4, -1);
  for (int j = 0; j < width; ++j) {
    int regionAbove;
    for (int i = 0 ; i < height; ++i) {
      int region = reindexed[height * j + i];
      if (i > 0) {
        neighbours[nRegions * region + regionAbove] = true;
        neighbours[nRegions * regionAbove + region] = true;
      }
      regionAbove = region;

      if (j > 0) {
        int regionLeft = reindexed[height * (j-1) + i];
        neighbours[nRegions * region + regionLeft] = true;
        neighbours[nRegions * regionLeft + region] = true;
      }

      rects[region * 4 + 0] = rects[region * 4 + 0] == -1 ? i : (std::min)(i, rects[region * 4 + 0]);
      rects[region * 4 + 1] = rects[region * 4 + 1] == -1 ? j : (std::min)(j, rects[region * 4 + 1]);
      rects[region * 4 + 2] = (std::max)(i, rects[region * 4 + 2]);
      rects[region * 4 + 3] = (std::max)(j, rects[region * 4 + 3]);
    }
  }
}

static void initialSegmentation(int *output, int& nRegions, std::vector<int>& sizes,
                                std::vector<bool>& neighbours, std::vector<int>& rects,
                                float *data, int height, int width, float threshConst, int minSize)
{

  clock_t initSegSetup = clock();

  // Initialise graph
  int const edgesPerVertex = 4;
  int nVerts = height*width;
  int nEdges = nVerts*edgesPerVertex;
  // Create map of edge weights, where for each vertex (=pixel) (i,j)
  // we store the weight to the vertex (i+ioffs[d], j+joffs[d]),
  // arranged spatially with four "planes" for d=0,1,2,3 and the
  // weight being infinity when the "to" pixel would be out of bounds
  std::vector<float> edgeWeights(nEdges, std::numeric_limits<float>::infinity());
  int ioffs[] = {0, 1, 1, 1};
  int joffs[] = {1, 0, 1, -1};
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      for (int d = 0; d < edgesPerVertex; ++d) {
        int di = i + ioffs[d];
        int dj = j + joffs[d];
        if (di > 0 && di < height && dj > 0 && dj < width) {
            // channel + 3*j + 3*width*i
          float diff0 = data[0 + 3*j + 3*width*i] - data[0 + 3*dj + 3*width*di];
          float diff1 = data[1 + 3*j + 3*width*i] - data[1 + 3*dj + 3*width*di];
          float diff2 = data[2 + 3*j + 3*width*i] - data[2 + 3*dj + 3*width*di];
          // Cludge since the original algorithm expects uint8 image in range 0-255
          // and the threshold is based on that
          diff0 *= 255.f; diff1 *= 255.f; diff2 *= 255.f;
          float distance = sqrtf(diff0*diff0 + diff1*diff1 + diff2*diff2);
          edgeWeights[height*width*d + height*j + i] = distance;
        }
      }
    }
  }
  if(timing) printf("initSegSetup %f\n", (clock() - initSegSetup)/(double)CLOCKS_PER_SEC);

  clock_t initSegSort = clock();

  // Sort by edge weight
  std::vector<int> edgeIndices(edgeWeights.size());
  for (int i = 0; i < edgeIndices.size(); ++i) {
    edgeIndices[i] = i;
  }

  std::sort(edgeIndices.begin(), edgeIndices.end(), SortIndices(edgeWeights));

  if(timing) printf("initSegSort %f\n", (clock() - initSegSort)/(double)CLOCKS_PER_SEC);

  clock_t initSegMain = clock();

  std::vector<float> thresh(width*height);
  for (int i = 0; i < thresh.size(); ++i) {
    thresh[i] = threshConst;
  }

  std::vector<int> connectionMap(height * width);
  for (int i = 0; i < nVerts; ++i) {
    connectionMap[i] = i;
  }

  std::vector<int> sizeMap(nVerts, 1);
  std::vector<int> rankMap(nVerts, 0);


  for (int i = 0; i < edgeIndices.size(); ++i) {
    float weight = edgeWeights[edgeIndices[i]];
    int vertexA = edgeIndices[i] % nVerts;
    int d = edgeIndices[i] / nVerts;
    int vertexB = vertexA + height * joffs[d] + ioffs[d];

    if (weight == std::numeric_limits<float>::infinity()) {
      break;
    }

    int vertexAHead = findHead(vertexA, connectionMap);
    int vertexBHead = findHead(vertexB, connectionMap);

    if (vertexAHead != vertexBHead && !isBoundary(weight, vertexAHead, vertexBHead, thresh)) {
      // Join components
      // mexPrintf("Joining %i and %i \n", vertexAHead, vertexBHead);
      int newHead = joinRegion(vertexAHead, vertexBHead, connectionMap, sizeMap, rankMap);
      thresh[newHead] = weight + threshConst/sizeMap[newHead];
      compressPath(vertexA, connectionMap);
      compressPath(vertexB, connectionMap);
    }
  }

  if(timing) printf("initSegMainLoop %f\n", (clock() - initSegMain)/(double)CLOCKS_PER_SEC);

  clock_t initSegPost = clock();

  // Remove regions smaller than minSize
  for (int i = 0; i < edgeIndices.size(); i++) {
    float weight = edgeWeights[edgeIndices[i]];
    int vertexA = edgeIndices[i] % nVerts;
    int d = edgeIndices[i] / nVerts;
    int vertexB = vertexA + height * joffs[d] + ioffs[d];

    if (weight == std::numeric_limits<float>::infinity()) {
      break;
    }

    int headA = findHead(vertexA, connectionMap);
    int headB = findHead(vertexB, connectionMap);
    if (headA != headB && (sizeMap[headA] < minSize || sizeMap[headB] < minSize)) {
      joinRegion(headA, headB, connectionMap, sizeMap, rankMap);
    }
  }

  // Compact indices
  std::vector<int> reindexed(nVerts, -1);
  sizes.clear();


  int newIndex = 0;
  for (int i = 0; i < nVerts; ++i) {
    int head = findHead(i, connectionMap);
    if (reindexed[head] == -1) {
      reindexed[head] = newIndex;
      sizes.push_back(sizeMap[head]);
      newIndex++;
    }
    reindexed[i] = reindexed[head];
  }

  nRegions = newIndex;

  findNeighboursAndRectangles(neighbours, nRegions, rects, height, width, reindexed);

  for (int i = 0; i < nVerts; ++i) {
    output[i] = reindexed[i];
  }

  if(timing) printf("initSegPost %f\n", (clock() - initSegPost)/(double)CLOCKS_PER_SEC);

  //saveCSV(neighbours, nRegions, nRegions, "/tmp/neighbours.csv");

}

static void computeColourHistogram(std::vector<float>& histOut, float const *image, int const *segmentedImage, int height, int width, int nRegions)
{
  float const maxHistValue = 1.0f;

  // Number of histogram bins
  int const nBins = 25;
  // Size of the concatenated histograms over channels (should be 75)
  int const descriptorSize = nchannels * nBins;

  histOut.clear();
  histOut.resize(nRegions * descriptorSize, 0.f);

  for (int c = 0; c < nchannels; ++c) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          int region = (int) segmentedImage[i];
          // channel + 3*j + 3*width*i
          float value = image[c + 3*j + 3*width*i];

          int bin = std::min(nBins - 1, (int) ceil(value * (nBins - 1) / maxHistValue - 0.5f));
          histOut[region * descriptorSize + c * nBins + bin]++;
      }
    }
  }

  // Normalise
  for (int r = 0; r < nRegions; ++r) {
    float sum = 0.0f;
    for(int i = 0; i < descriptorSize; ++i) {
      sum += histOut[r * descriptorSize + i];
    }
    if (sum != 0.0f) {
      for(int i = 0; i < descriptorSize; ++i) {
        histOut[r * descriptorSize + i] /= sum;
      }
    }
  }

//  // Print hists for region 18
//  for (int x=0; x < descriptorSize; ++x) {
//    printf("%f ", histOut[18 * descriptorSize + x]);
//  }printf("\n");

}

static void computeTextureHistogram(std::vector<float>& histOut, float const *image, float const *smoothImage,
                                    int const *segmentedImage, int height, int width, int nRegions)
{
  // "maximum" magnitude of the gradient (why?)
  float const maxHistValue = 0.43f;

  // Number of image derivative histograms per channel
  int const nHists = 8;
  // Number of histogram bins
  int const nBins = 10;
  // Size of the concatenated histograms over channels (should be 240)
  int const descriptorSize = nchannels * nHists * nBins;

  histOut.clear();
  histOut.resize(nRegions * descriptorSize, 0);

  // Layout:
  // Region 1: [channel1: [hist1,...,hist8]] [channel2: [hist1,...,hist8]] [channel3: [hist1,...,hist8]]
  // Region 2: [channel1: [hist1,...,hist8]] [channel2: [hist1,...,hist8]] [channel3: [hist1,...,hist8]]
  // ...

  // Axis-aligned gaussian derivative
  // Gaussian 2D derivative filter is separable outer product of 1D gaussian and 1D gaussian derivative
  std::vector<float> gaussian;
  std::vector<float> gaussianDeriv;
  float sum = 0.0f;
  for (int x=-4; x <= 4; ++x) {
    float g = expf(-0.5f*(x/0.8f)*(x/0.8f));
    sum += g;
    gaussian.push_back(g);
    gaussianDeriv.push_back(x);
  }
  for (int f=0; f <= gaussian.size(); ++f) {
    gaussian[f] /= sum;
    gaussianDeriv[f] = -gaussianDeriv[f] * gaussian[f] / (0.8f*0.8f);
  }

  std::vector<float> pass1(height * width * nchannels);
  std::vector<float> gradIm(height * width * nchannels);

  for (int c = 0; c < nchannels; ++c) {

    for (int histIdx = 0; histIdx < nHists/2; ++histIdx) {

      switch (histIdx) {
      case 0:
        // Y dir
        filter1D(pass1, image, gaussianDeriv, height, width, c, true);
        filter1D(gradIm, pass1, gaussian, height, width, c, false);
        //if(c==0) saveCSV(gradIm, width, height, "/tmp/Y.csv");
        break;
      case 1:
        // 135 deg
        angleGrad(gradIm, smoothImage, height, width, c, 135.f * (M_PI/180.f));
        //anigauss(&image[c*width*height], &gradIm[c*width*height], height, width, 0.8f,  0.8f, 135.0-90.0, 1, 0);
        //if(c==0) saveCSV(gradIm, width, height, "/tmp/XY.csv");
        break;
      case 2:
        // X dir
        filter1D(pass1, image, gaussianDeriv, height, width, c, false);
        filter1D(gradIm, pass1, gaussian, height, width, c, true);
        //if(c==0) saveCSV(gradIm, width, height, "/tmp/X.csv");
        break;
      case 3:
        // 45 deg
        angleGrad(gradIm, smoothImage, height, width, c, 45.f * (M_PI/180.f));
        //anigauss(&image[c*width*height], &gradIm[c*width*height], height, width, 0.8f,  0.8f,  45.0-90.0, 1, 0);
        //if(c==0) saveCSV(gradIm, width, height, "/tmp/YX.csv");
        break;
      }

      for (int i = 0; i < height * width; ++i) {
        int region = (int) segmentedImage[i];
        float value = gradIm[c*width*height + i];

        // Negative component
        float nvalue = (std::max)(0.f, -value);
        int bin = (std::min)(nBins - 1, (int) ceil(nvalue * (nBins - 1) / maxHistValue - 0.5f));
        assert(bin >= 0 && bin < nBins);
        assert(region * descriptorSize + c * nHists * nBins + histIdx * nBins + bin < histOut.size());
        histOut[region * descriptorSize + c * nHists * nBins + histIdx * nBins + bin]++;

        // Positive component
        float pvalue = (std::max)(0.f, value);
        int pHistIdx = histIdx + 4;
        bin = (std::min)(nBins - 1, (int) ceil(pvalue * (nBins - 1) / maxHistValue - 0.5f));
        histOut[region * descriptorSize + c * nHists * nBins + pHistIdx * nBins + bin]++;

        //        if (histIdx == 0 && region == 18) {
        //          printf("%f -> %d\n", pvalue, bin);
        //        }
      }
    }
  }

  //  // Print hists for region 63
  //  for (int x=0; x < 80; ++x) {
  //    printf("%d ", (int) histOut[63 * descriptorSize + x]);
  //  }printf("\n");

  // Normalise
  for (int r = 0; r < nRegions; ++r) {
    float sum = 0.0f;
    for(int i = 0; i < descriptorSize; ++i) {
      sum += histOut[r * descriptorSize + i];
    }
    if (sum != 0.0f) {
      for(int i = 0; i < descriptorSize; ++i) {
        histOut[r * descriptorSize + i] /= sum;
      }
    }
  }

  //savePPM(gradIm, width, height, "/tmp/out.ppm", -0.43, 0.43);
}

static float similarity(int ri, int rj, int nRegions, int imSize, std::vector<float>& histColour, std::vector<float>& histTexture,
                 std::vector<int>& regionSizes, std::vector<int>& rects, int similarityFlags)
{
  float sim = 0.0f;
  int nSimMeasures = 0;

  // Colour - histogram intersection
  if (similarityFlags & vl::SIM_COLOUR) {
    float colour = 0.0f;
    int hsize = histColour.size()/nRegions;
    for (int i = 0; i < hsize; ++i) {
      colour += (std::min)(histColour[ri * hsize + i], histColour[rj * hsize + i]);
    }
    sim += colour;
    nSimMeasures++;
  }

  // Texture
  if (similarityFlags & vl::SIM_TEXTURE) {
    float texture = 0.0f;
    int tsize = histTexture.size()/nRegions;
    for (int i = 0; i < tsize; ++i) {
      texture += (std::min)(histTexture[ri * tsize + i], histTexture[rj * tsize + i]);
    }
    sim += texture;
    nSimMeasures++;
  }

  // Size
  if (similarityFlags & vl::SIM_SIZE) {
    float size = (imSize - regionSizes[ri] - regionSizes[rj]) / (float) imSize;
    sim += size;
    nSimMeasures++;
  }

  // Fill
  if (similarityFlags & vl::SIM_FILL) {
    int h = (std::max)(rects[ri*4+2], rects[rj*4+2]) - (std::min)(rects[ri*4], rects[rj*4]) + 1;
    int w = (std::max)(rects[ri*4+3], rects[rj*4+3]) - (std::min)(rects[ri*4+1], rects[rj*4+1]) + 1;
    float fill = 1.f - (h*w - regionSizes[ri] - regionSizes[rj])/((float) imSize);
    sim += fill;
    nSimMeasures++;
  }

  sim /= nSimMeasures;

  assert(sim >= 0.f && sim <= 1.f);

  return sim;
}

static void mergeRegions(std::vector<int>& mergedRegions, int nRegions, int imSize, std::vector<bool>& neighbours,
                         std::vector<float> histColour, std::vector<float> histTexture,
                         std::vector<int> regionSizes, std::vector<int> rects, int similarityFlags)
{
  // The indices of neighbouring regions ri and rj are stored in
  // regioni and regionj with the similarity associated with the edge
  // between them stored in similarities.
  std::vector<float> similarities;
  std::vector<int> regioni;
  std::vector<int> regionj;

  // Create list of similarities between regions
  // using lower triangle of neighbour matrix
  for (int r = 0; r < nRegions; ++r) {
    for (int c = 0; c < r; ++c) {
      if(neighbours[c * nRegions + r]) {
          similarities.push_back(similarity(r, c, nRegions, imSize, histColour,
                                            histTexture, regionSizes, rects, similarityFlags));
          regioni.push_back(r);
          regionj.push_back(c);
      }
    }
  }

  int nInitRegions = nRegions;
  int histColourDescSize = histColour.size()/nRegions;
  int histTexDescSize = histTexture.size()/nRegions;

  // Iteratively group the 2 most similar regions
  // until we end up with a single region that is the
  // whole image
  while (true) {
    int newRegionIdx = regionSizes.size();
    float maxSim = -1.f;
    int maxIdx = -1;
    int nSims = 0;

    // Find two most similar regions
    for (int i = 0; i < similarities.size(); ++i) {
      if (similarities[i] > maxSim) {
        maxSim = similarities[i];
        maxIdx = i;
      }
      if (similarities[i] != -1) nSims++;
    }

    //printf("%d rects, %d sims\n", rects.size(), nSims);

    // finish if we have grouped all regions
    if (maxIdx == -1) {
      assert (nSims == 0);
      assert (regionSizes.size() == nInitRegions * 2 - 1);
      break;
    }

    // "delete" the edge between the two regions
    similarities[maxIdx] = -1;

    // Merge regions, combining their descriptors (size, rectangles, histograms)
    // and updating the relevant vectors
    int ri = regioni[maxIdx];
    int rj = regionj[maxIdx];
    //printf("Merging %d %d (%f)\n", ri, rj, maxSim);

    int newSize = regionSizes[ri] + regionSizes[rj];
    regionSizes.push_back(newSize);

    int mini = (std::min)(rects[ri*4], rects[rj*4]);
    int minj = (std::min)(rects[ri*4+1], rects[rj*4+1]);
    int maxi = (std::max)(rects[ri*4+2], rects[rj*4+2]);
    int maxj = (std::max)(rects[ri*4+3], rects[rj*4+3]);
    rects.push_back(mini);
    rects.push_back(minj);
    rects.push_back(maxi);
    rects.push_back(maxj);

    for (int i = 0; i < histColourDescSize; ++i) {
      histColour.push_back((regionSizes[ri] * histColour[ri * histColourDescSize + i]
                           + regionSizes[rj] * histColour[rj * histColourDescSize + i])/ (float) newSize);
    }

    for (int i = 0; i < histTexDescSize; ++i) {
      histTexture.push_back((regionSizes[ri] * histTexture[ri * histTexDescSize + i]
                           + regionSizes[rj] * histTexture[rj * histTexDescSize + i])/ (float) newSize);
    }

    nRegions++;

    // Remove similarities involving the neighbours of the old regions (ri, rj)
    // that have been merged and add new similarities between the merged region
    // (identified by newRegionIdx) and those neighbours (neighbourRegion).
    // Use a bool vector to avoid adding duplicate edges.
    std::vector<bool> edgeAdded(nInitRegions * 2); // nInitRegions * 2 is upper bound

    // Vector grows within loop, but only loop over
    // the initial elements
    int simInitSize = similarities.size();
    for (int s = 0; s < simInitSize; ++s) {
      // ignore "deleted" edges
      if (similarities[s] == -1) continue;

      // Identify regions neighbouring the two regions
      // that will be merged
      int neighbourRegion = -1;
      if (regioni[s] == ri || regioni[s] == rj) {
        neighbourRegion = regionj[s];
      } else if (regionj[s] == ri || regionj[s] == rj){
        neighbourRegion = regioni[s];
      }

      // If edge s involves ri or rj, "delete" it by setting the similarity to -1.
      // Then, if we do not yet have an edge between our merged region and neighbourRegion,
      // create one by computing the similarity and updating the relevant vectors.
      if (neighbourRegion != -1) {
        similarities[s] = -1.0f;
        //printf("Remove (%d %d)\n", regioni[s], regionj[s]);
        if (!edgeAdded[neighbourRegion]) {
          similarities.push_back(similarity(neighbourRegion, newRegionIdx, nRegions, imSize, histColour,
                                            histTexture, regionSizes, rects, similarityFlags));
          regioni.push_back(neighbourRegion);
          regionj.push_back(newRegionIdx);
          edgeAdded[neighbourRegion] = true;
          //printf("Added (%d %d): %f\n", neighbourRegion, newRegionIdx, similarities[similarities.size()-1]);
        }
      }
    }
  }

  // Output the new regions
  for (int i = nInitRegions; i < rects.size()/4; ++i) {
    mergedRegions.push_back(rects[i*4]);
    mergedRegions.push_back(rects[i*4 + 1]);
    mergedRegions.push_back(rects[i*4 + 2]);
    mergedRegions.push_back(rects[i*4 + 3]);
  }
}

void vl::selectivesearch(std::vector<int>& rectsOut, std::vector<int>& initSeg,
                         std::vector<float>& histTexOut, std::vector<float>& histColourOut,
                         float const *data, int height, int width, std::vector<int> similarityMeasures,
                         float threshConst, int minSize)
{
  int nRegions;
  std::vector<float> blurred(height * width * nchannels);
  std::vector<int> regionSizes;
  std::vector<bool> neighbours;
  std::vector<int> rects;

  gaussianBlur(&blurred[0], data, height, width);

  if (initSeg.size() == 0) {
    initSeg.resize(height * width);
    time_t initSegt = clock();
    initialSegmentation(&initSeg[0], nRegions, regionSizes, neighbours, rects, &blurred[0],
        height, width, threshConst, minSize);
    if(timing) printf("initSeg %f\n", (clock() - initSegt)/(double)CLOCKS_PER_SEC);
  } else {
    nRegions = 0;
    for (int i = 0; i < initSeg.size(); ++i) nRegions = initSeg[i] > nRegions ? initSeg[i] : nRegions;
    nRegions++;
    printf("%d regions\n", nRegions);
    regionSizes.resize(nRegions, 0);
    for (int i = 0; i < initSeg.size(); ++i) regionSizes[initSeg[i]]++;
    findNeighboursAndRectangles(neighbours, nRegions, rects, height, width, initSeg);
  }

  std::vector<float>& histTexture = histTexOut;
  std::vector<float>& histColour = histColourOut;
  if (histTexture.size() == 0) {
    time_t tex = clock();
    computeTextureHistogram(histTexture, data, &blurred[0], &initSeg[0], height, width, nRegions);
    if(timing) {
    printf("tex %f\n", (clock() - tex)/(double)CLOCKS_PER_SEC);
    }
  }

  if (histColour.size() == 0) {
    time_t col = clock();
    computeColourHistogram(histColour, data, &initSeg[0], height, width, nRegions);
    if(timing) printf("col %f\n", (clock() - col)/(double)CLOCKS_PER_SEC);
  }

  time_t merge = clock();
  std::vector<int> mergedRegions = rects;
  for (int i = 0; i < similarityMeasures.size(); ++i) {
    mergeRegions(mergedRegions, nRegions, height*width, neighbours, histColour, histTexture,
                 regionSizes, rects, similarityMeasures[i]);
  }
  if(timing) printf("merge %f\n", (clock() - merge)/(double)CLOCKS_PER_SEC);

  rectsOut = mergedRegions;
}
