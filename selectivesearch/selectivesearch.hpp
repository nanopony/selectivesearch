#ifndef __vl__selectivesearch__
#define __vl__selectivesearch__

#include <vector>

namespace vl {

  void selectivesearch(std::vector<int>& rectsOut, std::vector<int>& initSeg, std::vector<float>& histTexOut, std::vector<float>& histColourOut,
                       float const *data, int height, int width,
                       std::vector<int> similarityMeasures, float threshConst, int minSize);

  enum Similarity
  {
      SIM_COLOUR = 1 << 0,
      SIM_TEXTURE = 1 << 1,
      SIM_SIZE = 1 << 2,
      SIM_FILL = 1 << 3
  };
}

#endif /* defined(__vl__selectivesearch__) */
