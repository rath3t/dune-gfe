#ifndef MPIFUNCTIONS_HH
#define MPIFUNCTIONS_HH

#include <numeric>
#include <vector>

struct MPIFunctions {
  static std::vector<int> offsetsFromSizes(const std::vector<int>& sizes) {
    std::vector<int> offsets(sizes.size());

    for (size_t k = 0; k < offsets.size(); ++k)
      offsets[k] = std::accumulate(sizes.begin(), sizes.begin() + k, 0);

    return offsets;
  }

  template<typename GridView>
  static std::vector<int> shareSizes(const GridView& gridview, const int& shareRef) {
    std::vector<int> sizesVector(gridview.comm().size());

    int share = shareRef;
    gridview.comm().template allgather<int>(&share, 1, sizesVector.data());


    return sizesVector;
  }

  template<typename GridView, typename T>
  static void scatterv(const GridView& gridview, std::vector<T>& localVec, std::vector<T>& globalVec, std::vector<int>& sizes, int root_rank) {
    int mysize = localVec.size();

    std::vector<int> offsets(offsetsFromSizes(sizes));

    gridview.comm().template scatterv(globalVec.data(), sizes.data(), offsets.data(), localVec.data(), mysize, root_rank);
  }

  template<typename GridView, typename T>
  static std::vector<T> gatherv(const GridView& gridview, std::vector<T>& localVec, std::vector<int>& sizes, int root_rank) {
    int mysize = localVec.size();

    std::vector<T> globalVec;

    if (gridview.comm().rank() == root_rank)
      globalVec.resize(std::accumulate(sizes.begin(), sizes.end(), 0));

    std::vector<int> offsets(offsetsFromSizes(sizes));

    gridview.comm().template gatherv(localVec.data(), mysize, globalVec.data(), sizes.data(), offsets.data(), root_rank);


    return globalVec;
  }

  template<typename GridView, typename T>
  static std::vector<T> allgatherv(const GridView& gridview, std::vector<T>& localVec, std::vector<int>& sizes) {
    int mysize = localVec.size();

    std::vector<T> globalVec(std::accumulate(sizes.begin(), sizes.end(), 0));

    std::vector<int> offsets(offsetsFromSizes(sizes));

    gridview.comm().template allgatherv(localVec.data(), mysize, globalVec.data(), sizes.data(), offsets.data());


    return globalVec;
  }
};


#endif
