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

  template<typename Communicator>
  static std::vector<int> shareSizes(const Communicator& communicator, const int& shareRef) {
    std::vector<int> sizesVector(communicator.size());

    int share = shareRef;
    communicator.template allgather<int>(&share, 1, sizesVector.data());


    return sizesVector;
  }

  template<typename Communicator, typename T>
  static void scatterv(const Communicator& communicator, std::vector<T>& localVec, std::vector<T>& globalVec, std::vector<int>& sizes, int root_rank) {
    int mysize = localVec.size();

    std::vector<int> offsets(offsetsFromSizes(sizes));

    communicator.template scatterv(globalVec.data(), sizes.data(), offsets.data(), localVec.data(), mysize, root_rank);
  }

  template<typename Communicator, typename T>
  static std::vector<T> gatherv(const Communicator& communicator, std::vector<T>& localVec, std::vector<int>& sizes, int root_rank) {
    int mysize = localVec.size();

    std::vector<T> globalVec;

    if (communicator.rank() == root_rank)
      globalVec.resize(std::accumulate(sizes.begin(), sizes.end(), 0));

    std::vector<int> offsets(offsetsFromSizes(sizes));

    communicator.template gatherv(localVec.data(), mysize, globalVec.data(), sizes.data(), offsets.data(), root_rank);


    return globalVec;
  }

  template<typename Communicator, typename T>
  static std::vector<T> allgatherv(const Communicator& communicator, std::vector<T>& localVec, std::vector<int>& sizes) {
    int mysize = localVec.size();

    std::vector<T> globalVec(std::accumulate(sizes.begin(), sizes.end(), 0));

    std::vector<int> offsets(offsetsFromSizes(sizes));

    communicator.template allgatherv(localVec.data(), mysize, globalVec.data(), sizes.data(), offsets.data());


    return globalVec;
  }
};


#endif
