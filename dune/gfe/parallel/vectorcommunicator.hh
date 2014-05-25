#ifndef VECTORCOMMUNICATOR_HH
#define VECTORCOMMUNICATOR_HH

#include <vector>

#include <dune/gfe/parallel/globalindex.hh>
#include <dune/gfe/parallel/mpifunctions.hh>


template<typename GUIndex, typename VectorType>
class VectorCommunicator {

  struct TransferVectorTuple {
    typedef typename VectorType::value_type EntryType;

    size_t globalIndex_;
    EntryType value_;

    TransferVectorTuple() {}
    TransferVectorTuple(const size_t& r, const EntryType& e)
    : globalIndex_(r),
      value_(e) {}
  };

private:
  void transferVector(const VectorType& localVector) {
    // Create vector for transfer data
    std::vector<TransferVectorTuple> localVectorEntries;

    // Translate vector entries
    const auto& gridView = guIndex.getGridView();
    for (auto it = gridView.template begin<2>(); it != gridView.template end<2>(); ++ it)
        localVectorEntries.push_back(TransferVectorTuple(guIndex.globalIndex(*it), localVector[guIndex.localIndex(*it)]));

    // Get number of vector entries on each process
    localVectorEntriesSizes = MPIFunctions::shareSizes(guIndex.getGridView(), localVectorEntries.size());

    // Get vector entries from every process
    globalVectorEntries = MPIFunctions::gatherv(guIndex.getGridView(), localVectorEntries, localVectorEntriesSizes, root_rank);
  }

public:
  VectorCommunicator(const GUIndex& gi, const int& root)
  : guIndex(gi), root_rank(root)
  {
    // Get number of vector entries on each process
    localVectorEntriesSizes = MPIFunctions::shareSizes(guIndex.getGridView(), guIndex.nOwnedLocalEntity());
  }

  VectorType reduceAdd(const VectorType& localVector)
  {
    transferVector(localVector);

    VectorType globalVector(guIndex.nGlobalEntity());
    globalVector = 0;

    for (size_t k = 0; k < globalVectorEntries.size(); ++k)
      globalVector[globalVectorEntries[k].globalIndex_] += globalVectorEntries[k].value_;

    return globalVector;
  }

  VectorType reduceCopy(const VectorType& localVector)
  {
    transferVector(localVector);

    VectorType globalVector(guIndex.nGlobalEntity());

    for (size_t k = 0; k < globalVectorEntries.size(); ++k)
      globalVector[globalVectorEntries[k].globalIndex_] = globalVectorEntries[k].value_;

    return globalVector;
  }

  VectorType scatter(const VectorType& global)
  {
    for (size_t k = 0; k < globalVectorEntries.size(); ++k)
      globalVectorEntries[k].value_ = global[globalVectorEntries[k].globalIndex_];

    const int localSize = localVectorEntriesSizes[guIndex.getGridView().comm().rank()];

    // Create vector for transfer data
    std::vector<TransferVectorTuple> localVectorEntries(localSize);

    MPIFunctions::scatterv(guIndex.getGridView(), localVectorEntries, globalVectorEntries, localVectorEntriesSizes, root_rank);

    // Create vector for local solution
    VectorType x(localSize);

    // And translate solution again
    for (size_t k = 0; k < localVectorEntries.size(); ++k)
      x[guIndex.localIndex(localVectorEntries[k].globalIndex_)] = localVectorEntries[k].value_;

    return x;
  }

private:
  const GUIndex& guIndex;
  int root_rank;

  std::vector<int> localVectorEntriesSizes;
  std::vector<TransferVectorTuple> globalVectorEntries;
};

#endif
