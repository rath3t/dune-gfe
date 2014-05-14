#ifndef VECTORCOMMUNICATOR_HH
#define VECTORCOMMUNICATOR_HH

#include <vector>

#include <dune/gfe/parallel/globalindex.hh>
#include <dune/gfe/parallel/mpifunctions.hh>


template<typename GUIndex, typename VectorType>
class VectorCommunicator {

  struct TransferVectorTuple {
    typedef typename VectorType::value_type EntryType;

    size_t row;
    EntryType entry;

    TransferVectorTuple() {}
    TransferVectorTuple(const size_t& r, const EntryType& e) : row(r), entry(e) {}
  };

private:
  void transferVector(const VectorType& localVector) {
    // Create vector for transfer data
    std::vector<TransferVectorTuple> localVectorEntries;

    // Translate vector entries
    for (size_t k = 0; k < localVector.size(); ++k)
      localVectorEntries.push_back(TransferVectorTuple(guIndex.globalIndex(k), localVector[k]));

    // Get number of vector entries on each process
    localVectorEntriesSizes = MPIFunctions::shareSizes(guIndex.getGridView(), localVectorEntries.size());

    // Get vector entries from every process
    globalVectorEntries = MPIFunctions::gatherv(guIndex.getGridView(), localVectorEntries, localVectorEntriesSizes, root_rank);
  }


  VectorType createGlobalVector() const {
    VectorType globalVector(guIndex.nGlobalEntity());

    for (size_t k = 0; k < globalVectorEntries.size(); ++k)
      globalVector[globalVectorEntries[k].row] += globalVectorEntries[k].entry;

    return globalVector;
  }

  VectorType copyIntoGlobalVector() const {
    VectorType globalVector(guIndex.nGlobalEntity());

    for (size_t k = 0; k < globalVectorEntries.size(); ++k)
      globalVector[globalVectorEntries[k].row] = globalVectorEntries[k].entry;

    return globalVector;
  }

  VectorType createLocalSolution() {
    const int localSize = localVectorEntriesSizes[guIndex.getGridView().comm().rank()];

    // Create vector for transfer data
    std::vector<TransferVectorTuple> localVectorEntries(localSize);

    MPIFunctions::scatterv(guIndex.getGridView(), localVectorEntries, globalVectorEntries, localVectorEntriesSizes, root_rank);

    // Create vector for local solution
    VectorType x(localSize);

    // And translate solution again
    for (size_t k = 0; k < localVectorEntries.size(); ++k)
      x[guIndex.localIndex(localVectorEntries[k].row)] = localVectorEntries[k].entry;

    return x;
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
    return createGlobalVector();
  }

  VectorType reduceCopy(const VectorType& localVector)
  {
    transferVector(localVector);
    return copyIntoGlobalVector();
  }

  VectorType scatter(const VectorType& global)
  {
    for (size_t k = 0; k < globalVectorEntries.size(); ++k)
      globalVectorEntries[k].entry = global[globalVectorEntries[k].row];

    return createLocalSolution();
  }

private:
  const GUIndex& guIndex;
  int root_rank;

  std::vector<int> localVectorEntriesSizes;
  std::vector<TransferVectorTuple> globalVectorEntries;
};

#endif
