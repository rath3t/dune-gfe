#ifndef VECTORCOMMUNICATOR_HH
#define VECTORCOMMUNICATOR_HH

#include <vector>

#include <dune/grid/utility/globalindexset.hh>
#include <dune/gfe/parallel/mpifunctions.hh>


namespace Dune::GFE
{

  template<typename GUIndex, typename Communicator, typename VectorType>
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
      for (size_t k=0; k<localVector.size(); k++)
        localVectorEntries.push_back(TransferVectorTuple(guIndex.index(k), localVector[k]));

      // Get number of vector entries on each process
      localVectorEntriesSizes = MPIFunctions::shareSizes(communicator_, localVectorEntries.size());

      // Get vector entries from every process
      globalVectorEntries = MPIFunctions::gatherv(communicator_, localVectorEntries, localVectorEntriesSizes, root_rank);
    }

  public:
    VectorCommunicator(const GUIndex& gi,
                       const Communicator& communicator,
                       const int& root)
      : guIndex(gi), communicator_(communicator), root_rank(root)
    {}

    VectorType reduceAdd(const VectorType& localVector)
    {
      transferVector(localVector);

      VectorType globalVector(guIndex.size());
      globalVector = 0;

      for (size_t k = 0; k < globalVectorEntries.size(); ++k)
        globalVector[globalVectorEntries[k].globalIndex_] += globalVectorEntries[k].value_;

      return globalVector;
    }

    VectorType reduceCopy(const VectorType& localVector)
    {
      transferVector(localVector);

      VectorType globalVector(guIndex.size());

      for (size_t k = 0; k < globalVectorEntries.size(); ++k)
        globalVector[globalVectorEntries[k].globalIndex_] = globalVectorEntries[k].value_;

      return globalVector;
    }

    VectorType scatter(const VectorType& global)
    {
      for (size_t k = 0; k < globalVectorEntries.size(); ++k)
        globalVectorEntries[k].value_ = global[globalVectorEntries[k].globalIndex_];

      const int localSize = localVectorEntriesSizes[communicator_.rank()];

      // Create vector for transfer data
      std::vector<TransferVectorTuple> localVectorEntries(localSize);

      MPIFunctions::scatterv(communicator_, localVectorEntries, globalVectorEntries, localVectorEntriesSizes, root_rank);

      // Create vector for local solution
      VectorType x(localSize);

      // And translate solution again
      for (size_t k = 0; k < localVectorEntries.size(); ++k)
        x[k] = localVectorEntries[k].value_;

      return x;
    }

  private:
    const GUIndex& guIndex;
    const Communicator& communicator_;
    int root_rank;

    std::vector<int> localVectorEntriesSizes;
    std::vector<TransferVectorTuple> globalVectorEntries;
  };

}  // namespace Dune::GFE

#endif
