#ifndef MATRIXCOMMUNICATOR_HH
#define MATRIXCOMMUNICATOR_HH

#include <vector>

#include <dune/istl/matrixindexset.hh>

#include <dune/grid/utility/globalindexset.hh>
#include <dune/gfe/parallel/mpifunctions.hh>


template<typename GUIndex, typename Communicator, typename MatrixType, typename ColGUIndex=GUIndex>
class MatrixCommunicator {

  struct TransferMatrixTuple {
    typedef typename MatrixType::block_type EntryType;

    size_t row, col;
    EntryType entry;

    TransferMatrixTuple() {}
    TransferMatrixTuple(const size_t& r, const size_t& c, const EntryType& e) : row(r), col(c), entry(e) {}
  };

  void transferMatrix(const MatrixType& localMatrix) {
    // Create vector for transfer data
    std::vector<TransferMatrixTuple> localMatrixEntries;

    // Convert local matrix to serializable array
    typedef typename MatrixType::row_type::ConstIterator ColumnIterator;

    for (typename MatrixType::ConstIterator rIt = localMatrix.begin(); rIt != localMatrix.end(); ++rIt)
      for (ColumnIterator cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {
        const int i = rIt.index();
        const int j = cIt.index();

        localMatrixEntries.push_back(TransferMatrixTuple(guIndex1_.index(i), guIndex2_.index(j), *cIt));
      }

    // Get number of matrix entries on each process
    std::vector<int> localMatrixEntriesSizes(MPIFunctions::shareSizes(communicator_, localMatrixEntries.size()));

    // Get matrix entries from every process
    globalMatrixEntries = MPIFunctions::gatherv(communicator_, localMatrixEntries, localMatrixEntriesSizes, root_rank);
  }

public:
  MatrixCommunicator(const GUIndex& rowIndex, const Communicator& communicator, const int& root)
  : guIndex1_(rowIndex),
    guIndex2_(rowIndex),
    communicator_(communicator),
    root_rank(root)
  {}

  MatrixCommunicator(const GUIndex& rowIndex, const ColGUIndex& colIndex, const Communicator& communicator, const int& root)
  : guIndex1_(rowIndex),
    guIndex2_(colIndex),
    communicator_(communicator),
    root_rank(root)
  {}

  MatrixType reduceAdd(const MatrixType& local)
  {
    transferMatrix(local);

    MatrixType globalMatrix;

    // Create occupation pattern in matrix
    Dune::MatrixIndexSet occupationPattern;

    occupationPattern.resize(guIndex1_.nGlobalEntity(), guIndex2_.nGlobalEntity());

    for (size_t k = 0; k < globalMatrixEntries.size(); ++k)
      occupationPattern.add(globalMatrixEntries[k].row, globalMatrixEntries[k].col);

    occupationPattern.exportIdx(globalMatrix);

    // Initialize matrix to zero
    globalMatrix = 0;

    // Move entries to matrix
    for(size_t k = 0; k < globalMatrixEntries.size(); ++k)
      globalMatrix[globalMatrixEntries[k].row][globalMatrixEntries[k].col] += globalMatrixEntries[k].entry;

    return globalMatrix;
  }

  MatrixType reduceCopy(const MatrixType& local)
  {
    transferMatrix(local);

    MatrixType globalMatrix;

    // Create occupation pattern in matrix
    Dune::MatrixIndexSet occupationPattern;

    occupationPattern.resize(guIndex1_.nGlobalEntity(), guIndex2_.nGlobalEntity());

    for (size_t k = 0; k < globalMatrixEntries.size(); ++k)
      occupationPattern.add(globalMatrixEntries[k].row, globalMatrixEntries[k].col);

    occupationPattern.exportIdx(globalMatrix);

    // Move entries to matrix
    for(size_t k = 0; k < globalMatrixEntries.size(); ++k)
      globalMatrix[globalMatrixEntries[k].row][globalMatrixEntries[k].col] = globalMatrixEntries[k].entry;


    return globalMatrix;
  }

private:
  const GUIndex& guIndex1_;
  const ColGUIndex& guIndex2_;
  const Communicator& communicator_;
  int root_rank;

  std::vector<TransferMatrixTuple> globalMatrixEntries;
};

#endif
