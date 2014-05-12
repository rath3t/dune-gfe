#ifndef MATRIXCOMMUNICATOR_HH
#define MATRIXCOMMUNICATOR_HH

#include <vector>

#include <dune/istl/matrixindexset.hh>

#include <dune/gfe/parallel/globalindex.hh>
#include <dune/gfe/parallel/mpifunctions.hh>


template<typename GUIndex, typename MatrixType>
class MatrixCommunicator {
public:
  struct TransferMatrixTuple {
    typedef typename MatrixType::block_type EntryType;

    size_t row, col;
    EntryType entry;

    TransferMatrixTuple() {}
    TransferMatrixTuple(const size_t& r, const size_t& c, const EntryType& e) : row(r), col(c), entry(e) {}
  };

public:
  MatrixCommunicator(const GUIndex& gi, const int& root) : guIndex(gi), root_rank(root) {}


  void transferMatrix(const MatrixType& localMatrix) {
    // Create vector for transfer data
    std::vector<TransferMatrixTuple> localMatrixEntries;

    // Convert local matrix to serializable array
    typedef typename MatrixType::row_type::ConstIterator ColumnIterator;

    for (typename MatrixType::ConstIterator rIt = localMatrix.begin(); rIt != localMatrix.end(); ++rIt)
      for (ColumnIterator cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {
        const int i = rIt.index();
        const int j = cIt.index();

        localMatrixEntries.push_back(TransferMatrixTuple(guIndex.globalIndex(i), guIndex.globalIndex(j), *cIt));
      }

    // Get number of matrix entries on each process
    std::vector<int> localMatrixEntriesSizes(MPIFunctions::shareSizes(guIndex.getGridView(), localMatrixEntries.size()));

    // Get matrix entries from every process
    globalMatrixEntries = MPIFunctions::gatherv(guIndex.getGridView(), localMatrixEntries, localMatrixEntriesSizes, root_rank);
  }


  MatrixType createGlobalMatrix() const {
    MatrixType globalMatrix;

    // Create occupation pattern in matrix
    Dune::MatrixIndexSet occupationPattern;

    occupationPattern.resize(guIndex.nGlobalEntity(), guIndex.nGlobalEntity());

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

  MatrixType copyIntoGlobalMatrix() const {
    MatrixType globalMatrix;

    // Create occupation pattern in matrix
    Dune::MatrixIndexSet occupationPattern;

    occupationPattern.resize(guIndex.nGlobalEntity(), guIndex.nGlobalEntity());

    for (size_t k = 0; k < globalMatrixEntries.size(); ++k)
      occupationPattern.add(globalMatrixEntries[k].row, globalMatrixEntries[k].col);

    occupationPattern.exportIdx(globalMatrix);

    // Move entries to matrix
    for(size_t k = 0; k < globalMatrixEntries.size(); ++k)
      globalMatrix[globalMatrixEntries[k].row][globalMatrixEntries[k].col] = globalMatrixEntries[k].entry;


    return globalMatrix;
  }

private:
  const GUIndex& guIndex;
  int root_rank;

  std::vector<TransferMatrixTuple> globalMatrixEntries;
};

#endif
