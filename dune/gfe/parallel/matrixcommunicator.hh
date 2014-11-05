#ifndef MATRIXCOMMUNICATOR_HH
#define MATRIXCOMMUNICATOR_HH

#include <vector>

#include <dune/istl/matrixindexset.hh>

#include <dune/grid/utility/globalindexset.hh>
#include <dune/gfe/parallel/mpifunctions.hh>


template<typename GUIndex, typename GridView, typename MatrixType, typename LocalMapper1, typename LocalMapper2, typename ColGUIndex=GUIndex>
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

        localMatrixEntries.push_back(TransferMatrixTuple(localToGlobal1_[i], localToGlobal2_[j], *cIt));
      }

    // Get number of matrix entries on each process
    std::vector<int> localMatrixEntriesSizes(MPIFunctions::shareSizes(communicator_, localMatrixEntries.size()));

    // Get matrix entries from every process
    globalMatrixEntries = MPIFunctions::gatherv(communicator_, localMatrixEntries, localMatrixEntriesSizes, root_rank);
  }

public:
  MatrixCommunicator(const GUIndex& rowIndex, const GridView& gridView, const LocalMapper1& localMapper1, const LocalMapper2& localMapper2, const int& root)
  : guIndex1_(rowIndex),
    guIndex2_(rowIndex),
    localMapper1_(localMapper1),
    localMapper2_(localMapper2),
    communicator_(gridView.comm()),
    root_rank(root)
  {
    setLocalToGlobal(gridView);
  }

  MatrixCommunicator(const GUIndex& rowIndex, const ColGUIndex& colIndex, const GridView& gridView, const LocalMapper1& localMapper1, const LocalMapper2& localMapper2, const int& root)
  : guIndex1_(rowIndex),
    guIndex2_(colIndex),
    localMapper1_(localMapper1),
    localMapper2_(localMapper2),
    communicator_(gridView.comm()),
    root_rank(root)
  {
    setLocalToGlobal(gridView);
  }

  MatrixType reduceAdd(const MatrixType& local)
  {
    transferMatrix(local);

    MatrixType globalMatrix;

    // Create occupation pattern in matrix
    Dune::MatrixIndexSet occupationPattern;

    occupationPattern.resize(guIndex1_.size(), guIndex2_.size());

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

    occupationPattern.resize(localMapper1_.size(), localMapper2_.size());

    for (size_t k = 0; k < globalMatrixEntries.size(); ++k)
      occupationPattern.add(globalMatrixEntries[k].row, globalMatrixEntries[k].col);

    occupationPattern.exportIdx(globalMatrix);

    // Move entries to matrix
    for(size_t k = 0; k < globalMatrixEntries.size(); ++k)
      globalMatrix[globalMatrixEntries[k].row][globalMatrixEntries[k].col] = globalMatrixEntries[k].entry;


    return globalMatrix;
  }

private:

  void setLocalToGlobal(const GridView& gridView)
  {
    localToGlobal1_.resize(localMapper1_.size());
    localToGlobal2_.resize(localMapper2_.size());

    for (auto it = gridView.template begin<0>(); it != gridView.template end<0>(); ++it)
      for (int codim = 0; codim <= GridView::dimension; codim++)
        for (size_t i=0; i<it->subEntities(codim); i++)
        {
          typename GUIndex::Index localIdx = localMapper1_.map(*it,i,codim);
          typename GUIndex::Index globalIdx = guIndex1_.subIndex(*it,i,codim);
          localToGlobal1_[localIdx] = globalIdx;

          localIdx = localMapper2_.map(*it,i,codim);
          globalIdx = guIndex2_.subIndex(*it,i,codim);
          localToGlobal2_[localIdx] = globalIdx;
        }


  }

  // Mappers for the global numbering
  const GUIndex& guIndex1_;
  const ColGUIndex& guIndex2_;

  // Mappers for the local numbering
  const LocalMapper1& localMapper1_;
  const LocalMapper2& localMapper2_;

  const typename GridView::CollectiveCommunication& communicator_;
  int root_rank;

  std::vector<typename GUIndex::Index> localToGlobal1_;
  std::vector<typename ColGUIndex::Index> localToGlobal2_;

  std::vector<TransferMatrixTuple> globalMatrixEntries;
};

#endif
