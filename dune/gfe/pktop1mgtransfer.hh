#ifndef PK_TO_P1_MG_TRANSFER_HH
#define PK_TO_P1_MG_TRANSFER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functionspacebases/p2nodalbasis.hh>

#include <dune/solvers/transferoperators/truncatedmgtransfer.hh>
#include <dune/solvers/transferoperators/compressedmultigridtransfer.hh>

/** \brief Restriction and prolongation operator for truncated multigrid
 *     from a Pk nodal basis to a p1 one.
 *
 * \note This class is a temporary hack.  Certainly there are better ways to do
 *   this, and of course they should be in dune-solvers.  But I need this quick
 *   for some rapid prototyping.
 */

template<
    class VectorType,
    class BitVectorType = Dune::BitSetVector<VectorType::block_type::dimension>,
    class MatrixType = Dune::BCRSMatrix< typename Dune::FieldMatrix<
        typename VectorType::field_type, VectorType::block_type::dimension, VectorType::block_type::dimension> > >
class PKtoP1MGTransfer :
    public TruncatedCompressedMGTransfer<VectorType, BitVectorType, MatrixType>
{

    enum {blocksize = VectorType::block_type::dimension};

    typedef typename VectorType::field_type field_type;

public:

    typedef typename CompressedMultigridTransfer<VectorType, BitVectorType, MatrixType>::TransferOperatorType TransferOperatorType;
    

    /** \brief Default constructor */
    PKtoP1MGTransfer()
    {}

    template <class Basis, class GridView>
    void setup(const Basis& fineBasis,
               const P1NodalBasis<GridView>& p1Basis)
        {
        typedef typename TransferOperatorType::block_type TransferMatrixBlock;

        int rows = fineBasis.size();
        int cols = p1Basis.size();

        this->matrix_->setSize(rows,cols);
        this->matrix_->setBuildMode(TransferOperatorType::random);

        typedef typename GridView::template Codim<0>::Iterator ElementIterator;
        
        const GridView& gridView = p1Basis.getGridView();

        ElementIterator it    = gridView.template begin<0>();
        ElementIterator endIt = gridView.template end<0>();


        // ///////////////////////////////////////////
        // Determine the indices present in the matrix
        // /////////////////////////////////////////////////
        Dune::MatrixIndexSet indices(rows, cols);

        for (; it != endIt; ++it) {

            typedef typename P1NodalBasis<GridView,double>::LocalFiniteElement LP1FE;
            const LP1FE& coarseBaseSet = p1Basis.getLocalFiniteElement(*it);

            typedef typename Dune::LocalFiniteElementFunctionBase<LP1FE>::type FunctionBaseClass;

            typedef typename GridView::template Codim<0>::Entity Element;
            AlienElementLocalBasisFunction<Element, LP1FE, FunctionBaseClass> f(*it, *it, coarseBaseSet.localBasis());

            for (int i=0; i<coarseBaseSet.localBasis().size(); i++) {

                f.setIndex(i);
                std::vector<double> values;
                fineBasis.getLocalFiniteElement(*it).localInterpolation().interpolate(f,values);
                
                for (size_t j=0; j<values.size(); j++) {
                    
                    if (values[j] > 0.001) 
                    {
                        size_t globalFine   = fineBasis.index(*it, j);
                        size_t globalCoarse = p1Basis.index(*it, i);

                        indices.add(globalFine, globalCoarse);
                    }
                    
                }
                
            }

        }

        indices.exportIdx(*this->matrix_);

        // /////////////////////////////////////////////
        // Compute the matrix
        // /////////////////////////////////////////////

        for (it = gridView.template begin<0>(); it != endIt; ++it) {

            typedef typename P1NodalBasis<GridView,double>::LocalFiniteElement LP1FE;
            const LP1FE& coarseBaseSet = p1Basis.getLocalFiniteElement(*it);

            typedef typename Dune::LocalFiniteElementFunctionBase<LP1FE>::type FunctionBaseClass;

            typedef typename GridView::template Codim<0>::Entity Element;
            AlienElementLocalBasisFunction<Element, LP1FE, FunctionBaseClass> f(*it, *it, coarseBaseSet.localBasis());

            for (int i=0; i<coarseBaseSet.localBasis().size(); i++) {

                f.setIndex(i);
                std::vector<double> values;
                fineBasis.getLocalFiniteElement(*it).localInterpolation().interpolate(f,values);
                
                for (size_t j=0; j<values.size(); j++) {
                    
                    if (values[j] > 0.001) 
                    {
                        size_t globalFine   = fineBasis.index(*it, j);
                        size_t globalCoarse = p1Basis.index(*it, i);

                        (*this->matrix_)[globalFine][globalCoarse] = Dune::ScaledIdentityMatrix<double,TransferMatrixBlock::rows>(values[j]);
                    }
                    
                }
                
            }

        }

    }


#if 0
    /** \brief Restrict level fL of f and store the result in level cL of t
     *
     * \param critical Has to contain an entry for each degree of freedom.
     *        Those dofs with a set bit are treated as critical.
     */
    void restrict(const VectorType& f, VectorType &t, const BitVectorType& critical) const;

    /** \brief Restriction of  MultiGridTransfer*/
    using CompressedMultigridTransfer< VectorType, BitVectorType, MatrixType >::restrict;

    /** \brief Prolong level cL of f and store the result in level fL of t
     * 
     * \param critical Has to contain an entry for each degree of freedom.
     *        Those dofs with a set bit are treated as critical.
     */
    void prolong(const VectorType& f, VectorType &t, const BitVectorType& critical) const;

    /** \brief Prolongation of  MultiGridTransfer*/
    using CompressedMultigridTransfer< VectorType, BitVectorType, MatrixType >::prolong;

    /** \brief Galerkin assemble a coarse stiffness matrix
     *
     * \param critical Has to contain an entry for each degree of freedom.
     *        Those dofs with a set bit are treated as critical.
     */
    void galerkinRestrict(const MatrixType& fineMat, MatrixType& coarseMat, const BitVectorType& critical) const;

    /** \brief Galerkin restriction of  MultiGridTransfer*/
    using CompressedMultigridTransfer< VectorType, BitVectorType, MatrixType >::galerkinRestrict;
#endif
};

#endif
