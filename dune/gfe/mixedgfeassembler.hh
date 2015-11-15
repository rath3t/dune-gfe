#ifndef DUNE_GFE_MIXEDGFEASSEMBLER_HH
#define DUNE_GFE_MIXEDGFEASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/mixedlocalgeodesicfestiffness.hh>


/** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces
 */
template <class Basis0, class TargetSpace0, class Basis1, class TargetSpace1>
class MixedGFEAssembler {

    typedef typename Basis0::GridView GridView;
    typedef typename GridView::template Codim<0>::template Partition<Dune::Interior_Partition>::Iterator ElementIterator;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };

    //! Dimension of a tangent space
    enum { blocksize0 = TargetSpace0::TangentVector::dimension };
    enum { blocksize1 = TargetSpace1::TangentVector::dimension };

    //!
    typedef Dune::FieldMatrix<double, blocksize0, blocksize0> MatrixBlock00;
    typedef Dune::FieldMatrix<double, blocksize0, blocksize1> MatrixBlock01;
    typedef Dune::FieldMatrix<double, blocksize1, blocksize0> MatrixBlock10;
    typedef Dune::FieldMatrix<double, blocksize1, blocksize1> MatrixBlock11;

protected:
public:
    const Basis0 basis0_;
    const Basis1 basis1_;

    MixedLocalGeodesicFEStiffness<Basis0, TargetSpace0,
                                  Basis1, TargetSpace1>* localStiffness_;

public:

    /** \brief Constructor for a given grid */
    MixedGFEAssembler(const Basis0& basis0,
                      const Basis1& basis1,
                      MixedLocalGeodesicFEStiffness<Basis0, TargetSpace0,
                                                    Basis1, TargetSpace1>* localStiffness)
        : basis0_(basis0),
          basis1_(basis1),
          localStiffness_(localStiffness)
    {}

    /** \brief Assemble the tangent stiffness matrix and the functional gradient together
     *
     * This is more efficient than computing them separately, because you need the gradient
     * anyway to compute the Riemannian Hessian.
     */
    virtual void assembleGradientAndHessian(const std::vector<TargetSpace0>& configuration0,
                                            const std::vector<TargetSpace1>& configuration1,
                                            Dune::BlockVector<Dune::FieldVector<double, blocksize0> >& gradient0,
                                            Dune::BlockVector<Dune::FieldVector<double, blocksize1> >& gradient1,
                                            Dune::BCRSMatrix<MatrixBlock00>& hessian00,
                                            Dune::BCRSMatrix<MatrixBlock01>& hessian01,
                                            Dune::BCRSMatrix<MatrixBlock10>& hessian10,
                                            Dune::BCRSMatrix<MatrixBlock11>& hessian11,
                                            bool computeOccupationPattern=true) const;
#if 0
    /** \brief Assemble the gradient */
    virtual void assembleGradient(const std::vector<TargetSpace>& sol,
                          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;
#endif
    /** \brief Compute the energy of a deformation state */
    virtual double computeEnergy(const std::vector<TargetSpace0>& configuration0,
                                 const std::vector<TargetSpace1>& configuration1) const;

    //protected:
    void getMatrixPattern(Dune::MatrixIndexSet& nb00,
                          Dune::MatrixIndexSet& nb01,
                          Dune::MatrixIndexSet& nb10,
                          Dune::MatrixIndexSet& nb11) const;

}; // end class



template <class Basis0, class TargetSpace0, class Basis1, class TargetSpace1>
void MixedGFEAssembler<Basis0,TargetSpace0,Basis1,TargetSpace1>::
getMatrixPattern(Dune::MatrixIndexSet& nb00,
                 Dune::MatrixIndexSet& nb01,
                 Dune::MatrixIndexSet& nb10,
                 Dune::MatrixIndexSet& nb11) const
{
    nb00.resize(basis0_.indexSet().size(), basis0_.indexSet().size());
    nb01.resize(basis0_.indexSet().size(), basis1_.indexSet().size());
    nb10.resize(basis1_.indexSet().size(), basis0_.indexSet().size());
    nb11.resize(basis1_.indexSet().size(), basis1_.indexSet().size());

    // A view on the FE basis on a single element
    auto localView0 = basis0_.localView();
    auto localView1 = basis1_.localView();
    auto localIndexSet0 = basis0_.indexSet().localIndexSet();
    auto localIndexSet1 = basis1_.indexSet().localIndexSet();

    // Grid view must be the same for both bases
    ElementIterator it    = basis0_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endit = basis0_.gridView().template end<0,Dune::Interior_Partition>  ();

    for (; it!=endit; ++it) {

        // Bind the local FE basis view to the current element
        localView0.bind(*it);
        localView1.bind(*it);
        localIndexSet0.bind(localView0);
        localIndexSet1.bind(localView1);

        for (size_t i=0; i<localView0.size(); i++) {

            int iIdx = localIndexSet0.index(i)[0];

            for (size_t j=0; j<localView0.size(); j++) {
                int jIdx = localIndexSet0.index(j)[0];
                nb00.add(iIdx, jIdx);
            }

            for (size_t j=0; j<localView1.size(); j++) {
                int jIdx = localIndexSet1.index(j)[0];
                nb01.add(iIdx, jIdx);
            }

        }

        for (size_t i=0; i<localView1.size(); i++) {

            int iIdx = localIndexSet1.index(i)[0];

            for (size_t j=0; j<localView0.size(); j++) {
                int jIdx = localIndexSet0.index(j)[0];
                nb10.add(iIdx, jIdx);
            }

            for (size_t j=0; j<localView1.size(); j++) {
                int jIdx = localIndexSet1.index(j)[0];
                nb11.add(iIdx, jIdx);
            }

        }

    }

}

template <class Basis0, class TargetSpace0, class Basis1, class TargetSpace1>
void MixedGFEAssembler<Basis0,TargetSpace0,Basis1,TargetSpace1>::
assembleGradientAndHessian(const std::vector<TargetSpace0>& configuration0,
                           const std::vector<TargetSpace1>& configuration1,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize0> >& gradient0,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize1> >& gradient1,
                           Dune::BCRSMatrix<MatrixBlock00>& hessian00,
                           Dune::BCRSMatrix<MatrixBlock01>& hessian01,
                           Dune::BCRSMatrix<MatrixBlock10>& hessian10,
                           Dune::BCRSMatrix<MatrixBlock11>& hessian11,
                           bool computeOccupationPattern) const
{
    if (computeOccupationPattern) {

        Dune::MatrixIndexSet pattern00;
        Dune::MatrixIndexSet pattern01;
        Dune::MatrixIndexSet pattern10;
        Dune::MatrixIndexSet pattern11;

        getMatrixPattern(pattern00, pattern01, pattern10, pattern11);

        pattern00.exportIdx(hessian00);
        pattern01.exportIdx(hessian01);
        pattern10.exportIdx(hessian10);
        pattern11.exportIdx(hessian11);

    }

    hessian00 = 0;
    hessian01 = 0;
    hessian10 = 0;
    hessian11 = 0;

    gradient0.resize(configuration0.size());
    gradient0 = 0;
    gradient1.resize(configuration1.size());
    gradient1 = 0;

    // A view on the FE basis on a single element
    auto localView0 = basis0_.localView();
    auto localView1 = basis1_.localView();
    auto localIndexSet0 = basis0_.indexSet().localIndexSet();
    auto localIndexSet1 = basis1_.indexSet().localIndexSet();

    ElementIterator it    = basis0_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endit = basis0_.gridView().template end<0,Dune::Interior_Partition>  ();

    for( ; it != endit; ++it ) {

        // Bind the local FE basis view to the current element
        localView0.bind(*it);
        localView1.bind(*it);
        localIndexSet0.bind(localView0);
        localIndexSet1.bind(localView1);

        const int nDofs0 = localView0.size();
        const int nDofs1 = localView1.size();

        // Extract local solution
        std::vector<TargetSpace0> localConfiguration0(nDofs0);
        std::vector<TargetSpace1> localConfiguration1(nDofs1);

        for (int i=0; i<nDofs0; i++)
            localConfiguration0[i] = configuration0[localIndexSet0.index(i)[0]];

        for (int i=0; i<nDofs1; i++)
            localConfiguration1[i] = configuration1[localIndexSet1.index(i)[0]];

        std::vector<Dune::FieldVector<double,blocksize0> > localGradient0(nDofs0);
        std::vector<Dune::FieldVector<double,blocksize1> > localGradient1(nDofs1);

        // setup local matrix and gradient
        localStiffness_->assembleGradientAndHessian(*it,
                                                    localView0.tree().finiteElement(), localConfiguration0,
                                                    localView1.tree().finiteElement(), localConfiguration1,
                                                    localGradient0, localGradient1);

        // Add element matrix to global stiffness matrix
        for (int i=0; i<nDofs0; i++) {

            int row = localIndexSet0.index(i)[0];

            for (int j=0; j<nDofs0; j++ ) {
                int col = localIndexSet0.index(j)[0];
                hessian00[row][col] += localStiffness_->A00_[i][j];
            }

            for (int j=0; j<nDofs1; j++ ) {
                int col = localIndexSet1.index(j)[0];
                hessian01[row][col] += localStiffness_->A01_[i][j];
            }
        }

        for (int i=0; i<nDofs1; i++) {

            int row = localIndexSet1.index(i)[0];

            for (int j=0; j<nDofs0; j++ ) {
                int col = localIndexSet0.index(j)[0];
                hessian10[row][col] += localStiffness_->A10_[i][j];
            }

            for (int j=0; j<nDofs1; j++ ) {
                int col = localIndexSet1.index(j)[0];
                hessian11[row][col] += localStiffness_->A11_[i][j];
            }
        }

        // Add local gradient to global gradient
        for (int i=0; i<nDofs0; i++)
            gradient0[localIndexSet0.index(i)[0]] += localGradient0[i];

        for (int i=0; i<nDofs1; i++)
            gradient1[localIndexSet1.index(i)[0]] += localGradient1[i];
    }

}

#if 0
template <class Basis, class TargetSpace>
void GeodesicFEAssembler<Basis,TargetSpace>::
assembleGradient(const std::vector<TargetSpace>& sol,
                 Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const
{
    if (sol.size()!=basis_.size())
        DUNE_THROW(Dune::Exception, "Solution vector doesn't match the grid!");

    grad.resize(sol.size());
    grad = 0;

    ElementIterator it    = basis_.getGridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endIt = basis_.getGridView().template end<0,Dune::Interior_Partition>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // A 1d grid has two vertices
        const int nDofs = basis_.getLocalFiniteElement(*it).localBasis().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(nDofs);

        for (int i=0; i<nDofs; i++)
            localSolution[i] = sol[basis_.index(*it,i)];

        // Assemble local gradient
        std::vector<Dune::FieldVector<double,blocksize> > localGradient(nDofs);

        localStiffness_->assembleGradient(*it, basis_.getLocalFiniteElement(*it), localSolution, localGradient);

        // Add to global gradient
        for (int i=0; i<nDofs; i++)
            grad[basis_.index(*it,i)] += localGradient[i];

    }

}
#endif

template <class Basis0, class TargetSpace0, class Basis1, class TargetSpace1>
double MixedGFEAssembler<Basis0, TargetSpace0, Basis1, TargetSpace1>::
computeEnergy(const std::vector<TargetSpace0>& configuration0,
              const std::vector<TargetSpace1>& configuration1) const
{
    double energy = 0;

    if (configuration0.size()!=basis0_.indexSet().size())
        DUNE_THROW(Dune::Exception, "Configuration vector 0 doesn't match the basis!");

    if (configuration1.size()!=basis1_.indexSet().size())
        DUNE_THROW(Dune::Exception, "Configuration vector 1 doesn't match the basis!");

    // A view on the FE basis on a single element
    auto localView0 = basis0_.localView();
    auto localView1 = basis1_.localView();
    auto localIndexSet0 = basis0_.indexSet().localIndexSet();
    auto localIndexSet1 = basis1_.indexSet().localIndexSet();

    ElementIterator it    = basis0_.gridView().template begin<0,Dune::Interior_Partition>();
    ElementIterator endIt = basis0_.gridView().template end<0,Dune::Interior_Partition>();

    // Loop over all elements
    for (; it!=endIt; ++it) {

        // Bind the local FE basis view to the current element
        localView0.bind(*it);
        localView1.bind(*it);
        localIndexSet0.bind(localView0);
        localIndexSet1.bind(localView1);

        // Number of degrees of freedom on this element
        size_t nDofs0 = localView0.size();
        size_t nDofs1 = localView1.size();

        std::vector<TargetSpace0> localConfiguration0(nDofs0);
        std::vector<TargetSpace1> localConfiguration1(nDofs1);

        for (size_t i=0; i<nDofs0; i++)
            localConfiguration0[i] = configuration0[localIndexSet0.index(i)[0]];

        for (size_t i=0; i<nDofs1; i++)
            localConfiguration1[i] = configuration1[localIndexSet1.index(i)[0]];

        energy += localStiffness_->energy(*it,
                                          localView0.tree().finiteElement(), localConfiguration0,
                                          localView1.tree().finiteElement(), localConfiguration1);

    }

    return energy;

}

#endif

