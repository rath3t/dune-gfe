#ifndef DUNE_GFE_MIXEDGFEASSEMBLER_HH
#define DUNE_GFE_MIXEDGFEASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>

#include <dune/gfe/mixedlocalgeodesicfestiffness.hh>


/** \brief A global FE assembler for problems involving functions that map into non-Euclidean spaces
 */
template <class Basis, class TargetSpace0, class TargetSpace1>
class MixedGFEAssembler {

    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::template Partition<Dune::Interior_Partition>::Iterator ElementIterator;

    //! Dimension of the grid.
    enum { gridDim = GridView::dimension };

    //! Dimension of a tangent space
    enum { blocksize0 = TargetSpace0::TangentVector::dimension };
    enum { blocksize1 = TargetSpace1::TangentVector::dimension };

    //!
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize0, blocksize0> > MatrixBlock00;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize0, blocksize1> > MatrixBlock01;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize1, blocksize0> > MatrixBlock10;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, blocksize1, blocksize1> > MatrixBlock11;

    typedef Dune::MultiTypeBlockMatrix<Dune::MultiTypeBlockVector<MatrixBlock00,MatrixBlock01>,
                                       Dune::MultiTypeBlockVector<MatrixBlock10,MatrixBlock11> > MatrixType;

protected:
public:
    const Basis basis_;

    MixedLocalGeodesicFEStiffness<Basis,
                                  TargetSpace0,
                                  TargetSpace1>* localStiffness_;

public:

    /** \brief Constructor for a given grid */
    MixedGFEAssembler(const Basis& basis,
                      MixedLocalGeodesicFEStiffness<Basis, TargetSpace0, TargetSpace1>* localStiffness)
        : basis_(basis),
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
                                            MatrixType& hessian,
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



template <class Basis, class TargetSpace0, class TargetSpace1>
void MixedGFEAssembler<Basis,TargetSpace0,TargetSpace1>::
getMatrixPattern(Dune::MatrixIndexSet& nb00,
                 Dune::MatrixIndexSet& nb01,
                 Dune::MatrixIndexSet& nb10,
                 Dune::MatrixIndexSet& nb11) const
{
    nb00.resize(basis_.size({0}), basis_.size({0}));
    nb01.resize(basis_.size({0}), basis_.size({1}));
    nb10.resize(basis_.size({1}), basis_.size({0}));
    nb11.resize(basis_.size({1}), basis_.size({1}));

    // A view on the FE basis on a single element
    auto localView = basis_.localView();
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    // Loop over grid elements
    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        // Bind the local FE basis view to the current element
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);

        // Add element stiffness matrix onto the global stiffness matrix
        for (size_t i=0; i<localIndexSet.size(); i++)
        {
          // The global index of the i-th local degree of freedom of the element 'e'
          auto row = localIndexSet.index(i);

          for (size_t j=0; j<localIndexSet.size(); j++ )
          {
            // The global index of the j-th local degree of freedom of the element 'e'
            auto col = localIndexSet.index(j);
#else
        // Add element stiffness matrix onto the global stiffness matrix
        for (size_t i=0; i<localView.size(); i++)
        {
          // The global index of the i-th local degree of freedom of the element 'e'
          auto row = localView.index(i);

          for (size_t j=0; j<localView.size(); j++ )
          {
            // The global index of the j-th local degree of freedom of the element 'e'
            auto col = localView.index(j);
#endif

            if (row[0]==0 and col[0]==0)
              nb00.add(row[1],col[1]);
            if (row[0]==0 and col[0]==1)
              nb01.add(row[1],col[1]);
            if (row[0]==1 and col[0]==0)
              nb10.add(row[1],col[1]);
            if (row[0]==1 and col[0]==1)
              nb11.add(row[1],col[1]);

          }

        }

    }

}

template <class Basis, class TargetSpace0, class TargetSpace1>
void MixedGFEAssembler<Basis,TargetSpace0,TargetSpace1>::
assembleGradientAndHessian(const std::vector<TargetSpace0>& configuration0,
                           const std::vector<TargetSpace1>& configuration1,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize0> >& gradient0,
                           Dune::BlockVector<Dune::FieldVector<double, blocksize1> >& gradient1,
                           MatrixType& hessian,
                           bool computeOccupationPattern) const
{
    if (computeOccupationPattern) {

        Dune::MatrixIndexSet pattern00;
        Dune::MatrixIndexSet pattern01;
        Dune::MatrixIndexSet pattern10;
        Dune::MatrixIndexSet pattern11;

        getMatrixPattern(pattern00, pattern01, pattern10, pattern11);

        using namespace Dune::TypeTree::Indices;
        pattern00.exportIdx(hessian[_0][_0]);
        pattern01.exportIdx(hessian[_0][_1]);
        pattern10.exportIdx(hessian[_1][_0]);
        pattern11.exportIdx(hessian[_1][_1]);

    }

    hessian = 0;

    gradient0.resize(configuration0.size());
    gradient0 = 0;
    gradient1.resize(configuration1.size());
    gradient1 = 0;

    // A view on the FE basis on a single element
    auto localView = basis_.localView();
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        // Bind the local FE basis view to the current element
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        using namespace Dune::TypeTree::Indices;
        const int nDofs0 = localView.tree().child(_0).finiteElement().size();
        const int nDofs1 = localView.tree().child(_1).finiteElement().size();

        // Extract local solution
        std::vector<TargetSpace0> localConfiguration0(nDofs0);
        std::vector<TargetSpace1> localConfiguration1(nDofs1);

        for (int i=0; i<nDofs0+nDofs1; i++)
        {
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
          if (localIndexSet.index(i)[0] == 0)
            localConfiguration0[i] = configuration0[localIndexSet.index(i)[1]];
          else
            localConfiguration1[i-nDofs0] = configuration1[localIndexSet.index(i)[1]];
#else
          if (localView.index(i)[0] == 0)
            localConfiguration0[i] = configuration0[localView.index(i)[1]];
          else
            localConfiguration1[i-nDofs0] = configuration1[localView.index(i)[1]];
#endif
        }

        std::vector<Dune::FieldVector<double,blocksize0> > localGradient0(nDofs0);
        std::vector<Dune::FieldVector<double,blocksize1> > localGradient1(nDofs1);

        // setup local matrix and gradient
        localStiffness_->assembleGradientAndHessian(localView,
                                                    localConfiguration0, localConfiguration1,
                                                    localGradient0, localGradient1);

        // Add element matrix to global stiffness matrix
        for (int i=0; i<nDofs0+nDofs1; i++)
        {
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            auto row = localIndexSet.index(i);
#else
            auto row = localView.index(i);
#endif

            for (int j=0; j<nDofs0+nDofs1; j++ )
            {
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
                auto col = localIndexSet.index(j);
#else
                auto col = localView.index(j);
#endif

                if (row[0]==0 and col[0]==0)
                  hessian[_0][_0][row[1]][col[1]] += localStiffness_->A00_[i][j];

                if (row[0]==0 and col[0]==1)
                  hessian[_0][_1][row[1]][col[1]] += localStiffness_->A01_[i][j-nDofs0];

                if (row[0]==1 and col[0]==0)
                  hessian[_1][_0][row[1]][col[1]] += localStiffness_->A10_[i-nDofs0][j];

                if (row[0]==1 and col[0]==1)
                  hessian[_1][_1][row[1]][col[1]] += localStiffness_->A11_[i-nDofs0][j-nDofs0];
            }

            // Add local gradient to global gradient
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            if (localIndexSet.index(i)[0] == 0)
              gradient0[localIndexSet.index(i)[1]] += localGradient0[i];
            else
              gradient1[localIndexSet.index(i)[1]] += localGradient1[i-nDofs0];
#else
            if (localView.index(i)[0] == 0)
              gradient0[localView.index(i)[1]] += localGradient0[i];
            else
              gradient1[localView.index(i)[1]] += localGradient1[i-nDofs0];
#endif
        }

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

template <class Basis, class TargetSpace0, class TargetSpace1>
double MixedGFEAssembler<Basis, TargetSpace0, TargetSpace1>::
computeEnergy(const std::vector<TargetSpace0>& configuration0,
              const std::vector<TargetSpace1>& configuration1) const
{
    double energy = 0;

    if (configuration0.size()!=basis_.size({0}))
        DUNE_THROW(Dune::Exception, "Configuration vector 0 doesn't match the basis!");

    if (configuration1.size()!=basis_.size({1}))
        DUNE_THROW(Dune::Exception, "Configuration vector 1 doesn't match the basis!");

    // A view on the FE basis on a single element
    auto localView = basis_.localView();
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
    auto localIndexSet = basis_.localIndexSet();
#endif

    // Loop over all elements
    for (const auto& element : elements(basis_.gridView(), Dune::Partitions::interior))
    {
        // Bind the local FE basis view to the current element
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        localIndexSet.bind(localView);
#endif

        // Number of degrees of freedom on this element
        using namespace Dune::TypeTree::Indices;
        const int nDofs0 = localView.tree().child(_0).finiteElement().size();
        const int nDofs1 = localView.tree().child(_1).finiteElement().size();

        std::vector<TargetSpace0> localConfiguration0(nDofs0);
        std::vector<TargetSpace1> localConfiguration1(nDofs1);

        for (int i=0; i<nDofs0+nDofs1; i++)
        {
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
          if (localIndexSet.index(i)[0] == 0)
            localConfiguration0[i] = configuration0[localIndexSet.index(i)[1]];
          else
            localConfiguration1[i-nDofs0] = configuration1[localIndexSet.index(i)[1]];
#else
          if (localView.index(i)[0] == 0)
            localConfiguration0[i] = configuration0[localView.index(i)[1]];
          else
            localConfiguration1[i-nDofs0] = configuration1[localView.index(i)[1]];
#endif
        }

        energy += localStiffness_->energy(localView,
                                          localConfiguration0,
                                          localConfiguration1);

    }

    return energy;

}

#endif

