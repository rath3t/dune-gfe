#include "config.h"

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/geometry/type.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfestiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"


using namespace Dune;

//typedef std::conditional<domainDim==1,OneDGrid,UGGrid<domainDim> >::type GridType;
typedef OneDGrid GridType;









/** \brief A special energy functional of which I happen to be able to compute the Hessian */
template<class GridView, class LocalFiniteElement, class TargetSpace>
class TestEnergyLocalStiffness 
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,TargetSpace>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

public:
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const;
               
};

template <class GridView, class LocalFiniteElement, class TargetSpace>
typename TestEnergyLocalStiffness<GridView, LocalFiniteElement, TargetSpace>::RT TestEnergyLocalStiffness<GridView, LocalFiniteElement, TargetSpace>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localSolution) const
{
    return TargetSpace::distance(localSolution[0], localSolution[1]) 
           * TargetSpace::distance(localSolution[0], localSolution[1]);
}


template <int domainDim>
GridType* makeTestGrid()
{
    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    GridFactory<GridType> factory;

    FieldVector<double,domainDim> pos(0);
    factory.insertVertex(pos);

    for (int i=0; i<domainDim; i++) {
        pos = 0;
        pos[i] = 1;
        factory.insertVertex(pos);
    }

    std::vector<unsigned int> v(domainDim+1);
    for (int i=0; i<domainDim+1; i++)
        v[i] = i;
    factory.insertElement(GeometryType(GeometryType::simplex,domainDim), v);

    return factory.createGrid();
}









template <class TargetSpace, int domainDim>
void testHessian()
{
    const GridType* grid = makeTestGrid<domainDim>();

    const int spaceDim = TargetSpace::TangentVector::dimension;
    const int embeddedSpaceDim = TargetSpace::EmbeddedTangentVector::dimension;
    
    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////
    
    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    typedef P1NodalBasis<GridType::LeafGridView,double> P1Basis;
    P1Basis p1Basis(grid->leafGridView());
    TestEnergyLocalStiffness<typename GridType::LeafGridView, P1Basis::LocalFiniteElement, TargetSpace> assembler;

    // Set up elements of S^2
    std::vector<TargetSpace> coefficients(domainDim+1);

    MultiIndex index(domainDim+1, nTestPoints);
    int numIndices = index.cycle();
    
    size_t nDofs = domainDim+1;

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        std::cout << "coefficients:\n";
        for (int j=0; j<domainDim+1; j++)
            std::cout << coefficients[j] << std::endl;
        
        assembler.assembleHessian(*grid->template leafbegin<0>(), 
                                  p1Basis.getLocalFiniteElement(*grid->template leafbegin<0>()),
                                  coefficients);
        
        Matrix<FieldMatrix<double,spaceDim,spaceDim> > fdHessian = assembler.A_;
        
        std::cout << "fdHessian:\n";
        printmatrix(std::cout, fdHessian, "fdHessian", "--");
        
        Matrix<FieldMatrix<double,embeddedSpaceDim,embeddedSpaceDim> > embeddedHessian(nDofs,nDofs);
        embeddedHessian = 0;
        
        embeddedHessian[0][0] = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients[1],
                                                                                                coefficients[0]).matrix();

        embeddedHessian[1][1] = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients[0],
                                                                                                coefficients[1]).matrix();

        embeddedHessian[0][1] = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(coefficients[0],
                                                                                                        coefficients[1]);

        embeddedHessian[1][0] = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(coefficients[1],
                                                                                                        coefficients[0]);

        Matrix<FieldMatrix<double,spaceDim,spaceDim> > hessian(nDofs,nDofs);
        
        // transform to local tangent space bases
        std::vector<Dune::FieldMatrix<double,spaceDim,embeddedSpaceDim> > orthonormalFrames(nDofs);
        std::vector<Dune::FieldMatrix<double,embeddedSpaceDim,spaceDim> > orthonormalFramesTransposed(nDofs);

        for (size_t j=0; j<nDofs; j++) {
            orthonormalFrames[j] = coefficients[j].orthonormalFrame();

            for (int k=0; k<embeddedSpaceDim; k++)
                for (int l=0; l<spaceDim; l++)
                    orthonormalFramesTransposed[j][k][l] = orthonormalFrames[j][l][k];

        }

        for (size_t j=0; j<nDofs; j++)
            for (size_t k=0; k<nDofs; k++) {

                Dune::FieldMatrix<double,spaceDim,embeddedSpaceDim> tmp;
                Dune::FMatrixHelp::multMatrix(orthonormalFrames[j],embeddedHessian[j][k],tmp);
                hessian[j][k] = tmp.rightmultiplyany(orthonormalFramesTransposed[k]);

        }

        std::cout << "hessian:" << std::endl;
        printmatrix(std::cout, hessian, "hessian", "--");
        
        ///////////////////////////////////////////////////////////////////////////////////
        //  Abort if there is a difference
        ///////////////////////////////////////////////////////////////////////////////////
        Matrix<FieldMatrix<double,spaceDim,spaceDim> > difference = hessian;
        difference -= fdHessian;
        
        if (difference.infinity_norm() > 1e-4)
            assert(false);
        
    }

}


int main(int argc, char** argv)
{
    testHessian<RealTuple<double,1>, 1>();
    testHessian<UnitVector<double,3>, 1>();
}
