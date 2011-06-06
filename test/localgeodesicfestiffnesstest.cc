#include "config.h"

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/localgeodesicfestiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

typedef UnitVector<3> TargetSpace;

using namespace Dune;








template<class GridView, class TargetSpace>
class TestEnergyLocalStiffness 
    : public LocalGeodesicFEStiffness<GridView,TargetSpace>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};

public:
    
    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::size };

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const std::vector<TargetSpace>& localSolution) const;
               
    TestEnergyLocalStiffness(TargetSpace anchor, int idx)
    : anchor_(anchor),
      idx_(idx)
      {}

    TargetSpace anchor_;

    int idx_;
    
};

template <class GridView, class TargetSpace>
typename TestEnergyLocalStiffness<GridView, TargetSpace>::RT TestEnergyLocalStiffness<GridView, TargetSpace>::
energy(const Entity& element,
       const std::vector<TargetSpace>& localSolution) const
{
    return TargetSpace::distance(anchor_, localSolution[idx_]) * TargetSpace::distance(anchor_, localSolution[idx_]);
}












template <int domainDim>
void testUnitVector3d()
{
    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    //typedef std::conditional<domainDim==1,OneDGrid,UGGrid<domainDim> >::type GridType;
    typedef OneDGrid GridType;

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

    const GridType* grid = factory.createGrid();
    

    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////
    
    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);
    
    int nTestPoints = testPoints.size();
    
    TargetSpace anchor = testPoints[0];
    int idx = 0;
    
    TestEnergyLocalStiffness<typename GridType::LeafGridView, TargetSpace> assembler(anchor, idx);

    // Set up elements of S^2
    std::vector<TargetSpace> coefficients(domainDim+1);

    MultiIndex<domainDim+1> index(nTestPoints);
    int numIndices = index.cycle();
    
    size_t nDofs = domainDim+1;

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        assembler.assembleHessian(*grid->template leafbegin<0>(), coefficients);
        
        Matrix<FieldMatrix<double,2,2> > fdHessian = assembler.A_;
        
        std::cout << "fdHessian:\n";
        printmatrix(std::cout, fdHessian, "fdHessian", "--");
        
        FieldMatrix<double,3,3> embeddedHessian = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(anchor, coefficients[idx]);
        FieldMatrix<double,2,2> hessian;
        
        // transform to local tangent space bases
        const int blocksize = 2;
        const int embeddedBlocksize = 3;
        std::vector<Dune::FieldMatrix<double,blocksize,embeddedBlocksize> > orthonormalFrames(nDofs);
        std::vector<Dune::FieldMatrix<double,embeddedBlocksize,blocksize> > orthonormalFramesTransposed(nDofs);

        for (size_t j=0; j<nDofs; j++) {
            orthonormalFrames[j] = coefficients[j].orthonormalFrame();

            for (int k=0; k<embeddedBlocksize; k++)
                for (int l=0; l<blocksize; l++)
                    orthonormalFramesTransposed[j][k][l] = orthonormalFrames[j][l][k];

        }

#if 0
        for (size_t j=0; j<nDofs; j++)
            for (size_t k=0; k<nDofs; k++) {

                Dune::FieldMatrix<double,blocksize,embeddedBlocksize> tmp;
                Dune::FMatrixHelp::multMatrix(orthonormalFrames[j],embeddedHessian[j][k],tmp);
                hessian[j][k] = tmp.rightmultiplyany(orthonormalFramesTransposed[k]);

        }
#else
        Dune::FieldMatrix<double,blocksize,embeddedBlocksize> tmp;
        Dune::FMatrixHelp::multMatrix(orthonormalFrames[idx],embeddedHessian,tmp);
        hessian = tmp.rightmultiplyany(orthonormalFramesTransposed[idx]);

#endif

        std::cout << "hessian:\n" << hessian << std::endl;
        
    }

}


int main(int argc, char** argv)
{
    testUnitVector3d<1>();
}
