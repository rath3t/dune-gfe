#ifndef GFE_DIFFERENCE_NORM_SQUARED_HH
#define GFE_DIFFERENCE_NORM_SQUARED_HH

/** \file
    \brief Helper method which computes the energy norm (squared) of the difference of
    the fe functions on two different refinements of the same base grid.
*/

#include <dune/geometry/type.hh>
#include <dune/common/fvector.hh>

#include <dune/grid/common/mcmgmapper.hh>

#include <dune/fufem/assemblers/localoperatorassembler.hh>

#include <dune/gfe/globalgeodesicfefunction.hh>

template <class Basis, class TargetSpace>
class GFEDifferenceNormSquared {

    typedef typename Basis::GridView GridView;
    typedef typename Basis::GridView::Grid GridType;
    
    // Grid dimension
    const static int dim = GridType::dimension;

    /** \brief Check whether given local coordinates are contained in the reference element
        \todo This method exists in the Dune grid interface!  But we need the eps.
    */
    static bool checkInside(const Dune::GeometryType& type,
                            const Dune::FieldVector<double, dim> &loc,
                            double eps)
    {
        switch (type.dim()) {
            
        case 0: // vertex
            return false;
        case 1: // line
            return -eps <= loc[0] && loc[0] <= 1+eps;
        case 2: 
            
            if (type.isSimplex()) {
                return -eps <= loc[0] && -eps <= loc[1] && (loc[0]+loc[1])<=1+eps;
            } else if (type.isCube()) {
                return -eps <= loc[0] && loc[0] <= 1+eps
                    && -eps <= loc[1] && loc[1] <= 1+eps;
            } else {
                DUNE_THROW(Dune::GridError, "checkInside():  ERROR:  Unknown type " << type << " found!");
            }
            
        case 3: 
            if (type.isSimplex()) {
                return -eps <= loc[0] && -eps <= loc[1] && -eps <= loc[2]
                    && (loc[0]+loc[1]+loc[2]) <= 1+eps;
            } else if (type.isPyramid()) {
                return -eps <= loc[0] && -eps <= loc[1] && -eps <= loc[2]
                    && (loc[0]+loc[2]) <= 1+eps
                    && (loc[1]+loc[2]) <= 1+eps;
            } else if (type.isPrism()) {
                return -eps <= loc[0] && -eps <= loc[1] 
                    && (loc[0]+loc[1])<= 1+eps
                    && -eps <= loc[2] && loc[2] <= 1+eps;
            } else if (type.isCube()) {
                return -eps <= loc[0] && loc[0] <= 1+eps
                    && -eps <= loc[1] && loc[1] <= 1+eps
                    && -eps <= loc[2] && loc[2] <= 1+eps;
            }else {
                DUNE_THROW(Dune::GridError, "checkInside():  ERROR:  Unknown type " << type << " found!");
            }
        default:
            DUNE_THROW(Dune::GridError, "checkInside():  ERROR:  Unknown type " << type << " found!");
        }
        
    }
    
    
    /** \brief Coordinate function in one variable, constant in the others 
 
    This is used to extract the positions of the Lagrange nodes.
    */
    struct CoordinateFunction
        : public Dune::VirtualFunction<Dune::FieldVector<double,dim>, Dune::FieldVector<double,1> >
    {
        CoordinateFunction(int d)
        : d_(d)
        {}
    
        void evaluate(const Dune::FieldVector<double, dim>& x, Dune::FieldVector<double,1>& out) const {
            out[0] = x[d_];
        }

        //
        int d_;
    };

public:

    static void interpolate(const GridType& sourceGrid, 
                            const std::vector<TargetSpace>& source,
                            const GridType& targetGrid, 
                            std::vector<TargetSpace>& target)
    {
        // Create a leaf function, which we need to be able to call 'evalall()'
        Basis sourceBasis(sourceGrid.leafView());
        GlobalGeodesicFEFunction<Basis,TargetSpace> sourceFunction(sourceBasis, source);

        Basis targetBasis(targetGrid.leafView());

        // ///////////////////////////////////////////////////////////////////////////////////////////
        //   Prolong the adaptive solution onto the uniform grid in order to make it comparable
        // ///////////////////////////////////////////////////////////////////////////////////////////
        
        target.resize(targetBasis.size());
        
        // handle each dof only once
        Dune::BitSetVector<1> handled(targetBasis.size(), false);

        typename GridView::template Codim<0>::Iterator eIt    = targetGrid.template leafbegin<0>();
        typename GridView::template Codim<0>::Iterator eEndIt = targetGrid.template leafend<0>();
        
        // Loop over all dofs by looping over all elements
        for (; eIt!=eEndIt; ++eIt) {

            const typename Basis::LocalFiniteElement& targetLFE = targetBasis.getLocalFiniteElement(*eIt);

            // Generate position of the Lagrange nodes
            std::vector<Dune::FieldVector<double,dim> > lagrangeNodes(targetLFE.localBasis().size());
        
            for (int i=0; i<dim; i++) {
                CoordinateFunction lFunction(i);
                std::vector<Dune::FieldVector<double,1> > coordinates;
                targetLFE.localInterpolation().interpolate(lFunction, coordinates);
            
                for (size_t j=0; j<coordinates.size(); j++)
                    lagrangeNodes[j][i] = coordinates[j];
            
            }

            for (size_t i=0; i<targetLFE.localCoefficients().size(); i++) {

                typename GridType::template Codim<0>::EntityPointer element(eIt);

                Dune::FieldVector<double,dim> pos = lagrangeNodes[i];
                
                if (handled[targetBasis.index(*eIt,i)][0])
                    continue;

                handled[targetBasis.index(*eIt,i)] = true;

                assert(checkInside(element->type(), pos, 1e-7));
                
                // ////////////////////////////////////////////////////////////////////
                // Get an element on the coarsest grid which contains the vertex and
                // its local coordinates there
                // ////////////////////////////////////////////////////////////////////
                while (element->level() != 0){
                    
                    pos = element->geometryInFather().global(pos);
                    element = element->father();
                    
                    assert(checkInside(element->type(), pos, 1e-7));
                }
                
                // ////////////////////////////////////////////////////////////////////
                //   Find the corresponding coarse grid element on the adaptive grid.
                //   This is a linear algorithm, but we expect the coarse grid to be small.
                // ////////////////////////////////////////////////////////////////////
                Dune::LevelMultipleCodimMultipleGeomTypeMapper<GridType,Dune::MCMGElementLayout> uniformP0Mapper (targetGrid,  0);
                Dune::LevelMultipleCodimMultipleGeomTypeMapper<GridType,Dune::MCMGElementLayout> adaptiveP0Mapper(sourceGrid, 0);
                int coarseIndex = uniformP0Mapper.map(*element);
                
                typename GridType::template Codim<0>::LevelIterator adaptEIt    = sourceGrid.template lbegin<0>(0);
                typename GridType::template Codim<0>::LevelIterator adaptEEndIt = sourceGrid.template lend<0>(0);
                
                for (; adaptEIt!=adaptEEndIt; ++adaptEIt)
                    if (adaptiveP0Mapper.map(*adaptEIt) == coarseIndex)
                        break;
                
                element = adaptEIt;
                
                // ////////////////////////////////////////////////////////////////////////
                //   Find a corresponding point (not necessarily vertex) on the leaf level
                //   of the adaptive grid.
                // ////////////////////////////////////////////////////////////////////////
                
                while (!element->isLeaf()) {
                    

                    typename GridType::template Codim<0>::Entity::HierarchicIterator hIt    = element->hbegin(element->level()+1);
                    typename GridType::template Codim<0>::Entity::HierarchicIterator hEndIt = element->hend(element->level()+1);
                    
                    Dune::FieldVector<double,dim> childPos;
                    assert(checkInside(element->type(), pos, 1e-7));
                    
                    for (; hIt!=hEndIt; ++hIt) {
                        
                        childPos = hIt->geometryInFather().local(pos);
                        if (checkInside(hIt->type(), childPos, 1e-7))
                            break;
                        
                    }
                    
                    assert(hIt!=hEndIt);
                    element = hIt;
                    pos     = childPos;
                    
                }
                
                // ////////////////////////////////////////////////////////////////////////
                //   Sample adaptive function
                // ////////////////////////////////////////////////////////////////////////
                sourceFunction.evaluateLocal(*element, pos, 
                                            target[targetBasis.index(*eIt,i)]);
                
            }
            
        }
        
    }


    template <int blocksize>
    static double computeNormSquared(const GridType& grid, 
                              const Dune::BlockVector<Dune::FieldVector<double,blocksize> >& x,
                              const LocalOperatorAssembler<GridType,
                              typename Basis::LocalFiniteElement,
                              typename Basis::LocalFiniteElement,
                              Dune::FieldMatrix<double,1,1> >* localStiffness)
    {
        // ///////////////////////////////////////////////////////////////////////////////////////////
        //   Compute the energy norm 
        // ///////////////////////////////////////////////////////////////////////////////////////////
        
        double energyNormSquared = 0;
        Basis basis(grid.leafView());
        Dune::Matrix<Dune::FieldMatrix<double,1,1> > localMatrix;
        
        typename GridType::template Codim<0>::LeafIterator eIt    = grid.template leafbegin<0>();
        typename GridType::template Codim<0>::LeafIterator eEndIt = grid.template leafend<0>();
        
        for (; eIt!=eEndIt; ++eIt) {

            size_t nLocalDofs = basis.getLocalFiniteElement(*eIt).localCoefficients().size();
        
            localMatrix.setSize(nLocalDofs,nLocalDofs);
            
            localStiffness->assemble(*eIt, localMatrix,
                                     basis.getLocalFiniteElement(*eIt),
                                     basis.getLocalFiniteElement(*eIt));
            
            for (size_t i=0; i<nLocalDofs; i++)
                for (size_t j=0; j<nLocalDofs; j++)
                    energyNormSquared += localMatrix[i][j][0][0] * (x[basis.index(*eIt,i)] * x[basis.index(*eIt,j)]);
            
        }
        
        return energyNormSquared;
    }

    static double compute(const GridType& uniformGrid, 
                          const std::vector<TargetSpace>& uniformSolution,
                          const GridType& adaptiveGrid, 
                          const std::vector<TargetSpace>& adaptiveSolution,
                          const LocalOperatorAssembler<GridType,
                          typename Basis::LocalFiniteElement,
                          typename Basis::LocalFiniteElement,
                          Dune::FieldMatrix<double,1,1> >* localStiffness)
    {
        std::vector<TargetSpace> uniformAdaptiveSolution;

        interpolate(adaptiveGrid, adaptiveSolution, uniformGrid, uniformAdaptiveSolution);

        // Compute difference in the embedding space
        Dune::BlockVector<typename TargetSpace::CoordinateType> difference(uniformSolution.size());
        
        for (size_t i=0; i<difference.size(); i++)
            difference[i] = uniformAdaptiveSolution[i].globalCoordinates() - uniformSolution[i].globalCoordinates();

        //std::cout << "new difference:\n" << difference << std::endl;
//         std::cout << "new projected:\n" << std::endl;
//         for (size_t i=0; i<difference.size(); i++)
//             std::cout << uniformAdaptiveSolution[i].globalCoordinates() << std::endl;
            
        return computeNormSquared(uniformGrid, difference, localStiffness);
    }
};

#endif
