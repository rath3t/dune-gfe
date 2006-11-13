#ifndef AVERAGE_INTERFACE_HH
#define AVERAGE_INTERFACE_HH

#include <dune/common/fmatrix.hh>
#include "svd.hh"

template <class GridType>
void computeAverageInterface(const BoundaryPatch<GridType>& interface,
                             const Dune::BlockVector<Dune::FieldVector<double,GridType::dimension> > deformation,
                             Configuration& average)
{
    using namespace Dune;

    typedef typename GridType::template Codim<0>::LevelIterator ElementIterator;
    typedef typename GridType::template Codim<0>::Entity EntityType;
    typedef typename EntityType::LevelIntersectionIterator NeighborIterator;

    const GridType& grid = interface.getGrid();
    const int level      = interface.level();
    const typename GridType::Traits::LevelIndexSet& indexSet = grid.levelIndexSet(level);
    const int dim        = GridType::dimension;

    // ///////////////////////////////////////////
    //   Initialize output configuration
    // ///////////////////////////////////////////
    average.r = 0;
    
    double interfaceArea = 0;
    FieldMatrix<double,dim,dim> deformationGradient(0);

    // ///////////////////////////////////////////
    //   Loop and integrate over the interface
    // ///////////////////////////////////////////
    ElementIterator eIt    = grid.template lbegin<0>(level);
    ElementIterator eEndIt = grid.template lend<0>(level);
    for (; eIt!=eEndIt; ++eIt) {

        NeighborIterator nIt    = eIt->ilevelbegin();
        NeighborIterator nEndIt = eIt->ilevelend();

        for (; nIt!=nEndIt; ++nIt) {

            if (!interface.contains(*eIt, nIt))
                continue;

            const typename NeighborIterator::Geometry& segmentGeometry = nIt.intersectionGlobal();

            const ReferenceElement<double,dim>& refElement = ReferenceElements<double, dim>::general(eIt->geometry().type());
            int nDofs = refElement.size(nIt.numberInSelf(),1,dim);

            // Get quadrature rule
            const QuadratureRule<double, dim-1>& quad = QuadratureRules<double, dim-1>::rule(segmentGeometry.type(), dim-1);

            // Get set of shape functions on this segment
            const typename LagrangeShapeFunctionSetContainer<double,double,dim>::value_type& sfs
                = LagrangeShapeFunctions<double,double,dim>::general(eIt->geometry().type(),1);

            /* Loop over all integration points */
            for (int ip=0; ip<quad.size(); ip++) {
                
                // Local position of the quadrature point
                //const FieldVector<double,dim-1>& quadPos = quad[ip].position();
                const FieldVector<double,dim> quadPos = nIt.intersectionSelfLocal().global(quad[ip].position());
                
                const double integrationElement = segmentGeometry.integrationElement(quad[ip].position());

                // Evaluate base functions
                FieldVector<double,dim> posAtQuadPoint(0);

                for(int i=0; i<eIt->geometry().corners(); i++) {

                    int idx = indexSet.template subIndex<dim>(*eIt, i);

                    // Deformation at the quadrature point 
                    posAtQuadPoint.axpy(sfs[i].evaluateFunction(0,quadPos), deformation[idx]);
                }

                const FieldMatrix<double,dim,dim>& inv = eIt->geometry().jacobianInverseTransposed(quadPos);
                
                /* Compute the weight of the current integration point */
                double weight = quad[ip].weight() * integrationElement;
                
                /**********************************************/
                /* compute gradients of the shape functions   */
                /**********************************************/
                Array<FieldVector<double, dim> > shapeGrads(eIt->geometry().corners());
                
                for (int dof=0; dof<eIt->geometry().corners(); dof++) {
                    
                    for (int i=0; i<dim; i++)
                        shapeGrads[dof][i] = sfs[dof].evaluateDerivative(0, i, quadPos);
                    
                    // multiply with jacobian inverse 
                    FieldVector<double,dim> tmp(0);
                    inv.umv(shapeGrads[dof], tmp);
                    shapeGrads[dof] = tmp;
                    //std::cout << "Gradient " << dof << ": " << shape_grads[dof] << std::endl;
                }

                /****************************************************/
                // The deformation gradient of the deformation
                // in formulas: F(i,j) = $\partial \phi_i / \partial x_j$
                // or F(i,j) = Id + $\partial u_i / \partial x_j$
                /****************************************************/
                FieldMatrix<double, dim, dim> F;
                for (int i=0; i<dim; i++) {
                    
                    for (int j=0; j<dim; j++) {
                        
                        F[i][j] = (i==j) ? 1 : 0;
                        
                        for (int k=0; k<eIt->geometry().corners(); k++)
                            F[i][j] += deformation[indexSet.template subIndex<dim>(*eIt, k)][i]*shapeGrads[k][j];
                        
                    }
                    
                }

                interfaceArea += quad[ip].weight() * integrationElement;

                average.r.axpy(quad[ip].weight() * integrationElement, posAtQuadPoint);

                F *= quad[ip].weight();
                deformationGradient += F;

            }

        }

    }


    // average deformation of the interface is the integral over its
    // deformation divided by its area
    average.r /= interfaceArea;

    // average deformation is the integral over the deformation gradient
    // divided by its area
    deformationGradient /= interfaceArea;
    std::cout << "deformationGradient: " << deformationGradient << std::endl;

    // Get the rotational part of the deformation gradient by performing a 
    // polar composition.
    FieldVector<double,dim> W;
    FieldMatrix<double,dim,dim> VT;
    svdcmp<double,dim,dim>(deformationGradient, W, VT);

    deformationGradient.rightmultiply(VT);
    std::cout << "determinant: " << deformationGradient.determinant() << "  (should be 1)\n";
    

    // average orientation not implemented yet
    average.q = Quaternion<double>::identity();
}

#endif
