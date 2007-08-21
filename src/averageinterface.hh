#ifndef AVERAGE_INTERFACE_HH
#define AVERAGE_INTERFACE_HH

#include <dune/common/fmatrix.hh>
#include "svd.hh"

// template parameter dim is only there do make it compile when dim!=3
template <class T, int dim>
Dune::FieldVector<T,dim> crossProduct(const Dune::FieldVector<T,dim>& a, const Dune::FieldVector<T,dim>& b)
{
    Dune::FieldVector<T,dim> r;
    r[0] = a[1]*b[2] - a[2]*b[1];
    r[1] = a[2]*b[0] - a[0]*b[2];
    r[2] = a[0]*b[1] - a[1]*b[0];
    return r;
}

// Given a resultant force and torque (from a rod problem), this method computes the corresponding
// Neumann data for a 3d elasticity problem.
template <class GridType>
void computeAveragePressure(const Dune::FieldVector<double,GridType::dimension>& resultantForce,
                            const Dune::FieldVector<double,GridType::dimension>& resultantTorque,
                            const BoundaryPatch<GridType>& interface,
                            const Configuration& crossSection,
                            Dune::BlockVector<Dune::FieldVector<double, GridType::dimension> >& pressure)
{
    const GridType& grid = interface.getGrid();
    const int level      = interface.level();
    const typename GridType::Traits::LevelIndexSet& indexSet = grid.levelIndexSet(level);
    const int dim        = GridType::dimension;
    typedef typename GridType::ctype ctype;

    typedef typename GridType::template Codim<dim>::LevelIterator VertexIterator;

    // set up output array
    pressure.resize(indexSet.size(dim));
    pressure = 0;
    
    ctype area = interface.area();
    
    VertexIterator vIt    = indexSet.template begin<dim, Dune::All_Partition>();
    VertexIterator vEndIt = indexSet.template end<dim, Dune::All_Partition>();
    
    for (; vIt!=vEndIt; ++vIt) {

        int vIdx = indexSet.index(*vIt);

        if (interface.containsVertex(vIdx)) {

            // force part
            pressure[vIdx] = resultantForce;
            pressure[vIdx] /= area;

            // torque part
            double x = (vIt->geometry()[0] - crossSection.r) * crossSection.q.director(0);
            double y = (vIt->geometry()[0] - crossSection.r) * crossSection.q.director(1);
            
            Dune::FieldVector<double,3> localTorque;
            for (int i=0; i<3; i++)
                localTorque[i] = resultantTorque * crossSection.q.director(i);

            // add it up
            pressure[vIdx][0] += -2 * M_PI * localTorque[2] * y / (area*area);
            pressure[vIdx][1] +=  2 * M_PI * localTorque[2] * x / (area*area);
            pressure[vIdx][2] +=  4 * M_PI * localTorque[0] * y / (area*area);
            pressure[vIdx][2] += -4 * M_PI * localTorque[1] * x / (area*area);

        }

    }

    // /////////////////////////////////////////////////////////////////////////////////////
    //   Compute the overall force and torque to see whether the preceding code is correct
    // /////////////////////////////////////////////////////////////////////////////////////

    Dune::FieldVector<double,3> outputForce(0), outputTorque(0);
    Dune::LeafP1Function<GridType,double,dim> pressureFunction(grid);
    *pressureFunction = pressure;

    typename GridType::template Codim<0>::LevelIterator eIt    = indexSet.template begin<0,Dune::All_Partition>();
    typename GridType::template Codim<0>::LevelIterator eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;
            
            const Dune::QuadratureRule<double, dim-1>& quad 
                = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
            
            for (size_t qp=0; qp<quad.size(); qp++) {
                
                // Local position of the quadrature point
                const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                
                const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                
                // Evaluate function
                Dune::FieldVector<double,dim> localPressure;
                pressureFunction.evalalllocal(*eIt, nIt.intersectionSelfLocal().global(quadPos), localPressure);
                
                // Sum up the total force
                outputForce.axpy(quad[qp].weight()*integrationElement, localPressure);

                // Sum up the total torque   \int (x - x_0) \times f dx
                Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);
                outputTorque.axpy(quad[qp].weight()*integrationElement, 
                                  crossProduct(worldPos - crossSection.r, localPressure));

            }

        }

    }

    std::cout << "Output force:  " << outputForce << std::endl;
    std::cout << "Output torque: " << outputTorque << "      " << resultantTorque[0]/outputTorque[0] << std::endl;

}

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
                std::vector<FieldVector<double, dim> > shapeGrads(eIt->geometry().corners());
                
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
    std::cout << "deformationGradient: " << std::endl << deformationGradient << std::endl;

    // Get the rotational part of the deformation gradient by performing a 
    // polar composition.
    FieldVector<double,dim> W;
    FieldMatrix<double,dim,dim> VT;

    // returns a decomposition U W VT, where U is returned in the first argument
    svdcmp<double,dim,dim>(deformationGradient, W, VT);

    deformationGradient.rightmultiply(VT);

    // deformationGradient now contains the orthogonal part of the polar decomposition
    assert( std::abs(1-deformationGradient.determinant()) < 1e-3);

    average.q.set(deformationGradient);
}

#endif
