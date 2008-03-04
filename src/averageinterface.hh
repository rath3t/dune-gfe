#ifndef AVERAGE_INTERFACE_HH
#define AVERAGE_INTERFACE_HH

#include <dune/common/fmatrix.hh>
#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>

#include <dune/ag-common/dgindexset.hh>
#include <dune/ag-common/crossproduct.hh>
#include "svd.hh"
#include "lapackpp.h"
#undef max

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
    typedef double field_type;

    typedef typename GridType::template Codim<dim>::LevelIterator VertexIterator;

    // Get total interface area
    ctype area = interface.area();

    // set up output array
    DGIndexSet<GridType> dgIndexSet(grid,level);
    dgIndexSet.setup(grid,level);
    pressure.resize(dgIndexSet.size());
    pressure = 0;
    
    typename GridType::template Codim<0>::LevelIterator eIt    = indexSet.template begin<0,Dune::All_Partition>();
    typename GridType::template Codim<0>::LevelIterator eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;

            const Dune::LagrangeShapeFunctionSet<ctype, field_type, dim-1>& baseSet
                = Dune::LagrangeShapeFunctions<ctype, field_type, dim-1>::general(nIt.intersectionGlobal().type(),1);

            // four rows because a face may have no more than four vertices
            Dune::FieldVector<double,4> mu(0);
            Dune::FieldVector<double,3> mu_tilde[4][3];
            
            for (int i=0; i<4; i++)
                for (int j=0; j<3; j++)
                    mu_tilde[i][j] = 0;

            for (int i=0; i<nIt.intersectionGlobal().corners(); i++) {
                
                const Dune::QuadratureRule<double, dim-1>& quad 
                    = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
                
                for (size_t qp=0; qp<quad.size(); qp++) {
                    
                    // Local position of the quadrature point
                    const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                    
                    const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                    
                    // \mu_i = \int_t \varphi_i \ds
                    mu[i] += quad[qp].weight() * integrationElement * baseSet[i].evaluateFunction(0,quadPos);
                    
                    // \tilde{\mu}_i^j = \int_t \varphi_i \times (x - x_0) \ds
                    Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);

                    for (int j=0; j<dim; j++) {

                        // Vector-valued basis function
                        Dune::FieldVector<double,dim> phi_i(0);
                        phi_i[j] = baseSet[i].evaluateFunction(0,quadPos);
                        
                        mu_tilde[i][j].axpy(quad[qp].weight() * integrationElement,
                                            crossProduct(worldPos-crossSection.r, phi_i));

                    }
                    
                }
                
            }

            // Set up matrix
#if 0  // DUNE style
            Dune::Matrix<Dune::FieldMatrix<double,1,1> > matrix(6, 3*baseSet.size());
            matrix = 0;
            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    matrix[j][i*3+j] = mu[i];

            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    for (int k=0; k<3; k++)
                        matrix[3+k][3*i+j] = mu_tilde[i][j][k];

            Dune::BlockVector<Dune::FieldVector<double,1> > u(3*baseSet.size());
            Dune::FieldVector<double,6> b;

            // Scale the resultant force and torque with this segments area percentage.
            // That way the resulting pressure gets distributed fairly uniformly.
            ctype segmentArea = nIt.intersectionGlobal().volume() / area;

            for (int i=0; i<3; i++) {
                b[i]   = resultantForce[i] * segmentArea;
                b[i+3] = resultantTorque[i] * segmentArea;
            }

            matrix.solve(u,b);

#else   // LaPack++ style
            LaGenMatDouble matrix(6, 3*baseSet.size());
            matrix = 0;
            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    matrix(j, i*3+j) = mu[i];

            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    for (int k=0; k<3; k++)
                        matrix(3+k, 3*i+j) = mu_tilde[i][j][k];

            LaVectorDouble u(3*baseSet.size());
            LaVectorDouble b(6);

            // Scale the resultant force and torque with this segments area percentage.
            // That way the resulting pressure gets distributed fairly uniformly.
            ctype segmentArea = nIt.intersectionGlobal().volume() / area;

            for (int i=0; i<3; i++) {
                b(i)   = resultantForce[i] * segmentArea;
                b(i+3) = resultantTorque[i] * segmentArea;
            }

            LaLinearSolve(matrix, u, b);
#endif
//             std::cout << b << std::endl;
//             std::cout << matrix << std::endl;
            //std::cout << u << std::endl;

            for (int i=0; i<baseSet.size(); i++)
                for (int j=0; j<3; j++)
                    pressure[dgIndexSet(*eIt, nIt.numberInSelf())+i][j]   = u(i*3+j);

        }

    }


    // /////////////////////////////////////////////////////////////////////////////////////
    //   Compute the overall force and torque to see whether the preceding code is correct
    // /////////////////////////////////////////////////////////////////////////////////////
#ifdef NDEBUG
    Dune::FieldVector<double,3> outputForce(0), outputTorque(0);

    eIt    = indexSet.template begin<0,Dune::All_Partition>();
    eEndIt = indexSet.template end<0,Dune::All_Partition>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nIt    = eIt->ilevelbegin();
        typename GridType::template Codim<0>::Entity::LevelIntersectionIterator nEndIt = eIt->ilevelend();
        
        for (; nIt!=nEndIt; ++nIt) {
            
            if (!interface.contains(*eIt,nIt))
                continue;

            const Dune::LagrangeShapeFunctionSet<double, double, dim-1>& baseSet
                = Dune::LagrangeShapeFunctions<double, double, dim-1>::general(nIt.intersectionGlobal().type(),1);
            
            const Dune::QuadratureRule<double, dim-1>& quad 
                = Dune::QuadratureRules<double, dim-1>::rule(nIt.intersectionGlobal().type(), dim-1);
            
            for (size_t qp=0; qp<quad.size(); qp++) {
                
                // Local position of the quadrature point
                const Dune::FieldVector<double,dim-1>& quadPos = quad[qp].position();
                
                const double integrationElement         = nIt.intersectionGlobal().integrationElement(quadPos);
                
                // Evaluate function
                Dune::FieldVector<double,dim> localPressure(0);
                
                for (size_t i=0; i<baseSet.size(); i++) 
                    localPressure.axpy(baseSet[i].evaluateFunction(0,quadPos),
                                       pressure[dgIndexSet(*eIt,nIt.numberInSelf())+i]);


                // Sum up the total force
                outputForce.axpy(quad[qp].weight()*integrationElement, localPressure);

                // Sum up the total torque   \int (x - x_0) \times f dx
                Dune::FieldVector<double,dim> worldPos = nIt.intersectionGlobal().global(quadPos);
                outputTorque.axpy(quad[qp].weight()*integrationElement, 
                                  crossProduct(worldPos - crossSection.r, localPressure));

            }

        }

    }

    outputForce  -= resultantForce;
    outputTorque -= resultantTorque;
    assert( outputForce.infinity_norm() < 1e-6 );
    assert( outputTorque.infinity_norm() < 1e-6 );
//     std::cout << "Output force:  " << outputForce << std::endl;
//     std::cout << "Output torque: " << outputTorque << "      " << resultantTorque[0]/outputTorque[0] << std::endl;
#endif

}

template <class GridType>
void averageSurfaceDGFunction(const GridType& grid,
                              const Dune::BlockVector<Dune::FieldVector<double,GridType::dimension> >& dgFunction,
                              Dune::BlockVector<Dune::FieldVector<double,GridType::dimension> >& p1Function,
                              const DGIndexSet<GridType>& dgIndexSet)
{
    const int dim = GridType::dimension;

    const typename GridType::Traits::LeafIndexSet& indexSet = grid.leafIndexSet();

    p1Function.resize(indexSet.size(dim));
    p1Function = 0;

    std::vector<int> counter(indexSet.size(dim), 0);

    typename GridType::template Codim<0>::LeafIterator eIt    = grid.template leafbegin<0>();
    typename GridType::template Codim<0>::LeafIterator eEndIt = grid.template leafend<0>();

    for (; eIt!=eEndIt; ++eIt) {

        typename GridType::template Codim<0>::Entity::LeafIntersectionIterator nIt    = eIt->ileafbegin();
        typename GridType::template Codim<0>::Entity::LeafIntersectionIterator nEndIt = eIt->ileafend();

        for (; nIt!=nEndIt; ++nIt) {

            if (!nIt.boundary())
                continue;

            const Dune::ReferenceElement<double,dim>& refElement 
                = Dune::ReferenceElements<double, dim>::general(eIt->type());

            for (int i=0; i<refElement.size(nIt.numberInSelf(),1,dim); i++) {

                int idxInElement = refElement.subEntity(nIt.numberInSelf(),1, i, dim);
                
                p1Function[indexSet.template subIndex<dim>(*eIt,idxInElement)] 
                    += dgFunction[dgIndexSet(*eIt,nIt.numberInSelf())+i];
                
                counter[indexSet.template subIndex<dim>(*eIt,idxInElement)]++;
                
            }

        }

    }

    for (int i=0; i<p1Function.size(); i++)
        if (counter[i]!=0)
            p1Function[i] /= counter[i];

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

            // Get quadrature rule
            const QuadratureRule<double, dim-1>& quad = QuadratureRules<double, dim-1>::rule(segmentGeometry.type(), dim-1);

            // Get set of shape functions on this segment
            const typename LagrangeShapeFunctionSetContainer<double,double,dim>::value_type& sfs
                = LagrangeShapeFunctions<double,double,dim>::general(eIt->type(),1);

            /* Loop over all integration points */
            for (int ip=0; ip<quad.size(); ip++) {
                
                // Local position of the quadrature point
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
                
                /**********************************************/
                /* compute gradients of the shape functions   */
                /**********************************************/
                std::vector<FieldVector<double, dim> > shapeGrads(eIt->geometry().corners());
                
                for (int dof=0; dof<eIt->geometry().corners(); dof++) {
                    
                    FieldVector<double,dim> tmp;
                    for (int i=0; i<dim; i++)
                        tmp[i] = sfs[dof].evaluateDerivative(0, i, quadPos);
                    
                    // multiply with jacobian inverse 
                    shapeGrads[dof] = 0;
                    inv.umv(tmp, shapeGrads[dof]);
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

                // Sum up interface area
                interfaceArea += quad[ip].weight() * integrationElement;

                // Sum up average displacement
                average.r.axpy(quad[ip].weight() * integrationElement, posAtQuadPoint);

                // Sum up average deformation gradient
                for (int i=0; i<dim; i++)
                    for (int j=0; j<dim; j++)
                        deformationGradient[i][j] += F[i][j] * quad[ip].weight() * integrationElement;

            }

        }

    }


    // average deformation of the interface is the integral over its
    // deformation divided by its area
    average.r /= interfaceArea;

    // average deformation is the integral over the deformation gradient
    // divided by its area
    deformationGradient /= interfaceArea;
    //std::cout << "deformationGradient: " << std::endl << deformationGradient << std::endl;

    // Get the rotational part of the deformation gradient by performing a 
    // polar composition.
    FieldVector<double,dim> W;
    FieldMatrix<double,dim,dim> V, VT, U;

    FieldMatrix<double,dim,dim> deformationGradientBackup = deformationGradient;

#if 1
    // returns a decomposition U W VT, where U is returned in the first argument
    svdcmp<double,dim,dim>(deformationGradient, W, V);

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            VT[i][j] = V[j][i];

    deformationGradient.rightmultiply(VT);
#else
    lapackSVD(deformationGradientBackup, U, W, VT);
    deformationGradient = U;
    deformationGradient.rightmultiply(VT);
#endif
    //std::cout << deformationGradient << std::endl;

    // deformationGradient now contains the orthogonal part of the polar decomposition
    assert( std::abs(1-deformationGradient.determinant()) < 1e-3);

    average.q.set(deformationGradient);
}

#endif
