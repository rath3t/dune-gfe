#ifndef ASSEMBLER_FINITE_DIFFERENCE_CHECK
#define ASSEMBLER_FINITE_DIFFERENCE_CHECK

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/rodassembler.hh>

#define ABORT_ON_ERROR

void infinitesimalVariation(RigidBodyMotion<double,3>& c, double eps, int i)
{
    if (i<3)
        c.r[i] += eps;
    else {
        Dune::FieldVector<double,3> axial(0);
        axial[i-3] = eps;
        SkewMatrix<double,3> variation(axial);
        c.q = c.q.mult(Rotation<double,3>::exp(variation));
    }
}

template <class GridType>
void strainFD(const std::vector<RigidBodyMotion<double,3> >& x, 
              double pos,
              Dune::array<Dune::FieldMatrix<double,2,6>, 6>& fdStrainDerivatives,
              const RodAssembler<GridType,3>& assembler) 
{
    assert(x.size()==2);
    double eps = 1e-8;

    typename GridType::template Codim<0>::EntityPointer element = assembler.grid_->template leafbegin<0>();

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<RigidBodyMotion<double,3> > forwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardSolution = x;
    
    for (size_t i=0; i<x.size(); i++) {
        
        Dune::FieldVector<double,6> fdGradient;
        
        for (int j=0; j<6; j++) {
            
            infinitesimalVariation(forwardSolution[i], eps, j);
            infinitesimalVariation(backwardSolution[i], -eps, j);
            
//             fdGradient[j] = (assembler.computeEnergy(forwardSolution) - assembler.computeEnergy(backwardSolution))
//                 / (2*eps);
            Dune::FieldVector<double,6> strain;
            strain = assembler.getStrain(forwardSolution, element, pos);
            strain -= assembler.getStrain(backwardSolution, element, pos);
            strain /= 2*eps;
            
            for (int m=0; m<6; m++)
                fdStrainDerivatives[m][i][j] = strain[m];

            forwardSolution[i] = x[i];
            backwardSolution[i] = x[i];
        }
        
    }

}


template <class GridType>
void strain2ndOrderFD(const std::vector<RigidBodyMotion<double,3> >& x, 
                      double pos,
                      Dune::array<Dune::Matrix<Dune::FieldMatrix<double,6,6> >, 3>& translationDer,
                      Dune::array<Dune::Matrix<Dune::FieldMatrix<double,3,3> >, 3>& rotationDer,
                      const RodAssembler<GridType,3>& assembler) 
{
    assert(x.size()==2);
    double eps = 1e-3;

    typename GridType::template Codim<0>::EntityPointer element = assembler.grid_->template leafbegin<0>();

    for (int m=0; m<3; m++) {
        
        translationDer[m].setSize(2,2);
        translationDer[m] = 0;
        
        rotationDer[m].setSize(2,2);
        rotationDer[m] = 0;

    }

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<RigidBodyMotion<double,3> > forwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardSolution = x;
    
    std::vector<RigidBodyMotion<double,3> > forwardForwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > forwardBackwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardForwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardBackwardSolution = x;

    for (int i=0; i<2; i++) {
        
        for (int j=0; j<3; j++) {

            for (int k=0; k<2; k++) {

                for (int l=0; l<3; l++) {

                    if (i==k && j==l) {

                        infinitesimalVariation(forwardSolution[i], eps, j+3);
                        infinitesimalVariation(backwardSolution[i], -eps, j+3);
            
                        // Second derivative
//                         fdHessian[j][k] = (assembler.computeEnergy(forwardSolution) 
//                                            - 2*assembler.computeEnergy(x) 
//                                            + assembler.computeEnergy(backwardSolution)) / (eps*eps);
                        
                        Dune::FieldVector<double,6> strain;
                        strain = assembler.getStrain(forwardSolution, element, pos);
                        strain += assembler.getStrain(backwardSolution, element, pos);
                        strain.axpy(-2, assembler.getStrain(x, element, pos));
                        strain /= eps*eps;
            
                        for (int m=0; m<3; m++)
                            rotationDer[m][i][k][j][l] = strain[3+m];

                        forwardSolution = x;
                        backwardSolution = x;

                    } else {

                        infinitesimalVariation(forwardForwardSolution[i],   eps, j+3);
                        infinitesimalVariation(forwardForwardSolution[k],   eps, l+3);
                        infinitesimalVariation(forwardBackwardSolution[i],  eps, j+3);
                        infinitesimalVariation(forwardBackwardSolution[k], -eps, l+3);
                        infinitesimalVariation(backwardForwardSolution[i], -eps, j+3);
                        infinitesimalVariation(backwardForwardSolution[k],  eps, l+3);
                        infinitesimalVariation(backwardBackwardSolution[i],-eps, j+3);
                        infinitesimalVariation(backwardBackwardSolution[k],-eps, l+3);


//                         fdHessian[j][k] = (assembler.computeEnergy(forwardForwardSolution)
//                                            + assembler.computeEnergy(backwardBackwardSolution)
//                                            - assembler.computeEnergy(forwardBackwardSolution)
//                                            - assembler.computeEnergy(backwardForwardSolution)) / (4*eps*eps);

                        Dune::FieldVector<double,6> strain;
                        strain  = assembler.getStrain(forwardForwardSolution, element, pos);
                        strain += assembler.getStrain(backwardBackwardSolution, element, pos);
                        strain -= assembler.getStrain(forwardBackwardSolution, element, pos);
                        strain -= assembler.getStrain(backwardForwardSolution, element, pos);
                        strain /= 4*eps*eps;


                        for (int m=0; m<3; m++)
                            rotationDer[m][i][k][j][l] = strain[3+m];
                        
                        forwardForwardSolution   = x;
                        forwardBackwardSolution  = x;
                        backwardForwardSolution  = x;
                        backwardBackwardSolution = x;


                    }

                }

            }
        }
        
    }

}


template <class GridType>
void strain2ndOrderFD2(const std::vector<RigidBodyMotion<double,3> >& x, 
                       double pos,
                       Dune::FieldVector<double,1> shapeGrad[2],
                       Dune::FieldVector<double,1> shapeFunction[2],
                       Dune::array<Dune::Matrix<Dune::FieldMatrix<double,6,6> >, 3>& translationDer,
                       Dune::array<Dune::Matrix<Dune::FieldMatrix<double,3,3> >, 3>& rotationDer,
                       const RodAssembler<GridType,3>& assembler) 
{
    assert(x.size()==2);
    double eps = 1e-3;

    for (int m=0; m<3; m++) {
        
        translationDer[m].setSize(2,2);
        translationDer[m] = 0;
        
        rotationDer[m].setSize(2,2);
        rotationDer[m] = 0;

    }

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<RigidBodyMotion<double,3> > forwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardSolution = x;
    
    for (int k=0; k<2; k++) {
        
        for (int l=0; l<3; l++) {

            infinitesimalVariation(forwardSolution[k], eps, l+3);
            infinitesimalVariation(backwardSolution[k], -eps, l+3);
                
            Dune::array<Dune::FieldMatrix<double,2,6>, 6> forwardStrainDer;
            assembler.strainDerivative(forwardSolution, pos, shapeGrad, shapeFunction, forwardStrainDer);

            Dune::array<Dune::FieldMatrix<double,2,6>, 6> backwardStrainDer;
            assembler.strainDerivative(backwardSolution, pos, shapeGrad, shapeFunction, backwardStrainDer);
            
            for (int i=0; i<2; i++) {
                for (int j=0; j<3; j++) {
                    for (int m=0; m<3; m++) {
                        rotationDer[m][i][k][j][l] = (forwardStrainDer[m][i][j]-backwardStrainDer[m][i][j]) / (2*eps);
                    }
                }
            }

            forwardSolution = x;
            backwardSolution = x;
            
        }
        
    }
    
}





template <class GridType>
void expHessianFD()
{
    using namespace Dune;
    double eps = 1e-3;

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    SkewMatrix<double,3> forward;
    SkewMatrix<double,3> backward;
    
    SkewMatrix<double,3> forwardForward;
    SkewMatrix<double,3> forwardBackward;
    SkewMatrix<double,3> backwardForward;
    SkewMatrix<double,3> backwardBackward;

    for (int i=0; i<3; i++) {
        
        for (int j=0; j<3; j++) {

            Quaternion<double> hessian;

            if (i==j) {

                forward = backward = 0;
                forward.axial()[i]  += eps;
                backward.axial()[i] -= eps;
                
                // Second derivative
                //                         fdHessian[j][k] = (assembler.computeEnergy(forward) 
                //                                            - 2*assembler.computeEnergy(x) 
                //                                            + assembler.computeEnergy(backward)) / (eps*eps);
                
                hessian  = Rotation<double,3>::exp(forward);
                hessian += Rotation<double,3>::exp(backward);
                hessian.axpy(-2, Rotation<double,3>::exp(0,0,0));
                hessian /= eps*eps;
                
            } else {
                
                forwardForward = forwardBackward = 0;
                backwardForward = backwardBackward = 0;

                forwardForward.axial()[i]   += eps;
                forwardForward.axial()[j]   += eps;
                forwardBackward.axial()[i]  += eps;
                forwardBackward.axial()[j]  -= eps;
                backwardForward.axial()[i]  -= eps;
                backwardForward.axial()[j]  += eps;
                backwardBackward.axial()[i] -= eps;
                backwardBackward.axial()[j] -= eps;
                
                
                hessian  = Rotation<double,3>::exp(forwardForward);
                hessian += Rotation<double,3>::exp(backwardBackward);
                hessian -= Rotation<double,3>::exp(forwardBackward);
                hessian -= Rotation<double,3>::exp(backwardForward);
                hessian /= 4*eps*eps;
                
            }

            printf("i: %d,  j: %d    ", i, j);
            std::cout << hessian << std::endl;
            
        }
        
    }
}


template <class GridType>
void gradientFDCheck(const std::vector<RigidBodyMotion<double,3> >& x, 
                     const Dune::BlockVector<Dune::FieldVector<double,6> >& gradient, 
                     const RodAssembler<GridType,3>& assembler)
{
#ifndef ABORT_ON_ERROR
    static int gradientError = 0;
    double maxError = -1;
#endif
    double eps = 1e-6;

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////

    std::vector<RigidBodyMotion<double,3> > forwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardSolution = x;

    for (size_t i=0; i<x.size(); i++) {
        
        Dune::FieldVector<double,6> fdGradient;
        
        for (int j=0; j<6; j++) {
            
            infinitesimalVariation(forwardSolution[i],   eps, j);
            infinitesimalVariation(backwardSolution[i], -eps, j);
            
            fdGradient[j] = (assembler.computeEnergy(forwardSolution) - assembler.computeEnergy(backwardSolution))
                / (2*eps);
            
            forwardSolution[i] = x[i];
            backwardSolution[i] = x[i];
        }
        
        if ( (fdGradient-gradient[i]).two_norm() > 1e-6 ) {
#ifdef ABORT_ON_ERROR
            std::cout << "Differing gradients at " << i 
                      << "!  (" << (fdGradient-gradient[i]).two_norm() / gradient[i].two_norm()
                      << ")    Analytical: " << gradient[i] << ",  fd: " << fdGradient << std::endl;
            //std::cout << "Current configuration " << std::endl;
            for (size_t j=0; j<x.size(); j++)
                std::cout << x[j] << std::endl;
            //abort();
#else
            gradientError++;
            maxError = std::max(maxError, (fdGradient-gradient[i]).two_norm());
#endif
        }
        
    }

#ifndef ABORT_ON_ERROR
    std::cout << gradientError << " errors in the gradient corrected  (max: " << maxError << ")!" << std::endl;
#endif

}


template <class GridType>
void hessianFDCheck(const std::vector<RigidBodyMotion<double,3> >& x, 
                    const Dune::BCRSMatrix<Dune::FieldMatrix<double,6,6> >& hessian, 
                    const RodAssembler<GridType,3>& assembler)
{
#ifndef ABORT_ON_ERROR
    static int hessianError = 0;
#endif
    double eps = 1e-2;

    typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<double,6,6> >::row_type::const_iterator ColumnIterator;

    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////
    std::vector<RigidBodyMotion<double,3> > forwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardSolution = x;

    std::vector<RigidBodyMotion<double,3> > forwardForwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > forwardBackwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardForwardSolution = x;
    std::vector<RigidBodyMotion<double,3> > backwardBackwardSolution = x;
    

    // ///////////////////////////////////////////////////////////////
    //   Loop over all blocks of the outer matrix
    // ///////////////////////////////////////////////////////////////
    for (int i=0; i<hessian.N(); i++) {

        ColumnIterator cIt    = hessian[i].begin();
        ColumnIterator cEndIt = hessian[i].end();

        for (; cIt!=cEndIt; ++cIt) {

            // ////////////////////////////////////////////////////////////////////////////
            //   Compute a finite-difference approximation of this hessian matrix block
            // ////////////////////////////////////////////////////////////////////////////

            Dune::FieldMatrix<double,6,6> fdHessian;

            for (int j=0; j<6; j++) {

                for (int k=0; k<6; k++) {

                    if (i==cIt.index() && j==k) {

                        infinitesimalVariation(forwardSolution[i],   eps, j);
                        infinitesimalVariation(backwardSolution[i], -eps, j);

                        // Second derivative
                        fdHessian[j][k] = (assembler.computeEnergy(forwardSolution) 
                                           + assembler.computeEnergy(backwardSolution) 
                                           - 2*assembler.computeEnergy(x) 
                                           ) / (eps*eps);

                        forwardSolution[i]  = x[i];
                        backwardSolution[i] = x[i];

                    } else {

                        infinitesimalVariation(forwardForwardSolution[i],             eps, j);
                        infinitesimalVariation(forwardForwardSolution[cIt.index()],   eps, k);
                        infinitesimalVariation(forwardBackwardSolution[i],            eps, j);
                        infinitesimalVariation(forwardBackwardSolution[cIt.index()], -eps, k);
                        infinitesimalVariation(backwardForwardSolution[i],           -eps, j);
                        infinitesimalVariation(backwardForwardSolution[cIt.index()],  eps, k);
                        infinitesimalVariation(backwardBackwardSolution[i],          -eps, j);
                        infinitesimalVariation(backwardBackwardSolution[cIt.index()],-eps, k);


                        fdHessian[j][k] = (assembler.computeEnergy(forwardForwardSolution)
                                           + assembler.computeEnergy(backwardBackwardSolution)
                                           - assembler.computeEnergy(forwardBackwardSolution)
                                           - assembler.computeEnergy(backwardForwardSolution)) / (4*eps*eps);

                        forwardForwardSolution[i]             = x[i];
                        forwardForwardSolution[cIt.index()]   = x[cIt.index()];
                        forwardBackwardSolution[i]            = x[i];
                        forwardBackwardSolution[cIt.index()]  = x[cIt.index()];
                        backwardForwardSolution[i]            = x[i];
                        backwardForwardSolution[cIt.index()]  = x[cIt.index()];
                        backwardBackwardSolution[i]           = x[i];
                        backwardBackwardSolution[cIt.index()] = x[cIt.index()];
                        
                    }
                            
                }

            }
                //exit(0);
            // /////////////////////////////////////////////////////////////////////////////
            //   Compare analytical and fd Hessians
            // /////////////////////////////////////////////////////////////////////////////

            Dune::FieldMatrix<double,6,6> diff = fdHessian;
            diff -= *cIt;

            if ( diff.frobenius_norm() > 1e-5 ) {
#ifdef ABORT_ON_ERROR
                std::cout << "Differing hessians at [(" << i << ", "<< cIt.index() << ")]!" << std::endl
                          << "Analytical: " << std::endl << *cIt << std::endl
                          << "fd: " << std::endl << fdHessian << std::endl;
                abort();
#else
                hessianError++;
#endif
            }

        }

    }

#ifndef ABORT_ON_ERROR
    std::cout << hessianError << " errors in the Hessian corrected!" << std::endl;
#endif

}

#endif
