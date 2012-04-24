#include "config.h"

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <dune/fufem/functions/constantfunction.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/cosseratenergystiffness.hh>

#include "multiindex.hh"
#include "valuefactory.hh"

#ifdef COSSERAT_ENERGY_FD_GRADIENT
#warning Comparing two finite-difference approximations
#endif

const double eps = 1e-4;

typedef RigidBodyMotion<double,3> TargetSpace;

using namespace Dune;


/** \brief Computes the diameter of a set */
template <class TargetSpace>
double diameter(const std::vector<TargetSpace>& v)
{
    double d = 0;
    for (size_t i=0; i<v.size(); i++)
        for (size_t j=0; j<v.size(); j++)
            d = std::max(d, Rotation<double,3>::distance(v[i].q,v[j].q));
    return d;
}



// ////////////////////////////////////////////////////////
//   Make a test grid consisting of a single simplex
// ////////////////////////////////////////////////////////

template <class GridType>
std::auto_ptr<GridType> makeSingleSimplexGrid()
{
    static const int domainDim = GridType::dimension;
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

    return std::auto_ptr<GridType>(factory.createGrid());
}


template <int domainDim,class LocalFiniteElement>
Tensor3<double,3,3,3> evaluateDerivativeFD(const LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace>& f,
                                           const Dune::FieldVector<double,domainDim>& local)
{
    Tensor3<double,3,3,3> result(0);

    for (int i=0; i<domainDim; i++) {
        
        Dune::FieldVector<double, domainDim> forward  = local;
        Dune::FieldVector<double, domainDim> backward = local;
        
        forward[i]  += eps;
        backward[i] -= eps;

        TargetSpace forwardValue = f.evaluate(forward);
        TargetSpace backwardValue = f.evaluate(backward);
        
        FieldMatrix<double,3,3> forwardMatrix, backwardMatrix;
        forwardValue.q.matrix(forwardMatrix);
        backwardValue.q.matrix(backwardMatrix);
        
        FieldMatrix<double,3,3> fdDer = (forwardMatrix - backwardMatrix) / (2*eps);
        
        for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
                result[j][k][i] = fdDer[j][k];
        
    }

    return result;

}

template <int domainDim>
void testDerivativeOfRotationMatrix(const std::vector<TargetSpace>& corners)
{
    // Make local fe function to be tested
    PQkLocalFiniteElementCache<double,double,domainDim,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,domainDim,1>::FiniteElementType LocalFiniteElement;

    GeometryType simplex;
    simplex.makeSimplex(domainDim);
    
    LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement, TargetSpace> f(feCache.get(simplex), corners);

    // A quadrature rule as a set of test points
    int quadOrder = 3;
    
    const Dune::QuadratureRule<double, domainDim>& quad 
        = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        const Dune::FieldVector<double,domainDim>& quadPos = quad[pt].position();

        // evaluate actual derivative
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, domainDim> derivative = f.evaluateDerivative(quadPos);

        Tensor3<double,3,3,3> DR;
        CosseratEnergyLocalStiffness<typename UGGrid<domainDim>::LeafGridView,LocalFiniteElement,3>::computeDR(f.evaluate(quadPos),derivative, DR);

        //std::cout << "DR:\n" << DR << std::endl;

        // evaluate fd approximation of derivative
        Tensor3<double,3,3,3> DR_fd = evaluateDerivativeFD(f,quadPos);

        double maxDiff = 0;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<3; k++)
                    maxDiff = std::max(maxDiff, std::abs(DR[i][j][k] - DR_fd[i][j][k]));
        
        if ( maxDiff > 100*eps ) {
            std::cout << className(corners[0]) << ": Analytical gradient does not match fd approximation." << std::endl;
            std::cout << "Analytical:\n " << DR << std::endl;
            std::cout << "FD        :\n " << DR_fd << std::endl;
            assert(false);
        }

    }
}

//////////////////////////////////////////////////////////////////////////////////////
//   Test invariance of the energy functional under rotations
//////////////////////////////////////////////////////////////////////////////////////

template <class GridType>
void testEnergy(const GridType* grid, const std::vector<TargetSpace>& coefficients) {

    PQkLocalFiniteElementCache<double,double,GridType::dimension,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,GridType::dimension,1>::FiniteElementType LocalFiniteElement;
    
    //LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> f(feCache.get(element),corners);

    ParameterTree materialParameters;
    materialParameters["thickness"] = "0.1";
    materialParameters["mu"] = "3.8462e+05";
    materialParameters["lambda"] = "2.7149e+05";
    materialParameters["mu_c"] = "3.8462e+05";
    materialParameters["L_c"] = "0.1";
    materialParameters["q"] = "2.5";
    materialParameters["kappa"] = "0.1";

    ConstantFunction<Dune::FieldVector<double,GridType::dimension>, Dune::FieldVector<double,3> > zeroFunction(Dune::FieldVector<double,3>(0));
    
    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3> assembler(materialParameters,
                                                                                                 NULL,
                                                                                                 &zeroFunction);
    
    // compute reference energy
    double referenceEnergy = assembler.energy(*grid->template leafbegin<0>(), 
                                              feCache.get(grid->template leafbegin<0>()->type()),
                                              coefficients);
    
    // rotate the entire configuration
    std::vector<TargetSpace> rotatedCoefficients(coefficients.size());
    
    std::vector<Rotation<double,3> > testRotations;
    ValueFactory<Rotation<double,3> >::get(testRotations);

    for (size_t i=0; i<testRotations.size(); i++) {

        /////////////////////////////////////////////////////////////////////////
        //  Multiply the given configuration by the test rotation.
        //  The energy should remain unchanged.
        /////////////////////////////////////////////////////////////////////////
        FieldMatrix<double,3,3> matrix;
        testRotations[i].matrix(matrix);
        
        for (size_t j=0; j<coefficients.size(); j++) {
            FieldVector<double,3> tmp;
            matrix.mv(coefficients[j].r, tmp);
            rotatedCoefficients[j].r = tmp;
            
            rotatedCoefficients[j].q = testRotations[i].mult(coefficients[j].q);
        }
        
        double energy = assembler.energy(*grid->template leafbegin<0>(), 
                                         feCache.get(grid->template leafbegin<0>()->type()),
                                         rotatedCoefficients);
        
        assert(std::fabs(energy-referenceEnergy) < 1e-4);
        
        //std::cout << "energy: " << energy << std::endl;

    }

}


template <int domainDim>
void testFrameInvariance()
{
    std::cout << " --- Testing frame invariance of the Cosserat energy, domain dimension: " << domainDim << " ---" << std::endl;

    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    typedef UGGrid<domainDim> GridType;
    const std::auto_ptr<GridType> grid = makeSingleSimplexGrid<GridType>();
    
    // //////////////////////////////////////////////////////////
    //  Test whether the energy is invariant under isometries
    // //////////////////////////////////////////////////////////

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);

    // Set up elements of SE(3)
    std::vector<TargetSpace> coefficients(domainDim+1);

    MultiIndex index(domainDim+1, testPoints.size());
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        testEnergy<GridType>(grid.get(), coefficients);
        
    }
    
}

template <int domainDim, class LocalFiniteElement>
void
evaluateFD_dDR_WRTCoefficient(const LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,RigidBodyMotion<double,3> >& f,
                                             const Dune::FieldVector<double, domainDim>& local,
                                           int coefficient,
                                           Dune::array<Tensor3<double,3,3,4>, 3>& result)
{
    typedef RigidBodyMotion<double,3> TargetSpace;
    typedef double ctype;
    
    double eps = 1e-3;
    
    for (int j=0; j<4; j++) {

        assert(f.type().isSimplex());
        
        int ncoeff = domainDim+1;
        std::vector<TargetSpace> cornersPlus(ncoeff);
        std::vector<TargetSpace> cornersMinus(ncoeff);
        for (int k=0; k<ncoeff; k++)
            cornersMinus[k] = cornersPlus[k] = f.coefficient(k);
        
        typename TargetSpace::CoordinateType aPlus  = f.coefficient(coefficient).globalCoordinates();
        typename TargetSpace::CoordinateType aMinus = f.coefficient(coefficient).globalCoordinates();
        aPlus[j+3]  += eps;
        aMinus[j+3] -= eps;
        cornersPlus[coefficient]  = TargetSpace(aPlus);
        cornersMinus[coefficient] = TargetSpace(aMinus);
        LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> fPlus(f.localFiniteElement_,cornersPlus);
        LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,TargetSpace> fMinus(f.localFiniteElement_,cornersMinus);
                
        TargetSpace qPlus = fPlus.evaluate(local);
        FieldMatrix<double,7,domainDim> qDerPlus = fPlus.evaluateDerivative(local);
        TargetSpace qMinus = fMinus.evaluate(local);
        FieldMatrix<double,7,domainDim> qDerMinus = fMinus.evaluateDerivative(local);
        Tensor3<double,3,3,3> DRPlus,DRMinus;
        
        typedef typename SelectType<domainDim==1,OneDGrid,UGGrid<domainDim> >::Type GridType;
        CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3>::computeDR(qPlus,qDerPlus,DRPlus);
        CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3>::computeDR(qMinus,qDerMinus,DRMinus);
        
        for (int i=0; i<3; i++)
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    result[i][k][l][j] = (DRPlus[i][k][l] - DRMinus[i][k][l]) / (2*eps);
        
    }
    
    // Project onto the tangent space at M(q)
    TargetSpace q = f.evaluate(local);
    Dune::FieldMatrix<double,3,3> Mtmp;
    q.q.matrix(Mtmp);
    OrthogonalMatrix<double,3> M(Mtmp);                   
        
    for (int k=0; k<domainDim; k++)
        for (int v_i=0; v_i<4; v_i++) {
                
            Dune::FieldMatrix<double,3,3> unprojected;
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                    unprojected[i][j] = result[i][j][k][v_i];
                    
            Dune::FieldMatrix<double,3,3> projected = M.projectOntoTangentSpace(unprojected);
                
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                    result[i][j][k][v_i] = projected[i][j];
        }

}



template <int domainDim, class GridType, class LocalFiniteElement>
void testDerivativeOfGradientOfRotationMatrixWRTCoefficient(const LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,RigidBodyMotion<double,3> >& f,
                                                            const FieldVector<double,domainDim>& local,
                                                            unsigned int coeff)
{
    RigidBodyMotion<double,3> value = f.evaluate(local);
    FieldMatrix<double,7,domainDim> derOfValueWRTx = f.evaluateDerivative(local);

    FieldMatrix<double,7,7> derOfValueWRTCoefficient;
    f.evaluateDerivativeOfValueWRTCoefficient(local, coeff, derOfValueWRTCoefficient);

    Tensor3<double,7,7,domainDim> derOfGradientWRTCoefficient;
    f.evaluateDerivativeOfGradientWRTCoefficient(local, coeff, derOfGradientWRTCoefficient);

    Tensor3<double,7,7,domainDim> FDderOfGradientWRTCoefficient;
    f.evaluateFDDerivativeOfGradientWRTCoefficient(local, coeff, FDderOfGradientWRTCoefficient);

    // Evaluate the analytical derivative
    Dune::array<Tensor3<double,3,3,4>, 3> dDR_dv;
    Tensor3<double,3,domainDim,4> dDR3_dv;

    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3>::compute_dDR_dv(value,
                                                                                                       derOfValueWRTx,
                                                                                                       derOfValueWRTCoefficient,
                                                                                                       derOfGradientWRTCoefficient,
                                                                                                       dDR_dv,
                                                                                                       dDR3_dv);

    Dune::array<Tensor3<double,3,3,4>, 3> dDR_dv_FD;

    evaluateFD_dDR_WRTCoefficient<domainDim,LocalFiniteElement>(f,local, coeff, dDR_dv_FD);
    
    // Check whether the two are the same
    double maxError = 0;
    for (size_t j=0; j<3; j++)
        for (size_t k=0; k<3; k++)
            for (size_t l=0; l<3; l++)
                for (size_t m=0; m<4; m++) {
                    double diff = std::fabs(dDR_dv[j][k][l][m] - dDR_dv_FD[j][k][l][m]);
                    maxError = std::max(maxError, diff);
                }

    if (maxError > 1e-3) {
         
        std::cout << "Analytical and FD derivatives differ!"
                  << "    local: " << local
                  << "    coefficient: " << coeff << std::endl;
            
        std::cout << "dDR_dv" << std::endl;
        for (size_t i=0; i<dDR_dv.size(); i++)
            std::cout << dDR_dv[i] << std::endl << std::endl;

        std::cout << "finite differences: dDR_dv" << std::endl;
        std::cout << dDR_dv_FD << std::endl;

        abort();
    }

}


template <int domainDim, class GridType, class LocalFiniteElement>
void testDerivativeOfBendingEnergy(const LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,RigidBodyMotion<double,3> >& f,
                                                            const FieldVector<double,domainDim>& local,
                                                            unsigned int coeff)
{
    ParameterTree materialParameters;
    materialParameters["thickness"] = "1";
    materialParameters["mu"] = "1";
    materialParameters["lambda"] = "1";
    materialParameters["mu_c"] = "1";
    materialParameters["L_c"] = "1";
    materialParameters["q"] = "1";
    materialParameters["kappa"] = "1";

    ConstantFunction<Dune::FieldVector<double,GridType::dimension>, Dune::FieldVector<double,3> > zeroFunction(Dune::FieldVector<double,3>(0));
    
    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3> assembler(materialParameters,
                                                                                                 NULL,
                                                                                                 &zeroFunction);

    
    RigidBodyMotion<double,3> value = f.evaluate(local);
    FieldMatrix<double,7,domainDim> derOfValueWRTx = f.evaluateDerivative(local);

    FieldMatrix<double,7,7> derOfValueWRTCoefficient;
    f.evaluateDerivativeOfValueWRTCoefficient(local, coeff, derOfValueWRTCoefficient);

    Tensor3<double,7,7,domainDim> derOfGradientWRTCoefficient;
    f.evaluateDerivativeOfGradientWRTCoefficient(local, coeff, derOfGradientWRTCoefficient);

    Tensor3<double,7,7,domainDim> FDderOfGradientWRTCoefficient;
    f.evaluateFDDerivativeOfGradientWRTCoefficient(local, coeff, FDderOfGradientWRTCoefficient);

    // Evaluate the analytical derivative
    
    Dune::array<Tensor3<double,3,3,4>, 3> dDR_dv;
    Tensor3<double,3,domainDim,4> dDR3_dv;

    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3>::compute_dDR_dv(value,
                                                                                                       derOfValueWRTx,
                                                                                                       derOfValueWRTCoefficient,
                                                                                                       derOfGradientWRTCoefficient,
                                                                                                       dDR_dv,
                                                                                                       dDR3_dv);
              
    //
    Dune::FieldMatrix<double,3,3> R;
    value.q.matrix(R);

    Tensor3<double,3,3,3> DR;
    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3>::computeDR(value, derOfValueWRTx, DR);

    Tensor3<double,3,3,4> dR_dv;
    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3>::computeDR_dv(value, derOfValueWRTCoefficient, dR_dv);
    
    FieldVector<double,7> embeddedGradient;
    assembler.bendingEnergyGradient(embeddedGradient,
                                    R,
                                    dR_dv,
                                    DR,
                                    dDR3_dv);

    
    // ///////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    // ///////////////////////////////////////////////////////////

    double eps = 1e-4;
    FieldVector<double,4> embeddedFDGradient;

    size_t nDofs = f.localFiniteElement_.localBasis().size();
    std::vector<TargetSpace> forwardSolution(nDofs);
    std::vector<TargetSpace> backwardSolution(nDofs);
    
    for (size_t i=0; i<nDofs; i++)
        forwardSolution[i] = backwardSolution[i] = f.coefficient(i);

    for (int j=0; j<4; j++) {
            
        FieldVector<double,7> forwardCorrection(0);
        forwardCorrection[j+3] += eps;
            
        FieldVector<double,7> backwardCorrection(0);
        backwardCorrection[j+3] -= eps;
            
        forwardSolution[coeff]  = TargetSpace(f.coefficient(coeff).globalCoordinates() + forwardCorrection);
        backwardSolution[coeff] = TargetSpace(f.coefficient(coeff).globalCoordinates() + backwardCorrection);

        LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,RigidBodyMotion<double,3> > forwardF(f.localFiniteElement_,forwardSolution);
        LocalGeodesicFEFunction<domainDim,double,LocalFiniteElement,RigidBodyMotion<double,3> > backwardF(f.localFiniteElement_,backwardSolution);
        
        RigidBodyMotion<double,3> forwardValue = forwardF.evaluate(local);
        RigidBodyMotion<double,3> backwardValue = backwardF.evaluate(local);
        
        FieldMatrix<double,3,3> forwardR, backwardR;
        forwardValue.q.matrix(forwardR);
        backwardValue.q.matrix(backwardR);

        FieldMatrix<double,7,domainDim> forwardDerivative  = forwardF.evaluateDerivative(local);
        FieldMatrix<double,7,domainDim> backwardDerivative = backwardF.evaluateDerivative(local);

        Tensor3<double,3,3,3> forwardDR;
        Tensor3<double,3,3,3> backwardDR;
        
        assembler.computeDR(forwardValue, forwardDerivative, forwardDR);
        assembler.computeDR(backwardValue, backwardDerivative, backwardDR);

        double forwardEnergy = assembler.bendingEnergy(forwardR, forwardDR);
        double backwardEnergy = assembler.bendingEnergy(backwardR, backwardDR);
        
        embeddedFDGradient[j] = (forwardEnergy - backwardEnergy) / (2*eps);

    }

    embeddedFDGradient = f.coefficient(coeff).q.projectOntoTangentSpace(embeddedFDGradient);

    ////////////////////////////////////////////////
    // Check whether the two are the same
    ////////////////////////////////////////////////

    double maxError = 0;
    for (size_t i=0; i<4; i++) {
        double diff = std::fabs(embeddedGradient[i+3] - embeddedFDGradient[i]);
        maxError = std::max(maxError, diff);
    }

    if (maxError > 1e-3) {
         
        std::cout << "Analytical and FD gradients differ!"
                  << "    local: " << local
                  << "    coefficient: " << coeff << std::endl;
            
        std::cout << "embeddedGradient:" << std::endl;
        std::cout << embeddedGradient[3] << " " << embeddedGradient[4] << " " << embeddedGradient[5] << " " << embeddedGradient[6] << std::endl << std::endl;

        std::cout << "embeddedFDGradient:" << std::endl;
        std::cout << embeddedFDGradient << std::endl;

        abort();
    }

}




template <int domainDim>
void testDerivativeOfGradientOfRotationMatrixWRTCoefficient()
{
    std::cout << " --- test derivative of gradient of rotation matrix wrt coefficient ---" << std::endl;

    // we just need the type
    typedef typename SelectType<domainDim==1,OneDGrid,UGGrid<domainDim> >::Type GridType;

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);

    // Set up elements of SE(3)
    std::vector<TargetSpace> coefficients(domainDim+1);

    MultiIndex index(domainDim+1, testPoints.size());
    int numIndices = index.cycle();
    
    for (int i=0; i<numIndices; i++, ++index) {
        
        std::cout << "testing der of grad index: " << i << std::endl;
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        if (diameter(coefficients) > M_PI-0.1) {
            std::cout << "skipping, diameter = " << diameter(coefficients) << std::endl;
            continue;
        }
        
        PQkLocalFiniteElementCache<double,double,GridType::dimension,1> feCache;
        typedef typename PQkLocalFiniteElementCache<double,double,GridType::dimension,1>::FiniteElementType LocalFiniteElement;
    
        GeometryType simplex;
        simplex.makeSimplex(domainDim);
        LocalGeodesicFEFunction<GridType::dimension,double,LocalFiniteElement,RigidBodyMotion<double,3> > f(feCache.get(simplex),coefficients);
    
        // Loop over the coefficients -- we test derivation with respect to each of them
        for (size_t j=0; j<coefficients.size(); j++) {

            // A quadrature rule as a set of test points
            int quadOrder = 3;
    
            const Dune::QuadratureRule<double, domainDim>& quad 
                = Dune::QuadratureRules<double, domainDim>::rule(GeometryType(GeometryType::simplex,domainDim), quadOrder);
    
            for (size_t pt=0; pt<quad.size(); pt++) {
                
                FieldVector<double,domainDim> quadPos = quad[pt].position();
                
                testDerivativeOfGradientOfRotationMatrixWRTCoefficient<domainDim,GridType,LocalFiniteElement>(f, quadPos, j);
                testDerivativeOfBendingEnergy<domainDim,GridType,LocalFiniteElement>(f, quadPos, j);

            }

        }
        
    }

}



template <int domainDim>
void testEnergyGradient()
{
    std::cout << " --- Testing the gradient of the Cosserat energy functional, domain dimension: " << domainDim << " ---" << std::endl;

    // ////////////////////////////////////////////////////////
    //   Make a test grid consisting of a single simplex
    // ////////////////////////////////////////////////////////

    typedef UGGrid<domainDim> GridType;
    const std::auto_ptr<GridType> grid = makeSingleSimplexGrid<GridType>();

    ////////////////////////////////////////////////////////////////////////////
    //  Create a local assembler object
    ////////////////////////////////////////////////////////////////////////////
    
    PQkLocalFiniteElementCache<double,double,GridType::dimension,1> feCache;
    typedef typename PQkLocalFiniteElementCache<double,double,GridType::dimension,1>::FiniteElementType LocalFiniteElement;

    ParameterTree materialParameters;
    materialParameters["thickness"] = "0.1";
    materialParameters["mu"] = "3.8462e+05";
    materialParameters["lambda"] = "2.7149e+05";
    materialParameters["mu_c"] = "3.8462e+05";
    materialParameters["L_c"] = "0.1";
    materialParameters["q"] = "2.5";
    materialParameters["kappa"] = "0.1";

    ConstantFunction<Dune::FieldVector<double,GridType::dimension>, Dune::FieldVector<double,3> > zeroFunction(Dune::FieldVector<double,3>(0));
    
    CosseratEnergyLocalStiffness<typename GridType::LeafGridView,LocalFiniteElement,3> assembler(materialParameters,
                                                                                                 NULL,
                                                                                                 &zeroFunction);

    // //////////////////////////////////////////////////////////////////////////////////////////
    //   Compare the gradient of the energy function with a finite difference approximation
    // //////////////////////////////////////////////////////////////////////////////////////////

    std::vector<TargetSpace> testPoints;
    ValueFactory<TargetSpace>::get(testPoints);

    // Set up elements of SE(3)
    std::vector<TargetSpace> coefficients(domainDim+1);

    MultiIndex index(domainDim+1, testPoints.size());
    int numIndices = index.cycle();
    
    std::vector<typename RigidBodyMotion<double,3>::TangentVector> gradient(coefficients.size());
    std::vector<typename RigidBodyMotion<double,3>::TangentVector> fdGradient(coefficients.size());

    for (int i=0; i<numIndices; i++, ++index) {
        
        std::cout << "testing index: " << i << std::endl;
        
        for (int j=0; j<domainDim+1; j++)
            coefficients[j] = testPoints[index[j]];

        if (diameter(coefficients) > M_PI-0.05) { 
            std::cout << "skipped, diameter: " << diameter(coefficients) << std::endl;
            continue;
        }
        
        // Compute the analytical gradient
        assembler.assembleGradient(*grid->template leafbegin<0>(),
                                   feCache.get(grid->template leafbegin<0>()->type()),
                                   coefficients,
                                   gradient);
        
        // Compute the finite difference gradient
        assembler.LocalGeodesicFEStiffness<typename UGGrid<domainDim>::LeafGridView,
                                           LocalFiniteElement,
                                           RigidBodyMotion<double,3> >::assembleGradient(*grid->template leafbegin<0>(),
                                   feCache.get(grid->template leafbegin<0>()->type()),
                                   coefficients,
                                   fdGradient);
        
        // Check whether the two are the same
        double maxError = 0;
        for (size_t j=0; j<gradient.size(); j++)
            for (size_t k=0; k<gradient[j].size(); k++) {
                double diff = std::abs(gradient[j][k] - fdGradient[j][k]);
                maxError = std::max(maxError, diff);
            }
            
        if (maxError > 1e-3) {
         
            std::cout << "Analytical and FD gradients differ!" << std::endl;
            
            std::cout << "gradient:" << std::endl;
            for (size_t j=0; j<gradient.size(); j++)
                std::cout << gradient[j] << std::endl;

            std::cout << "fd gradient:" << std::endl;
            for (size_t j=0; j<fdGradient.size(); j++)
                std::cout << fdGradient[j] << std::endl;

            abort();
        }
        
    }
    
}


int main(int argc, char** argv) try
{
    const int domainDim = 2;
    std::cout << " --- Testing derivative of rotation matrix, domain dimension: " << domainDim << " ---" << std::endl;
    
    std::vector<Rotation<double,3> > testPoints;
    
    ValueFactory<Rotation<double,3> >::get(testPoints);
    
    int nTestPoints = testPoints.size();

    // Set up elements of SO(3)
    std::vector<TargetSpace> corners(domainDim+1);

    MultiIndex index(domainDim+1, nTestPoints);
    int numIndices = index.cycle();

    for (int i=0; i<numIndices; i++, ++index) {
        
        for (int j=0; j<domainDim+1; j++)
            corners[j].q = testPoints[index[j]];

        testDerivativeOfRotationMatrix<domainDim>(corners);
                
    }
    
    //////////////////////////////////////////////////////////////////////////////////////
    //   Test invariance of the energy functional under rotations
    //////////////////////////////////////////////////////////////////////////////////////
    
    testFrameInvariance<domainDim>();

    //////////////////////////////////////////////////////////////////////////////////////
    //   Test the gradient of the energy functional
    //////////////////////////////////////////////////////////////////////////////////////

    testDerivativeOfGradientOfRotationMatrixWRTCoefficient<1>();
    testDerivativeOfGradientOfRotationMatrixWRTCoefficient<2>();

    testEnergyGradient<2>();
    
} catch (Exception e) {

    std::cout << e << std::endl;

 }
