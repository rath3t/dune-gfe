#include <config.h>

#include <fenv.h>

#include <array>

//#define MULTIPRECISION

#ifdef MULTIPRECISION
#include <boost/multiprecision/mpfr.hpp>
#endif

#ifdef MULTIPRECISION
typedef boost::multiprecision::mpfr_float_50 FDType;
#else
typedef double FDType;
#endif

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/fmatrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>


#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/localgeodesicfestiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/localgeodesicfefdstiffness.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
typedef RigidBodyMotion<double,3> TargetSpace;

using namespace Dune;

/** \brief Assembles energy gradient and Hessian with ADOL-C
 */
template<class Basis>
class LocalADOLCStiffness
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

    LocalADOLCStiffness(const LocalGeodesicFEStiffness<Basis, ATargetSpace>* energy)
    : localEnergy_(energy)
    {}

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const typename Basis::LocalView& localView,
               const std::vector<TargetSpace>& localSolution) const;

    /** \brief Assemble the local stiffness matrix at the current position

    This uses the automatic differentiation toolbox ADOL_C.
    */
    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                         const std::vector<TargetSpace>& localSolution,
                         std::vector<Dune::FieldVector<double, embeddedBlocksize> >& localGradient,
                         Dune::Matrix<Dune::FieldMatrix<RT,embeddedBlocksize,embeddedBlocksize> >& localHessian,
                         bool vectorMode);

    const LocalGeodesicFEStiffness<Basis, ATargetSpace>* localEnergy_;

};


template <class Basis>
typename LocalADOLCStiffness<Basis>::RT
LocalADOLCStiffness<Basis>::
energy(const typename Basis::LocalView& localView,
       const std::vector<TargetSpace>& localSolution) const
{
    double pureEnergy;

    std::vector<ATargetSpace> localASolution(localSolution.size());

    trace_on(1);

    adouble energy = 0;

    // The following loop is not quite intuitive: we copy the localSolution into an
    // array of FieldVector<double>, go from there to FieldVector<adouble> and
    // only then to ATargetSpace.
    // Rationale: The constructor/assignment-from-vector of TargetSpace frequently
    // contains a projection onto the manifold from the surrounding Euclidean space.
    // ADOL-C needs a function on the whole Euclidean space, hence that projection
    // is part of the function and needs to be taped.

    // The following variable cannot be declared inside of the loop, or ADOL-C will report wrong results
    // (Presumably because several independent variables use the same memory location.)
    std::vector<typename ATargetSpace::CoordinateType> aRaw(localSolution.size());
    for (size_t i=0; i<localSolution.size(); i++) {
      typename TargetSpace::CoordinateType raw = localSolution[i].globalCoordinates();
      for (size_t j=0; j<raw.size(); j++)
        aRaw[i][j] <<= raw[j];
      localASolution[i] = aRaw[i];  // may contain a projection onto M -- needs to be done in adouble
    }

    energy = localEnergy_->energy(localView,localASolution);

    energy >>= pureEnergy;

    trace_off();
    return pureEnergy;
}



// ///////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
// ///////////////////////////////////////////////////////////
template <class Basis>
void LocalADOLCStiffness<Basis>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                const std::vector<TargetSpace>& localSolution,
                std::vector<Dune::FieldVector<double,embeddedBlocksize> >& localGradient,
                Dune::Matrix<Dune::FieldMatrix<RT,embeddedBlocksize,embeddedBlocksize> >& localHessian,
                bool vectorMode)
{
    // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
    energy(localView, localSolution);

    /////////////////////////////////////////////////////////////////
    // Compute the gradient.
    /////////////////////////////////////////////////////////////////

    // Copy data from Dune data structures to plain-C ones
    size_t nDofs = localSolution.size();
    size_t nDoubles = nDofs*embeddedBlocksize;
    std::vector<double> xp(nDoubles);
    int idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<embeddedBlocksize; j++)
            xp[idx++] = localSolution[i].globalCoordinates()[j];

  // Compute gradient
    std::vector<double> g(nDoubles);
    gradient(1,nDoubles,xp.data(),g.data());                  // gradient evaluation

    // Copy into Dune type
    std::vector<typename TargetSpace::EmbeddedTangentVector> localEmbeddedGradient(localSolution.size());

    idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<embeddedBlocksize; j++)
            localGradient[i][j] = g[idx++];

    /////////////////////////////////////////////////////////////////
    // Compute Hessian
    /////////////////////////////////////////////////////////////////

    localHessian.setSize(nDofs,nDofs);

    double* rawHessian[nDoubles];
    for(size_t i=0; i<nDoubles; i++)
        rawHessian[i] = (double*)malloc((i+1)*sizeof(double));

    if (vectorMode)
      hessian2(1,nDoubles,xp.data(),rawHessian);
    else
      hessian(1,nDoubles,xp.data(),rawHessian);

    // Copy Hessian into Dune data type
    for(size_t i=0; i<nDoubles; i++)
      for (size_t j=0; j<nDoubles; j++)
      {
        double value = (i>=j) ? rawHessian[i][j] : rawHessian[j][i];
        localHessian[j/embeddedBlocksize][i/embeddedBlocksize][j%embeddedBlocksize][i%embeddedBlocksize] = value;
      }

    for(size_t i=0; i<nDoubles; i++)
        free(rawHessian[i]);

}

/** \brief Assembles energy gradient and Hessian with finite differences
 */
template<class Basis, class field_type=double>
class LocalFDStiffness
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    typedef typename TargetSpace::template rebind<field_type>::other ATargetSpace;


public:

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

    LocalFDStiffness(const LocalGeodesicFEStiffness<Basis, ATargetSpace>* energy)
    : localEnergy_(energy)
    {}

    virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                 const std::vector<TargetSpace>& localSolution,
                                 std::vector<Dune::FieldVector<double,embeddedBlocksize> >& localGradient,
                                 Dune::Matrix<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> >& localHessian);

    const LocalGeodesicFEStiffness<Basis, ATargetSpace>* localEnergy_;
};

// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class Basis, class field_type>
void LocalFDStiffness<Basis, field_type>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                const std::vector<TargetSpace>& localSolution,
                std::vector<Dune::FieldVector<double, embeddedBlocksize> >& localGradient,
                Dune::Matrix<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> >& localHessian)
{
    // Number of degrees of freedom for this element
    size_t nDofs = localSolution.size();

    // Clear assemble data
    localHessian.setSize(nDofs, nDofs);
    localHessian = 0;

#ifdef MULTIPRECISION
    const field_type eps = 1e-10;
#else
    const field_type eps = 1e-4;
#endif

    std::vector<ATargetSpace> localASolution(localSolution.size());
    std::vector<typename ATargetSpace::CoordinateType> aRaw(localSolution.size());
    for (size_t i=0; i<localSolution.size(); i++) {
      typename TargetSpace::CoordinateType raw = localSolution[i].globalCoordinates();
      for (size_t j=0; j<raw.size(); j++)
          aRaw[i][j] = raw[j];
      localASolution[i] = aRaw[i];  // may contain a projection onto M -- needs to be done in adouble
    }

    std::vector<Dune::FieldMatrix<field_type,embeddedBlocksize,embeddedBlocksize> > B(localSolution.size());
    for (size_t i=0; i<B.size(); i++)
    {
        B[i] = 0;
        for (int j=0; j<embeddedBlocksize; j++)
          B[i][j][j] = 1.0;
    }

    // Precompute negative energy at the current configuration
    // (negative because that is how we need it as part of the 2nd-order fd formula)
    field_type centerValue   = -localEnergy_->energy(localView, localASolution);

    // Precompute energy infinitesimal corrections in the directions of the local basis vectors
    std::vector<std::array<field_type,embeddedBlocksize> > forwardEnergy(nDofs);
    std::vector<std::array<field_type,embeddedBlocksize> > backwardEnergy(nDofs);

    for (size_t i=0; i<localSolution.size(); i++) {
        for (size_t i2=0; i2<embeddedBlocksize; i2++) {
            typename ATargetSpace::EmbeddedTangentVector epsXi = B[i][i2];
            epsXi *= eps;
            typename ATargetSpace::EmbeddedTangentVector minusEpsXi = epsXi;
            minusEpsXi  *= -1;

            std::vector<ATargetSpace> forwardSolution  = localASolution;
            std::vector<ATargetSpace> backwardSolution = localASolution;

            forwardSolution[i]  = ATargetSpace(localASolution[i].globalCoordinates() + epsXi);
            backwardSolution[i] = ATargetSpace(localASolution[i].globalCoordinates() + minusEpsXi);

            forwardEnergy[i][i2]  = localEnergy_->energy(localView, forwardSolution);
            backwardEnergy[i][i2] = localEnergy_->energy(localView, backwardSolution);

        }

    }

    //////////////////////////////////////////////////////////////
    //   Compute gradient by finite-difference approximation
    //////////////////////////////////////////////////////////////

    localGradient.resize(localSolution.size());

    for (size_t i=0; i<localSolution.size(); i++)
        for (int j=0; j<embeddedBlocksize; j++)
        {
          field_type foo = (forwardEnergy[i][j] - backwardEnergy[i][j]) / (2*eps);
#ifdef MULTIPRECISION
            localGradient[i][j] = foo.template convert_to<double>();
#else
            localGradient[i][j] = foo;
#endif
        }

    ///////////////////////////////////////////////////////////////////////////
    //   Compute Riemannian Hesse matrix by finite-difference approximation.
    //   We loop over the lower left triangular half of the matrix.
    //   The other half follows from symmetry.
    ///////////////////////////////////////////////////////////////////////////
    //#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<localSolution.size(); i++) {
        for (size_t i2=0; i2<embeddedBlocksize; i2++) {
            for (size_t j=0; j<=i; j++) {
                for (size_t j2=0; j2<((i==j) ? i2+1 : embeddedBlocksize); j2++) {

                    std::vector<ATargetSpace> forwardSolutionXiEta  = localASolution;
                    std::vector<ATargetSpace> backwardSolutionXiEta  = localASolution;

                    typename ATargetSpace::EmbeddedTangentVector epsXi  = B[i][i2];    epsXi *= eps;
                    typename ATargetSpace::EmbeddedTangentVector epsEta = B[j][j2];   epsEta *= eps;

                    typename ATargetSpace::EmbeddedTangentVector minusEpsXi  = epsXi;   minusEpsXi  *= -1;
                    typename ATargetSpace::EmbeddedTangentVector minusEpsEta = epsEta;  minusEpsEta *= -1;

                    if (i==j)
                        forwardSolutionXiEta[i] = ATargetSpace(localASolution[i].globalCoordinates() + epsXi+epsEta);
                    else {
                        forwardSolutionXiEta[i] = ATargetSpace(localASolution[i].globalCoordinates() + epsXi);
                        forwardSolutionXiEta[j] = ATargetSpace(localASolution[j].globalCoordinates() + epsEta);
                    }

                    if (i==j)
                        backwardSolutionXiEta[i] = ATargetSpace(localASolution[i].globalCoordinates() + minusEpsXi+minusEpsEta);
                    else {
                        backwardSolutionXiEta[i] = ATargetSpace(localASolution[i].globalCoordinates() + minusEpsXi);
                        backwardSolutionXiEta[j] = ATargetSpace(localASolution[j].globalCoordinates() + minusEpsEta);
                    }

                    field_type forwardValue  = localEnergy_->energy(localView, forwardSolutionXiEta) - forwardEnergy[i][i2] - forwardEnergy[j][j2];
                    field_type backwardValue = localEnergy_->energy(localView, backwardSolutionXiEta) - backwardEnergy[i][i2] - backwardEnergy[j][j2];

                    field_type foo = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
#ifdef MULTIPRECISION
                    localHessian[i][j][i2][j2] = localHessian[j][i][j2][i2] = foo.template convert_to<double>();
#else
                    localHessian[i][j][i2][j2] = localHessian[j][i][j2][i2] = foo;
#endif
                }
            }
        }
    }
}


// Compare two matrices
template <int N>
void compareMatrices(const Matrix<FieldMatrix<double,N,N> >& matrixA, std::string nameA,
                     const Matrix<FieldMatrix<double,N,N> >& matrixB, std::string nameB)
{
  double maxAbsDifference = -1;
  double maxRelDifference = -1;

  for(size_t i=0; i<matrixA.N(); i++) {

    for (size_t j=0; j<matrixA.M(); j++ ) {

      for (size_t ii=0; ii<matrixA[i][j].N(); ii++)
        for (size_t jj=0; jj<matrixA[i][j].M(); jj++)
        {
          double valueA = matrixA[i][j][ii][jj];
          double valueB = matrixB[i][j][ii][jj];

          double absDifference = valueA - valueB;
          double relDifference = std::abs(absDifference) / std::abs(valueA);
          maxAbsDifference = std::max(maxAbsDifference, std::abs(absDifference));
          if (not std::isinf(relDifference))
            maxRelDifference = std::max(maxRelDifference, relDifference);

          if (relDifference > 1)
            std::cout << i << ", " << j << "   " << ii << ", " << jj
            << ",       " << nameA << ": " << valueA << ",           " << nameB << ": " << valueB << std::endl;
        }
    }
  }

  std::cout << nameA << " vs. " << nameB << " -- max absolute / relative difference is " << maxAbsDifference << " / " << maxRelDifference << std::endl;
}


int main (int argc, char *argv[]) try
{
    typedef std::vector<TargetSpace> SolutionType;
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };
    enum { blocksize = TargetSpace::TangentVector::dimension };

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef YaspGrid<dim> GridType;

    FieldVector<double,dim> upper = {{0.38, 0.128}};

    std::array<int,dim> elements = {{5, 5}};
    GridType grid(upper, elements);

    typedef GridType::LeafGridView GridView;
    GridView gridView = grid.leafGridView();

    typedef Functions::LagrangeBasis<GridView,1> FEBasis;
    FEBasis feBasis(gridView);

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    SolutionType x(feBasis.size());

    //////////////////////////////////////////7
    //  Read initial iterate from file
    //////////////////////////////////////////7
#if 0
    Dune::BlockVector<FieldVector<double,7> > xEmbedded(x.size());

    std::ifstream file("dangerous_iterate", std::ios::in|std::ios::binary);
    if (not(file))
      DUNE_THROW(SolverError, "Couldn't open file 'dangerous_iterate' for reading");

    GenericVector::readBinary(file, xEmbedded);

    file.close();

    for (int ii=0; ii<x.size(); ii++)
      x[ii] = xEmbedded[ii];
#else
    auto identity = [](const FieldVector<double,2>& x) -> FieldVector<double,3> { return {x[0], x[1], 0};};

    std::vector<FieldVector<double,3> > v;
    Functions::interpolate(feBasis, v, identity);

    for (size_t i=0; i<x.size(); i++)
      x[i].r = v[i];
#endif

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    ParameterTree materialParameters;
    materialParameters["thickness"] = "1";
    materialParameters["mu"] = "1";
    materialParameters["lambda"] = "1";
    materialParameters["mu_c"] = "1";
    materialParameters["L_c"] = "1";
    materialParameters["q"] = "2";
    materialParameters["kappa"] = "1";

    ///////////////////////////////////////////////////////////////////////
    //  Assemblers for the Euclidean derivatives in an embedding space
    ///////////////////////////////////////////////////////////////////////

    // Assembler using ADOL-C
    CosseratEnergyLocalStiffness<FEBasis,
                                 3,adouble> cosseratEnergyADOLCLocalStiffness(materialParameters, nullptr, nullptr, nullptr);

    LocalADOLCStiffness<FEBasis> localADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    CosseratEnergyLocalStiffness<FEBasis,
                                 3,FDType> cosseratEnergyFDLocalStiffness(materialParameters, nullptr, nullptr, nullptr);

    LocalFDStiffness<FEBasis,FDType> localFDStiffness(&cosseratEnergyFDLocalStiffness);

    ///////////////////////////////////////////////////////////////////////
    //  Assemblers for the Riemannian derivatives without embedding space
    ///////////////////////////////////////////////////////////////////////

    // Assembler using ADOL-C
    LocalGeodesicFEADOLCStiffness<FEBasis,
                                  TargetSpace> localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    LocalGeodesicFEFDStiffness<FEBasis,
                               TargetSpace,
                               FDType> localGFEFDStiffness(&cosseratEnergyFDLocalStiffness);

    // Compute and compare matrices
    for (const auto& element : Dune::elements(gridView))
    {
        std::cout << "  ++++  element " << gridView.indexSet().index(element) << " ++++" << std::endl;

        auto localView     = feBasis.localView();
        localView.bind(element);
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
        auto localIndexSet = feBasis.localIndexSet();
        localIndexSet.bind(localView);
#endif

        const int numOfBaseFct = localView.size();

        // Extract local configuration
        std::vector<TargetSpace> localSolution(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
#if DUNE_VERSION_LT(DUNE_FUNCTIONS,2,7)
            localSolution[i] = x[localIndexSet.index(i)];
#else
            localSolution[i] = x[localView.index(i)];
#endif

        std::vector<Dune::FieldVector<double,embeddedBlocksize> > localADGradient(numOfBaseFct);
        std::vector<Dune::FieldVector<double,embeddedBlocksize> > localADVMGradient(numOfBaseFct);  // VM: vector-mode
        std::vector<Dune::FieldVector<double,embeddedBlocksize> > localFDGradient(numOfBaseFct);

        Matrix<FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> > localADHessian;
        Matrix<FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> > localADVMHessian;   // VM: vector-mode
        Matrix<FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> > localFDHessian;

        // Assemble Euclidean derivatives
        localADOLCStiffness.assembleGradientAndHessian(localView,
                                                          localSolution,
                                                          localADGradient,
                                                          localADHessian,
                                                          false);   // 'true' means 'vector mode'

        localADOLCStiffness.assembleGradientAndHessian(localView,
                                                          localSolution,
                                                          localADGradient,
                                                          localADVMHessian,
                                                          true);   // 'true' means 'vector mode'

        localFDStiffness.assembleGradientAndHessian(localView,
                                                    localSolution,
                                                    localFDGradient,
                                                    localFDHessian);

        // compare
        compareMatrices(localADHessian, "AD", localFDHessian, "FD");
        compareMatrices(localADHessian, "AD scalar", localADVMHessian, "AD vector");

        // Assemble Riemannian derivatives
        std::vector<Dune::FieldVector<double,blocksize> > localRiemannianADGradient(numOfBaseFct);
        std::vector<Dune::FieldVector<double,blocksize> > localRiemannianFDGradient(numOfBaseFct);

        Matrix<FieldMatrix<double,blocksize,blocksize> > localRiemannianADHessian;
        Matrix<FieldMatrix<double,blocksize,blocksize> > localRiemannianFDHessian;

        localGFEADOLCStiffness.assembleGradientAndHessian(localView,
                                                          localSolution,
                                                          localRiemannianADGradient);

        localGFEFDStiffness.assembleGradientAndHessian(localView,
                                                       localSolution,
                                                       localRiemannianFDGradient);

        // compare
        compareMatrices(localGFEADOLCStiffness.A_, "Riemannian AD", localGFEFDStiffness.A_, "Riemannian FD");

    }

    // //////////////////////////////
 } catch (Exception& e) {

    std::cout << e << std::endl;

 }
