#include <config.h>

#define SECOND_ORDER

#include <fenv.h>

#include <boost/multiprecision/mpfr.hpp>

//typedef double FDType;
typedef boost::multiprecision::mpfr_float_50 FDType;

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/gfe/adolcnamespaceinjections.hh>
#include <dune/common/fmatrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/io.hh>

#include <dune/fufem/functionspacebases/p2nodalbasis.hh>


#include <dune/gfe/rotation.hh>
#include <dune/gfe/localgeodesicfestiffness.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/rotation.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
typedef Rotation<double,3> TargetSpace;

using namespace Dune;



template<class GridView, class LocalFiniteElement, int dim, class field_type=double>
class CosseratEnergyLocalStiffness
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,Rotation<field_type,dim> >
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef Rotation<field_type,dim> TargetSpace;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& element,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const
    {
      assert(element.type() == localFiniteElement.type());
      typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

      RT energy = 0;

      typedef LocalGeodesicFEFunction<gridDim, DT, LocalFiniteElement, TargetSpace> LocalGFEFunctionType;
      LocalGFEFunctionType localGeodesicFEFunction(localFiniteElement,localSolution);

      int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                   : localFiniteElement.localBasis().order() * gridDim;

      const Dune::QuadratureRule<DT, gridDim>& quad
          = Dune::QuadratureRules<DT, gridDim>::rule(element.type(), quadOrder);

      for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<DT,gridDim>& quadPos = quad[pt].position();

        const DT integrationElement = element.geometry().integrationElement(quadPos);

        const typename Geometry::JacobianInverseTransposed& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        DT weight = quad[pt].weight() * integrationElement;

        // The value of the local function
        Rotation<field_type,dim> value = localGeodesicFEFunction.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        typename LocalGFEFunctionType::DerivativeType referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos,value);

        // The derivative of the function defined on the actual element
        typename LocalGFEFunctionType::DerivativeType derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

        //////////////////////////////////////////////////////////
        //  Compute the derivative of the rotation
        //  Note: we need it in matrix coordinates
        //////////////////////////////////////////////////////////

        Dune::FieldMatrix<field_type,dim,dim> R;
        value.matrix(R);

        // Add the local energy density
        energy += 2.5e3*weight *derivative.frobenius_norm2();

      }

      return energy;
    }

};

/** \brief Assembles energy gradient and Hessian with ADOL-C
 */
template<class GridView, class LocalFiniteElement>
class LocalGeodesicFEADOLCStiffness
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

    LocalGeodesicFEADOLCStiffness(const LocalGeodesicFEStiffness<GridView, LocalFiniteElement, ATargetSpace>* energy)
    : localEnergy_(energy)
    {}

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const;

    /** \brief Assemble the local stiffness matrix at the current position

    This uses the automatic differentiation toolbox ADOL_C.
    */
    virtual void assembleGradientAndHessian(const Entity& e,
                         const LocalFiniteElement& localFiniteElement,
                         const std::vector<TargetSpace>& localSolution,
                         std::vector<Dune::FieldVector<double, 4> >& localGradient,
                         Dune::Matrix<Dune::FieldMatrix<RT,embeddedBlocksize,embeddedBlocksize> >& localHessian,
                         bool vectorMode);

    const LocalGeodesicFEStiffness<GridView, LocalFiniteElement, ATargetSpace>* localEnergy_;

};


template <class GridView, class LocalFiniteElement>
typename LocalGeodesicFEADOLCStiffness<GridView, LocalFiniteElement>::RT
LocalGeodesicFEADOLCStiffness<GridView, LocalFiniteElement>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
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

    energy = localEnergy_->energy(element,localFiniteElement,localASolution);

    energy >>= pureEnergy;

    trace_off();
    return pureEnergy;
}



// ///////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
// ///////////////////////////////////////////////////////////
template <class GridType, class LocalFiniteElement>
void LocalGeodesicFEADOLCStiffness<GridType, LocalFiniteElement>::
assembleGradientAndHessian(const Entity& element,
                const LocalFiniteElement& localFiniteElement,
                const std::vector<TargetSpace>& localSolution,
                std::vector<Dune::FieldVector<double,4> >& localGradient,
                Dune::Matrix<Dune::FieldMatrix<RT,embeddedBlocksize,embeddedBlocksize> >& localHessian,
                bool vectorMode)
{
    // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
    energy(element, localFiniteElement, localSolution);

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
template<class GridView, class LocalFiniteElement, class field_type=double>
class LocalGeodesicFEFDStiffness
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    typedef typename TargetSpace::template rebind<field_type>::other ATargetSpace;


public:

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

    LocalGeodesicFEFDStiffness(const LocalGeodesicFEStiffness<GridView, LocalFiniteElement, ATargetSpace>* energy)
    : localEnergy_(energy)
    {}

    virtual void assembleGradientAndHessian(const Entity& e,
                                 const LocalFiniteElement& localFiniteElement,
                                 const std::vector<TargetSpace>& localSolution,
                                 std::vector<Dune::FieldVector<double,4> >& localGradient,
                                 Dune::Matrix<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> >& localHessian);

    const LocalGeodesicFEStiffness<GridView, LocalFiniteElement, ATargetSpace>* localEnergy_;
};

// ///////////////////////////////////////////////////////////
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class GridType, class LocalFiniteElement, class field_type>
void LocalGeodesicFEFDStiffness<GridType, LocalFiniteElement, field_type>::
assembleGradientAndHessian(const Entity& element,
                const LocalFiniteElement& localFiniteElement,
                const std::vector<TargetSpace>& localSolution,
                std::vector<Dune::FieldVector<double, 4> >& localGradient,
                Dune::Matrix<Dune::FieldMatrix<double,embeddedBlocksize,embeddedBlocksize> >& localHessian)
{
    // Number of degrees of freedom for this element
    size_t nDofs = localSolution.size();

    // Clear assemble data
    localHessian.setSize(nDofs, nDofs);
    localHessian = 0;

    const field_type eps = 1e-4;

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
    field_type centerValue   = -localEnergy_->energy(element, localFiniteElement, localASolution);

    // Precompute energy infinitesimal corrections in the directions of the local basis vectors
    std::vector<Dune::array<field_type,embeddedBlocksize> > forwardEnergy(nDofs);
    std::vector<Dune::array<field_type,embeddedBlocksize> > backwardEnergy(nDofs);

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

            forwardEnergy[i][i2]  = localEnergy_->energy(element, localFiniteElement, forwardSolution);
            backwardEnergy[i][i2] = localEnergy_->energy(element, localFiniteElement, backwardSolution);

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
            localGradient[i][j] = foo.template convert_to<double>();
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

                    field_type forwardValue  = localEnergy_->energy(element, localFiniteElement, forwardSolutionXiEta) - forwardEnergy[i][i2] - forwardEnergy[j][j2];
                    field_type backwardValue = localEnergy_->energy(element, localFiniteElement, backwardSolutionXiEta) - backwardEnergy[i][i2] - backwardEnergy[j][j2];

                    field_type foo = 0.5 * (forwardValue - 2*centerValue + backwardValue) / (eps*eps);
                    localHessian[i][j][i2][j2] = localHessian[j][i][j2][i2] = foo.template convert_to<double>();

                }
            }
        }
    }
}


// Compare two matrices
void compareMatrices(const Matrix<FieldMatrix<double,4,4> >& matrixA, std::string nameA,
                     const Matrix<FieldMatrix<double,4,4> >& matrixB, std::string nameB)
{
  double maxAbsDifference = -1;
  double maxRelDifference = -1;

  for(int i=0; i<matrixA.N(); i++) {

    for (int j=0; j<matrixA.M(); j++ ) {

      for (int ii=0; ii<4; ii++)
        for (int jj=0; jj<4; jj++)
        {
          double valueA = matrixA[i][j][ii][jj];
          double valueB = matrixB[i][j][ii][jj];

          double absDifference = valueA - valueB;
          double relDifference = std::abs(absDifference) / std::abs(valueA);
          maxAbsDifference = std::max(maxAbsDifference, std::abs(absDifference));
          if (not isinf(relDifference))
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

    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef YaspGrid<dim> GridType;

    FieldVector<double,dim> upper = {{0.38, 0.128}};

    array<int,dim> elements = {{15, 5}};
    GridType grid(upper, elements);

    typedef GridType::LeafGridView GridView;
    GridView gridView = grid.leafGridView();

    typedef P2NodalBasis<GridView,double> FEBasis;
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
    Dune::BlockVector<FieldVector<double,7> > xEmbedded(x.size());

    std::ifstream file("dangerous_iterate", std::ios::in|std::ios::binary);
    if (not(file))
      DUNE_THROW(SolverError, "Couldn't open file 'dangerous_iterate' for reading");

    GenericVector::readBinary(file, xEmbedded);

    file.close();

    for (int ii=0; ii<x.size(); ii++)
    {
      // The first 3 of the 7 entries are irrelevant
      FieldVector<double, 4> rotationEmbedded;
      for (int jj=0; jj<4; jj++)
        rotationEmbedded[jj] = xEmbedded[ii][jj+3];

      x[ii] = TargetSpace(rotationEmbedded);
    }

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    // Assembler using ADOL-C
    CosseratEnergyLocalStiffness<GridView,
                                 FEBasis::LocalFiniteElement,
                                 3,adouble> cosseratEnergyADOLCLocalStiffness;

    LocalGeodesicFEADOLCStiffness<GridView,
                                  FEBasis::LocalFiniteElement> localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    CosseratEnergyLocalStiffness<GridView,
                                 FEBasis::LocalFiniteElement,
                                 3,FDType> cosseratEnergyFDLocalStiffness;

    LocalGeodesicFEFDStiffness<GridView,
                             FEBasis::LocalFiniteElement,FDType> localGFEFDStiffness(&cosseratEnergyFDLocalStiffness);

    // Compute and compare matrices
    auto it    = gridView.template begin<0>();
    auto endit = gridView.template end<0>  ();

    for( ; it != endit; ++it ) {

        std::cout << "  ++++  element " << gridView.indexSet().index(*it) << " ++++" << std::endl;

        const int numOfBaseFct = feBasis.getLocalFiniteElement(*it).localBasis().size();

        // Extract local solution
        std::vector<TargetSpace> localSolution(numOfBaseFct);

        for (int i=0; i<numOfBaseFct; i++)
            localSolution[i] = x[feBasis.index(*it,i)];

        std::vector<Dune::FieldVector<double,4> > localADGradient(numOfBaseFct);
        std::vector<Dune::FieldVector<double,4> > localADVMGradient(numOfBaseFct);  // VM: vector-mode
        std::vector<Dune::FieldVector<double,4> > localFDGradient(numOfBaseFct);

        Matrix<FieldMatrix<double,4,4> > localADHessian;
        Matrix<FieldMatrix<double,4,4> > localADVMHessian;   // VM: vector-mode
        Matrix<FieldMatrix<double,4,4> > localFDHessian;

        // setup local matrix and gradient
        localGFEADOLCStiffness.assembleGradientAndHessian(*it,
                                                          feBasis.getLocalFiniteElement(*it),
                                                          localSolution,
                                                          localADGradient,
                                                          localADHessian,
                                                          false);   // 'true' means 'vector mode'

        localGFEADOLCStiffness.assembleGradientAndHessian(*it,
                                                          feBasis.getLocalFiniteElement(*it),
                                                          localSolution,
                                                          localADGradient,
                                                          localADVMHessian,
                                                          true);   // 'true' means 'vector mode'

        localGFEFDStiffness.assembleGradientAndHessian(*it, feBasis.getLocalFiniteElement(*it), localSolution, localFDGradient, localFDHessian);

        compareMatrices(localADHessian, "AD", localFDHessian, "FD");
        compareMatrices(localADHessian, "AD scalar", localADVMHessian, "AD vector");
    }

    // //////////////////////////////
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
