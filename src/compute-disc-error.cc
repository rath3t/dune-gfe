#include <config.h>

#include <array>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#if HAVE_DUNE_FOAMGRID
#include <dune/foamgrid/foamgrid.hh>
#else
#include <dune/grid/onedgrid.hh>
#endif

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include <dune/matrix-vector/genericvectortools.hh>

#include <dune/fufem/discretizationerror.hh>
#include <dune/fufem/dunepython.hh>
#include <dune/fufem/makesphere.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/localprojectedfefunction.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>

// grid dimension
const int dim = 2;
const int dimworld = 2;

using namespace Dune;

/** \brief Check whether given local coordinates are contained in the reference element
    \todo This method exists in the Dune grid interface!  But we need the eps.
*/
static bool checkInside(const Dune::GeometryType& type,
                        const Dune::FieldVector<double, dim> &loc,
                        double eps)
{
  switch (type.dim())
  {
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
      } else
        DUNE_THROW(Dune::GridError, "checkInside():  ERROR:  Unknown type " << type << " found!");

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
      }else
        DUNE_THROW(Dune::GridError, "checkInside():  ERROR:  Unknown type " << type << " found!");

    default:
      DUNE_THROW(Dune::GridError, "checkInside():  ERROR:  Unknown type " << type << " found!");
  }

}

/** \param element An element of the source grid
 */
template <class GridType>
auto findSupportingElement(const GridType& sourceGrid,
                           const GridType& targetGrid,
                           typename GridType::template Codim<0>::Entity element,
                           FieldVector<double,dim> pos)
{
  while (element.level() != 0)
  {
    pos = element.geometryInFather().global(pos);
    element = element.father();

    assert(checkInside(element.type(), pos, 1e-7));
  }

  //////////////////////////////////////////////////////////////////////
  //   Find the corresponding coarse grid element on the adaptive grid.
  //   This is a linear algorithm, but we expect the coarse grid to be small.
  //////////////////////////////////////////////////////////////////////
  LevelMultipleCodimMultipleGeomTypeMapper<GridType> sourceP0Mapper (sourceGrid, 0, mcmgElementLayout());
  LevelMultipleCodimMultipleGeomTypeMapper<GridType> targetP0Mapper(targetGrid, 0, mcmgElementLayout());
  const auto coarseIndex = sourceP0Mapper.index(element);

  auto targetLevelView = targetGrid.levelGridView(0);

  for (auto&& targetElement : elements(targetLevelView))
    if (targetP0Mapper.index(targetElement) == coarseIndex)
    {
      element = std::move(targetElement);
      break;
    }

  //////////////////////////////////////////////////////////////////////////
  //   Find a corresponding point (not necessarily vertex) on the leaf level
  //   of the adaptive grid.
  //////////////////////////////////////////////////////////////////////////

  while (!element.isLeaf())
  {
    FieldVector<double,dim> childPos;
    assert(checkInside(element.type(), pos, 1e-7));

    auto hIt    = element.hbegin(element.level()+1);
    auto hEndIt = element.hend(element.level()+1);

    for (; hIt!=hEndIt; ++hIt)
    {
      childPos = hIt->geometryInFather().local(pos);
      if (checkInside(hIt->type(), childPos, 1e-7))
        break;
    }

    element = *hIt;
    pos     = childPos;
  }

  return std::tie(element, pos);
}

template <class GridView, int order, class TargetSpace>
void measureDiscreteEOC(const GridView gridView,
                        const GridView referenceGridView,
                        const ParameterTree& parameterSet)
{
  typedef std::vector<TargetSpace> SolutionType;

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct the scalar function space bases corresponding to the GFE space
  //////////////////////////////////////////////////////////////////////////////////

  typedef Dune::Functions::LagrangeBasis<GridView, order> FEBasis;
  FEBasis feBasis(gridView);
  FEBasis referenceFEBasis(referenceGridView);

  //typedef LocalGeodesicFEFunction<GridView::dimension, double, typename FEBasis::LocalView::Tree::FiniteElement, TargetSpace> LocalInterpolationRule;
  //if (parameterSet["interpolationMethod"] != "geodesic")
  //  DUNE_THROW(Exception, "Inconsistent choice of interpolation method");
  typedef GFE::LocalProjectedFEFunction<GridView::dimension, double, typename FEBasis::LocalView::Tree::FiniteElement, TargetSpace> LocalInterpolationRule;
  if (parameterSet["interpolationMethod"] != "projected")
    DUNE_THROW(Exception, "Inconsistent choice of interpolation method");
  std::cout << "Using local interpolation: " << className<LocalInterpolationRule>() << std::endl;

  //////////////////////////////////////////////////////////////////////////////////
  //  Read the data whose error is to be measured
  //////////////////////////////////////////////////////////////////////////////////

  // Input data
  typedef BlockVector<typename TargetSpace::CoordinateType> EmbeddedVectorType;

  EmbeddedVectorType embeddedX(feBasis.size());
  std::ifstream inFile(parameterSet.get<std::string>("simulationData"), std::ios_base::binary);
  if (not inFile)
    DUNE_THROW(IOError, "File " << parameterSet.get<std::string>("simulationData") << " could not be opened.");
  MatrixVector::Generic::readBinary(inFile, embeddedX);
  inFile.peek();   // try to advance beyond the end of the file
  if (not inFile.eof())
    DUNE_THROW(IOError, "File '" << parameterSet.get<std::string>("simulationData") << "' does not have the correct size!");
  inFile.close();

  SolutionType x(embeddedX.size());
  for (size_t i=0; i<x.size(); i++)
    x[i] = TargetSpace(embeddedX[i]);

  // The numerical solution, as a grid function
  GFE::EmbeddedGlobalGFEFunction<FEBasis, LocalInterpolationRule, TargetSpace> numericalSolution(feBasis, x);

  ///////////////////////////////////////////////////////////////////////////
  // Read the reference configuration
  ///////////////////////////////////////////////////////////////////////////
  EmbeddedVectorType embeddedReferenceX(referenceFEBasis.size());
  inFile.open(parameterSet.get<std::string>("referenceData"), std::ios_base::binary);
  if (not inFile)
    DUNE_THROW(IOError, "File " << parameterSet.get<std::string>("referenceData") << " could not be opened.");
  MatrixVector::Generic::readBinary(inFile, embeddedReferenceX);
  inFile.peek();   // try to advance beyond the end of the file
  if (not inFile.eof())
    DUNE_THROW(IOError, "File '" << parameterSet.get<std::string>("referenceData") << "' does not have the correct size!");

  SolutionType referenceX(embeddedReferenceX.size());
  for (size_t i=0; i<referenceX.size(); i++)
    referenceX[i] = TargetSpace(embeddedReferenceX[i]);

  // The reference solution, as a grid function
  GFE::EmbeddedGlobalGFEFunction<FEBasis, LocalInterpolationRule, TargetSpace> referenceSolution(referenceFEBasis, referenceX);

  /////////////////////////////////////////////////////////////////
  //   Measure the discretization error
  /////////////////////////////////////////////////////////////////

  HierarchicSearch<typename GridView::Grid,typename GridView::IndexSet> hierarchicSearch(gridView.grid(), gridView.indexSet());

  if (std::is_same<TargetSpace,RigidBodyMotion<double,3> >::value)
  {
    double deformationL2ErrorSquared = 0;
    double orientationL2ErrorSquared = 0;
    double deformationH1ErrorSquared = 0;
    double orientationH1ErrorSquared = 0;

    for (const auto& rElement : elements(referenceGridView))
    {
      const auto& quadRule = QuadratureRules<double, dim>::rule(rElement.type(), 6);

      for (const auto& qp : quadRule)
      {
        auto integrationElement = rElement.geometry().integrationElement(qp.position());

        auto globalPos = rElement.geometry().global(qp.position());

        auto element = hierarchicSearch.findEntity(globalPos);
        auto localPos = element.geometry().local(globalPos);

        auto refValue = referenceSolution(rElement, qp.position());
        auto numValue = numericalSolution(element, localPos);
        auto diff = refValue - numValue;
        assert(diff.size()==7);

        // Compute error of the displacement degrees of freedom
        for (int i=0; i<3; i++)
          deformationL2ErrorSquared += integrationElement * qp.weight() * diff[i] * diff[i];

        auto derDiff = referenceSolution.derivative(rElement, qp.position()) - numericalSolution.derivative(element, localPos);

        for (int i=0; i<3; i++)
          deformationH1ErrorSquared += integrationElement * qp.weight() * derDiff[i].two_norm2();

        // Compute error of the orientation degrees of freedom
        // We need to transform from quaternion coordinates to matrix coordinates first.
        FieldMatrix<double,3,3> referenceValue, numericalValue;
        Rotation<double,3> referenceRotation(std::array<double,4>{refValue[3], refValue[4], refValue[5], refValue[6]});
        referenceRotation.matrix(referenceValue);

        Rotation<double,3> numericalRotation(std::array<double,4>{numValue[3], numValue[4], numValue[5], numValue[6]});
        numericalRotation.matrix(numericalValue);

        auto orientationDiff = referenceValue - numericalValue;

        orientationL2ErrorSquared += integrationElement * qp.weight() * orientationDiff.frobenius_norm2();

        auto referenceDerQuat = referenceSolution.derivative(rElement, qp.position());
        auto numericalDerQuat = numericalSolution.derivative(element, localPos);

        // Transform to matrix coordinates
        Tensor3<double,3,3,4> derivativeQuaternionToMatrixRef = Rotation<double,3>::derivativeOfQuaternionToMatrix(FieldVector<double,4>{refValue[3], refValue[4], refValue[5], refValue[6]});
        Tensor3<double,3,3,4> derivativeQuaternionToMatrixNum = Rotation<double,3>::derivativeOfQuaternionToMatrix(FieldVector<double,4>{numValue[3], numValue[4], numValue[5], numValue[6]});

        Tensor3<double,3,3,dim> refDerivative(0);
        Tensor3<double,3,3,dim> numDerivative(0);

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<dim; k++)
              for (int l=0; l<4; l++)
              {
                refDerivative[i][j][k] += derivativeQuaternionToMatrixRef[i][j][l] * referenceDerQuat[3+l][k];
                numDerivative[i][j][k] += derivativeQuaternionToMatrixNum[i][j][l] * numericalDerQuat[3+l][k];
              }

        auto derOrientationDiff = refDerivative - numDerivative;  // compute the difference
        orientationH1ErrorSquared += derOrientationDiff.frobenius_norm2() * qp.weight() * integrationElement;
      }
    }

    std::cout << "levels: " << gridView.grid().maxLevel()+1
              << "      "
              << "L^2 error deformation: " << std::sqrt(deformationL2ErrorSquared)
              << "      "
              << "L^2 error orientation: " << std::sqrt(orientationL2ErrorSquared)
              << "      "
              << "H^1 error deformation: " << std::sqrt(deformationH1ErrorSquared)
              << "      "
              << "H^1 error orientation: " << std::sqrt(orientationH1ErrorSquared)
              << std::endl;
  }
  else if constexpr (std::is_same<TargetSpace,Rotation<double,3> >::value)
  {
    double l2ErrorSquared = 0;
    double h1ErrorSquared = 0;

    for (const auto& rElement : elements(referenceGridView))
    {
      const auto& quadRule = QuadratureRules<double, dim>::rule(rElement.type(), 6);

      for (const auto& qp : quadRule)
      {
        auto integrationElement = rElement.geometry().integrationElement(qp.position());

        auto globalPos = rElement.geometry().global(qp.position());

        auto element = hierarchicSearch.findEntity(globalPos);
        auto localPos = element.geometry().local(globalPos);

        FieldMatrix<double,3,3> referenceValue, numericalValue;
        auto refValue = referenceSolution(rElement, qp.position());
        Rotation<double,3> referenceRotation(refValue);
        referenceRotation.matrix(referenceValue);

        auto numValue = numericalSolution(element, localPos);
        Rotation<double,3> numericalRotation(numValue);
        numericalRotation.matrix(numericalValue);

        auto diff = referenceValue - numericalValue;

        l2ErrorSquared += integrationElement * qp.weight() * diff.frobenius_norm2();

        auto referenceDerQuat = referenceSolution.derivative(rElement, qp.position());
        auto numericalDerQuat = numericalSolution.derivative(element, localPos);

        // Transform to matrix coordinates
        Tensor3<double,3,3,4> derivativeQuaternionToMatrixRef = Rotation<double,3>::derivativeOfQuaternionToMatrix(refValue);
        Tensor3<double,3,3,4> derivativeQuaternionToMatrixNum = Rotation<double,3>::derivativeOfQuaternionToMatrix(numValue);

        Tensor3<double,3,3,dim> refDerivative(0);
        Tensor3<double,3,3,dim> numDerivative(0);

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<dim; k++)
              for (int l=0; l<4; l++)
              {
                refDerivative[i][j][k] += derivativeQuaternionToMatrixRef[i][j][l] * referenceDerQuat[l][k];
                numDerivative[i][j][k] += derivativeQuaternionToMatrixNum[i][j][l] * numericalDerQuat[l][k];
              }

        auto derDiff = refDerivative - numDerivative;  // compute the difference
        h1ErrorSquared += derDiff.frobenius_norm2() * qp.weight() * integrationElement;

      }
    }

    std::cout << "levels: " << gridView.grid().maxLevel()+1
              << "      "
              << "L^2 error: " << std::sqrt(l2ErrorSquared)
              << "      "
              << "h^1 error: " << std::sqrt(h1ErrorSquared)
              << std::endl;
  }
  else
  {
  double l2ErrorSquared = 0;
  double h1ErrorSquared = 0;

  for (const auto& rElement : elements(referenceGridView))
  {
    const auto& quadRule = QuadratureRules<double, dim>::rule(rElement.type(), 6);

    for (const auto& qp : quadRule)
    {
      auto integrationElement = rElement.geometry().integrationElement(qp.position());

      // Given a point with local coordinates qp.position() on element rElement of the reference grid
      // find the element and local coordinates on the grid of the numerical simulation
      auto supportingElement = findSupportingElement(referenceGridView.grid(),
                                                     gridView.grid(),
                                                     rElement, qp.position());
      auto element  = std::get<0>(supportingElement);
      auto localPos = std::get<1>(supportingElement);

      auto diff = referenceSolution(rElement, qp.position()) - numericalSolution(element, localPos);

      l2ErrorSquared += integrationElement * qp.weight() * diff.two_norm2();

      auto derDiff = referenceSolution.derivative(rElement, qp.position()) - numericalSolution.derivative(element, localPos);

      h1ErrorSquared += integrationElement * qp.weight() * derDiff.frobenius_norm2();

    }
  }

  std::cout << "levels: " << gridView.grid().maxLevel()+1
            << "      "
            << "L^2 error: " << std::sqrt(l2ErrorSquared)
            << "      "
            << "h^1 error: " << std::sqrt(h1ErrorSquared)
            << std::endl;
  }
}

template <class GridView, int order, class TargetSpace>
void measureAnalyticalEOC(const GridView gridView,
                          const ParameterTree& parameterSet)
{
  typedef std::vector<TargetSpace> SolutionType;

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct the scalar function space bases corresponding to the GFE space
  //////////////////////////////////////////////////////////////////////////////////

  typedef Dune::Functions::LagrangeBasis<GridView, order> FEBasis;
  FEBasis feBasis(gridView);

  //////////////////////////////////////////////////////////////////////////////////
  //  Read the data whose error is to be measured
  //////////////////////////////////////////////////////////////////////////////////

  // Input data
  typedef BlockVector<typename TargetSpace::CoordinateType> EmbeddedVectorType;

  EmbeddedVectorType embeddedX(feBasis.size());
  std::ifstream inFile(parameterSet.get<std::string>("simulationData"), std::ios_base::binary);
  if (not inFile)
    DUNE_THROW(IOError, "File " << parameterSet.get<std::string>("simulationData") << " could not be opened.");
  MatrixVector::Generic::readBinary(inFile, embeddedX);
  inFile.peek();   // try to advance beyond the end of the file
  if (not inFile.eof())
    DUNE_THROW(IOError, "File '" << parameterSet.get<std::string>("simulationData") << "' does not have the correct size!");
  inFile.close();

  SolutionType x(embeddedX.size());
  for (size_t i=0; i<x.size(); i++)
    x[i] = TargetSpace(embeddedX[i]);

  /////////////////////////////////////////////////////////////////
  //   Measure the discretization error
  /////////////////////////////////////////////////////////////////

  // Read reference solution and its derivative into a PythonFunction
  typedef VirtualDifferentiableFunction<FieldVector<double, dimworld>, typename TargetSpace::CoordinateType> FBase;

  Python::Module module = Python::import(parameterSet.get<std::string>("referenceSolution"));
  auto referenceSolution = module.get("fdf").toC<std::shared_ptr<FBase>>();

  // The numerical solution, as a grid function
  std::unique_ptr<VirtualGridViewFunction<GridView, typename TargetSpace::CoordinateType> > numericalSolution;

  if (parameterSet["interpolationMethod"] == "geodesic")
    numericalSolution = std::make_unique<GFE::EmbeddedGlobalGFEFunction<FEBasis,
                                                                        LocalGeodesicFEFunction<dim, double, typename FEBasis::LocalView::Tree::FiniteElement, TargetSpace>,
                                                                        TargetSpace> > (feBasis, x);

  if (parameterSet["interpolationMethod"] == "projected")
    numericalSolution = std::make_unique<GFE::EmbeddedGlobalGFEFunction<FEBasis,
                                                                        GFE::LocalProjectedFEFunction<dim, double, typename FEBasis::LocalView::Tree::FiniteElement, TargetSpace>,
                                                                        TargetSpace> > (feBasis, x);

  // QuadratureRule for the integral of the L^2 error
  QuadratureRuleKey quadKey(dim,6);

  // Compute errors in the L2 norm and the h1 seminorm.
  // SO(3)-valued maps need special treatment, because they are stored as quaternions,
  // but the errors need to be computed in matrix space.
  if constexpr (std::is_same<TargetSpace,Rotation<double,3> >::value)
  {
    constexpr int blocksize = TargetSpace::CoordinateType::dimension;

    // The error to be computed
    double l2ErrorSquared = 0;
    double h1ErrorSquared = 0;

    for (auto&& element : elements(gridView))
    {
      // Get quadrature formula
      quadKey.setGeometryType(element.type());
      const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

      for (auto quadPoint : quad)
      {
        auto quadPos = quadPoint.position();
        const auto integrationElement = element.geometry().integrationElement(quadPos);
        const auto weight = quadPoint.weight();

        // Evaluate function a
        FieldVector<double,blocksize> numValue;
        numericalSolution.get()->evaluateLocal(element, quadPos,numValue);

        // Evaluate function b.  If it is a grid function use that to speed up the evaluation
        FieldVector<double,blocksize> refValue;
        if (std::dynamic_pointer_cast<const VirtualGridViewFunction<GridView,FieldVector<double,blocksize> >>(referenceSolution))
            std::dynamic_pointer_cast<const VirtualGridViewFunction<GridView,FieldVector<double,blocksize> >>(referenceSolution)->evaluateLocal(element,
                                                                                                                          quadPos,
                                                                                                                          refValue
                                                                                                                         );
        else
          referenceSolution->evaluate(element.geometry().global(quadPos), refValue);

        // Get error in matrix space
        Rotation<double,3> numRotation(numValue);
        FieldMatrix<double,3,3> numValueMatrix;
        numRotation.matrix(numValueMatrix);

        Rotation<double,3> refRotation(refValue);
        FieldMatrix<double,3,3> refValueMatrix;
        refRotation.matrix(refValueMatrix);

        // Evaluate derivatives in quaternion space
        FieldMatrix<double,blocksize,dimworld> num_di;
        FieldMatrix<double,blocksize,dimworld> ref_di;

        if (dynamic_cast<const VirtualGridViewFunction<GridView,FieldVector<double,blocksize> >*>(numericalSolution.get()))
          dynamic_cast<const VirtualGridViewFunction<GridView,FieldVector<double,blocksize> >*>(numericalSolution.get())->evaluateDerivativeLocal(element,
                                                                                                                                quadPos,
                                                                                                                                num_di);
        else
          numericalSolution->evaluateDerivative(element.geometry().global(quadPos), num_di);

        if (std::dynamic_pointer_cast<const VirtualGridViewFunction<GridView,FieldVector<double,blocksize> >>(referenceSolution))
        std::dynamic_pointer_cast<const VirtualGridViewFunction<GridView,FieldVector<double,blocksize> >>(referenceSolution)->evaluateDerivativeLocal(element,
                                                                                                                                quadPos,
                                                                                                                                ref_di);
        else
          referenceSolution->evaluateDerivative(element.geometry().global(quadPos), ref_di);

        // Transform into matrix space
        Tensor3<double,3,3,4> derivativeQuaternionToMatrixNum = Rotation<double,3>::derivativeOfQuaternionToMatrix(numValue);
        Tensor3<double,3,3,4> derivativeQuaternionToMatrixRef = Rotation<double,3>::derivativeOfQuaternionToMatrix(refValue);

        Tensor3<double,3,3,dim> numDerivative(0);
        Tensor3<double,3,3,dim> refDerivative(0);

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<dim; k++)
              for (int l=0; l<blocksize; l++)
              {
                numDerivative[i][j][k] += derivativeQuaternionToMatrixNum[i][j][l] * num_di[l][k];
                refDerivative[i][j][k] += derivativeQuaternionToMatrixRef[i][j][l] * ref_di[l][k];
              }

        // integrate error
        l2ErrorSquared += (numValueMatrix - refValueMatrix).frobenius_norm2() * weight * integrationElement;
        auto diff = numDerivative - refDerivative;
        h1ErrorSquared += diff.frobenius_norm2() * weight * integrationElement;
      }
    }

    std::cout << "elements: " << gridView.size(0)
              << "      "
              << "L^2 error: " << std::sqrt(l2ErrorSquared)
              << "      ";
    std::cout << "h^1 error: " << std::sqrt(h1ErrorSquared) << std::endl;
  }
  else
  {
    auto l2Error = DiscretizationError<GridView>::computeL2Error(numericalSolution.get(),
                                                            referenceSolution.get(),
                                                            quadKey);

    auto h1Error = DiscretizationError<GridView>::computeH1HalfNormDifferenceSquared(gridView,
                                                                                numericalSolution.get(),
                                                                                referenceSolution.get(),
                                                                                quadKey);

    std::cout << "elements: " << gridView.size(0)
              << "      "
              << "L^2 error: " << l2Error
              << "      ";
    std::cout << "h^1 error: " << std::sqrt(h1Error) << std::endl;
  }
}

template <class GridType, class TargetSpace>
void measureEOC(const std::shared_ptr<GridType> grid,
                const std::shared_ptr<GridType> referenceGrid,
                const ParameterTree& parameterSet)
{
  const int order = parameterSet.get<int>("order");

  if (parameterSet.get<std::string>("discretizationErrorMode")=="discrete")
  {
    referenceGrid->globalRefine(parameterSet.get<int>("numReferenceLevels")-1);

    switch (order)
    {
      case 1:
      measureDiscreteEOC<typename GridType::LeafGridView,1,TargetSpace>(grid->leafGridView(), referenceGrid->leafGridView(), parameterSet);
      break;

      case 2:
      measureDiscreteEOC<typename GridType::LeafGridView,2,TargetSpace>(grid->leafGridView(), referenceGrid->leafGridView(), parameterSet);
      break;

      case 3:
      measureDiscreteEOC<typename GridType::LeafGridView,3,TargetSpace>(grid->leafGridView(), referenceGrid->leafGridView(), parameterSet);
      break;

      default:
        DUNE_THROW(NotImplemented, "Order '" << order << "' is not implemented");
    }
    return;  // Success
  }

  if (parameterSet.get<std::string>("discretizationErrorMode")=="analytical")
  {
    switch (order)
    {
      case 1:
      measureAnalyticalEOC<typename GridType::LeafGridView,1,TargetSpace>(grid->leafGridView(), parameterSet);
      break;

      case 2:
      measureAnalyticalEOC<typename GridType::LeafGridView,2,TargetSpace>(grid->leafGridView(), parameterSet);
      break;

      case 3:
      measureAnalyticalEOC<typename GridType::LeafGridView,3,TargetSpace>(grid->leafGridView(), parameterSet);
      break;

      default:
        DUNE_THROW(NotImplemented, "Order '" << order << "' is not implemented");
    }
    return;  // Success
  }

  DUNE_THROW(NotImplemented, "Unknown discretization error mode encountered!");
}

int main (int argc, char *argv[]) try
{
  MPIHelper::instance(argc, argv);

  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  Python::runStream()
      << std::endl << "import sys"
      << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/problems')"
      << std::endl;

  // parse data file
  ParameterTree parameterSet;
  if (argc < 2)
    DUNE_THROW(Exception, "Usage: ./compute-disc-error <parameter file>");

  ParameterTreeParser::readINITree(argv[1], parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // Print all parameters, to have them in the log file
  parameterSet.report();

  /////////////////////////////////////////
  //    Create the grids
  /////////////////////////////////////////
#if HAVE_DUNE_FOAMGRID
    typedef std::conditional<dim==1 or dim!=dimworld,FoamGrid<dim,dimworld>,UGGrid<dim> >::type GridType;
#else
    static_assert(dim==dimworld, "You need to have dune-foamgrid installed for dim != dimworld!");
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;
#endif

  const int numLevels = parameterSet.get<int>("numLevels");

  std::shared_ptr<GridType> grid, referenceGrid;

  FieldVector<double,dimworld> lower(0), upper(1);

  std::string structuredGridType = parameterSet["structuredGrid"];
  if (structuredGridType == "sphere")
  {
    if constexpr (dimworld==3)
    {
      grid          = makeSphereOnOctahedron<GridType>({0,0,0},1);
      referenceGrid = makeSphereOnOctahedron<GridType>({0,0,0},1);
    } else
      DUNE_THROW(Exception, "Dimworld must be 3 to create a sphere grid!");
  }
  else if (structuredGridType == "simplex" || structuredGridType == "cube")
  {
    lower = parameterSet.get<FieldVector<double,dimworld> >("lower");
    upper = parameterSet.get<FieldVector<double,dimworld> >("upper");

    auto elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
    if (structuredGridType == "simplex")
    {
      grid          = StructuredGridFactory<GridType>::createSimplexGrid(lower, upper, elements);
      referenceGrid = StructuredGridFactory<GridType>::createSimplexGrid(lower, upper, elements);
    } else if (structuredGridType == "cube")
    {
      grid          = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
      referenceGrid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
    } else
      DUNE_THROW(Exception, "Unknown structured grid type '" << structuredGridType << "' found!");
  }
  else
  {
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");
    grid          = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
    referenceGrid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->globalRefine(numLevels-1);

  // Do the actual measurement
  const int targetDim = parameterSet.get<int>("targetDim");
  const std::string targetSpace = parameterSet.get<std::string>("targetSpace");

  switch (targetDim)
  {
    case 1:
      if (targetSpace=="RealTuple")
      {
        measureEOC<GridType,RealTuple<double,1> >(grid,
                                                  referenceGrid,
                                                  parameterSet);
      } else if (targetSpace=="UnitVector")
      {
        measureEOC<GridType,UnitVector<double,1> >(grid,
                                                   referenceGrid,
                                                   parameterSet);
      } else
        DUNE_THROW(NotImplemented, "Target space '" << targetSpace << "' is not implemented");
      break;

    case 2:
      if (targetSpace=="RealTuple")
      {
        measureEOC<GridType,RealTuple<double,2> >(grid,
                                                  referenceGrid,
                                                  parameterSet);
      } else if (targetSpace=="UnitVector")
      {
        measureEOC<GridType,UnitVector<double,2> >(grid,
                                                   referenceGrid,
                                                   parameterSet);
#if 0
      } else if (targetSpace=="Rotation")
      {
        measureEOC<GridType,Rotation<double,2> >(grid,
                                                 referenceGrid,
                                                 parameterSet);
      } else if (targetSpace=="RigidBodyMotion")
      {
        measureEOC<GridType,RigidBodyMotion<double,2> >(grid,
                                                        referenceGrid,
                                                        parameterSet);
#endif
      } else
        DUNE_THROW(NotImplemented, "Target space '" << targetSpace << "' is not implemented");
      break;

    case 3:
      if (targetSpace=="RealTuple")
      {
        measureEOC<GridType,RealTuple<double,3> >(grid,
                                                  referenceGrid,
                                                  parameterSet);
      } else if (targetSpace=="UnitVector")
      {
        measureEOC<GridType,UnitVector<double,3> >(grid,
                                                   referenceGrid,
                                                   parameterSet);
      } else if (targetSpace=="Rotation")
      {
        measureEOC<GridType,Rotation<double,3> >(grid,
                                                 referenceGrid,
                                                 parameterSet);
      } else if (targetSpace=="RigidBodyMotion")
      {
        measureEOC<GridType,RigidBodyMotion<double,3> >(grid,
                                                        referenceGrid,
                                                        parameterSet);
      } else
        DUNE_THROW(NotImplemented, "Target space '" << targetSpace << "' is not implemented");
      break;

    case 4:
      if (targetSpace=="RealTuple")
      {
        measureEOC<GridType,RealTuple<double,4> >(grid,
                                                  referenceGrid,
                                                  parameterSet);
      } else if (targetSpace=="UnitVector")
      {
        measureEOC<GridType,UnitVector<double,4> >(grid,
                                                   referenceGrid,
                                                   parameterSet);
#if 0
      } else if (targetSpace=="Rotation")
      {
        measureEOC<GridType,Rotation<double,4> >(grid,
                                                 referenceGrid,
                                                 parameterSet);
      } else if (targetSpace=="RigidBodyMotion")
      {
        measureEOC<GridType,RigidBodyMotion<double,4> >(grid,
                                                        referenceGrid,
                                                        parameterSet);
#endif
      } else
        DUNE_THROW(NotImplemented, "Target space '" << targetSpace << "' is not implemented");
      break;

    default:
      DUNE_THROW(NotImplemented, "Target dimension '" << targetDim << "' is not implemented");
  }

  return 0;
}
catch (Exception& e)
{
  std::cout << e << std::endl;
  return 1;
}
