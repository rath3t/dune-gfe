#include <config.h>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/functions/functionspacebases/pqknodalbasis.hh>

#include <dune/fufem/discretizationerror.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>
#include <dune/gfe/cosseratvtkwriter.hh>

// grid dimension
const int dim = 2;
const int dimworld = 2;

using namespace Dune;

template <class GridView, int order, class TargetSpace>
void measureDiscreteEOC(const GridView gridView,
                        const GridView referenceGridView,
                        const ParameterTree& parameterSet)
{
  typedef std::vector<TargetSpace> SolutionType;

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct the scalar function space bases corresponding to the GFE space
  //////////////////////////////////////////////////////////////////////////////////

  typedef Dune::Functions::PQkNodalBasis<GridView, order> FEBasis;
  FEBasis feBasis(gridView);
  FEBasis referenceFEBasis(referenceGridView);

  //////////////////////////////////////////////////////////////////////////////////
  //  Read the data whose error is to be measured
  //////////////////////////////////////////////////////////////////////////////////

  // Input data
  typedef BlockVector<typename TargetSpace::CoordinateType> EmbeddedVectorType;

  EmbeddedVectorType embeddedX(feBasis.size());
  std::ifstream inFile(parameterSet.get<std::string>("simulationData"), std::ios_base::binary);
  if (not inFile)
    DUNE_THROW(IOError, "File " << parameterSet.get<std::string>("simulationData") << " could not be opened.");
  GenericVector::readBinary(inFile, embeddedX);
  inFile.peek();   // try to advance beyond the end of the file
  if (not inFile.eof())
    DUNE_THROW(IOError, "File '" << parameterSet.get<std::string>("simulationData") << "' does not have the correct size!");
  inFile.close();

  SolutionType x(embeddedX.size());
  for (size_t i=0; i<x.size(); i++)
    x[i] = TargetSpace(embeddedX[i]);

  // The numerical solution, as a grid function
  GFE::EmbeddedGlobalGFEFunction<FEBasis, TargetSpace> numericalSolution(feBasis, x);

  ///////////////////////////////////////////////////////////////////////////
  // Read the reference configuration
  ///////////////////////////////////////////////////////////////////////////
  EmbeddedVectorType embeddedReferenceX(referenceFEBasis.size());
  inFile.open(parameterSet.get<std::string>("referenceData"), std::ios_base::binary);
  if (not inFile)
    DUNE_THROW(IOError, "File " << parameterSet.get<std::string>("referenceData") << " could not be opened.");
  GenericVector::readBinary(inFile, embeddedReferenceX);
  inFile.peek();   // try to advance beyond the end of the file
  if (not inFile.eof())
    DUNE_THROW(IOError, "File '" << parameterSet.get<std::string>("referenceData") << "' does not have the correct size!");

  SolutionType referenceX(embeddedReferenceX.size());
  for (size_t i=0; i<referenceX.size(); i++)
    referenceX[i] = TargetSpace(embeddedReferenceX[i]);

  // The reference solution, as a grid function
  GFE::EmbeddedGlobalGFEFunction<FEBasis, TargetSpace> referenceSolution(referenceFEBasis, referenceX);

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

        auto diff = referenceSolution(rElement, qp.position()) - numericalSolution(element, localPos);
        assert(diff.size()==7);

        for (int i=0; i<3; i++)
          deformationL2ErrorSquared += integrationElement * qp.weight() * diff[i] * diff[i];

        for (int i=3; i<7; i++)
          orientationL2ErrorSquared += integrationElement * qp.weight() * diff[i] * diff[i];

        auto derDiff = referenceSolution.derivative(rElement, qp.position()) - numericalSolution.derivative(element, localPos);

        std::cout << "size: " << derDiff.N() << ", " << derDiff.M() << std::endl;

        for (int i=0; i<3; i++)
          deformationH1ErrorSquared += integrationElement * qp.weight() * derDiff[i].two_norm2();

        for (int i=3; i<7; i++)
          orientationH1ErrorSquared += integrationElement * qp.weight() * derDiff[i].two_norm2();

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

      auto globalPos = rElement.geometry().global(qp.position());

      auto element = hierarchicSearch.findEntity(globalPos);
      auto localPos = element.geometry().local(globalPos);

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
            << "H^1 error: " << std::sqrt(l2ErrorSquared + h1ErrorSquared)
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

  typedef Dune::Functions::PQkNodalBasis<GridView, order> FEBasis;
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
  GenericVector::readBinary(inFile, embeddedX);
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
  typedef VirtualDifferentiableFunction<FieldVector<double, dim>, typename TargetSpace::CoordinateType> FBase;

  Python::Module module = Python::import(parameterSet.get<std::string>("referenceSolution"));
  auto referenceSolution = module.get("fdf").toC<std::shared_ptr<FBase>>();

  // The numerical solution, as a grid function
  GFE::EmbeddedGlobalGFEFunction<FEBasis, TargetSpace> numericalSolution(feBasis, x);

  // QuadratureRule for the integral of the L^2 error
  QuadratureRuleKey quadKey(dim,6);

  // Compute the embedded L^2 error
  double l2Error = DiscretizationError<GridView>::computeL2Error(&numericalSolution,
                                                                 referenceSolution.get(),
                                                                 quadKey);

  // Compute the embedded H^1 error
  double h1Error = DiscretizationError<GridView>::computeH1HalfNormDifferenceSquared(gridView,
                                                                                     &numericalSolution,
                                                                                     referenceSolution.get(),
                                                                                     quadKey);

  std::cout << "elements: " << gridView.size(0)
            << "      "
            << "L^2 error: " << l2Error
            << "      ";
  std::cout << "H^1 error: " << std::sqrt(l2Error*l2Error + h1Error) << std::endl;
}

template <class GridType, class TargetSpace>
void measureEOC(const std::shared_ptr<GridType> grid,
                const std::shared_ptr<GridType> referenceGrid,
                const ParameterTree& parameterSet)
{
  const int order = parameterSet.get<int>("order");

  if (parameterSet.get<std::string>("discretizationErrorMode")=="discrete")
  {
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
  }
}

int main (int argc, char *argv[]) try
{
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
  typedef UGGrid<dim> GridType;

  const int numLevels = parameterSet.get<int>("numLevels");

  shared_ptr<GridType> grid, referenceGrid;

  FieldVector<double,dimworld> lower(0), upper(1);

  if (parameterSet.get<bool>("structuredGrid"))
  {
    lower = parameterSet.get<FieldVector<double,dimworld> >("lower");
    upper = parameterSet.get<FieldVector<double,dimworld> >("upper");

    array<unsigned int,dim> elements = parameterSet.get<array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
    referenceGrid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
  }
  else
  {
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");
    grid = shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
    referenceGrid = shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->globalRefine(numLevels-1);
  referenceGrid->globalRefine(parameterSet.get<int>("numReferenceLevels")-1);

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
catch (Exception e)
{
  std::cout << e << std::endl;
  return 1;
}
