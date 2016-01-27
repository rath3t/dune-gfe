// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/pqknodalbasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/discretizationerror.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>

// grid dimension
const int dim = 1;

// Image space of the geodesic fe functions
// typedef Rotation<double,2> TargetSpace;
// typedef Rotation<double,3> TargetSpace;
// typedef UnitVector<double,2> TargetSpace;
typedef UnitVector<double,3> TargetSpace;
// typedef UnitVector<double,4> TargetSpace;
// typedef RealTuple<double,1> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

// Approximation order of the finite element space
const int order = 1;

using namespace Dune;


int main (int argc, char *argv[]) try
{
  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  //feenableexcept(FE_INVALID);
  Python::runStream()
      << std::endl << "import sys"
      << std::endl << "sys.path.append('../../problems/')"
      << std::endl;

  typedef std::vector<TargetSpace> SolutionType;

  // parse data file
  ParameterTree parameterSet;
  if (argc < 2)
    DUNE_THROW(Exception, "Usage: ./gradient-flow <parameter file>");

  ParameterTreeParser::readINITree(argv[1], parameterSet);
  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read problem settings
  const int numLevels                   = parameterSet.get<int>("numLevels");

  // read solver settings
  const double tolerance                = parameterSet.get<double>("tolerance");
  const int maxTrustRegionSteps         = parameterSet.get<int>("maxTrustRegionSteps");
  const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
  const int multigridIterations         = parameterSet.get<int>("numIt");
  const int nu1                         = parameterSet.get<int>("nu1");
  const int nu2                         = parameterSet.get<int>("nu2");
  const int mu                          = parameterSet.get<int>("mu");
  const int baseIterations              = parameterSet.get<int>("baseIt");
  const double mgTolerance              = parameterSet.get<double>("mgTolerance");
  const double baseTolerance            = parameterSet.get<double>("baseTolerance");
  std::string resultPath                = parameterSet.get("resultPath", "");

  /////////////////////////////////////////
  //    Create the grid
  /////////////////////////////////////////
  typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

  std::shared_ptr<GridType> grid;
  FieldVector<double,dim> lower(0), upper(1);
  std::array<unsigned int,dim> elements;

  if (parameterSet.get<bool>("structuredGrid"))
  {
    lower = parameterSet.get<FieldVector<double,dim> >("lower");
    upper = parameterSet.get<FieldVector<double,dim> >("upper");

    elements = parameterSet.get<array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);
  }
  else
  {
    std::string path     = parameterSet.get<std::string>("path");
    std::string gridFile = parameterSet.get<std::string>("gridFile");

    grid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->globalRefine(numLevels-1);

  //////////////////////////////////////////////////////////////////////////////////
  //  Construct the scalar function space basis corresponding to the GFE space
  //////////////////////////////////////////////////////////////////////////////////

  typedef Dune::Functions::PQkNodalBasis<typename GridType::LeafGridView, order> FEBasis;
  FEBasis feBasis(grid->leafGridView());

  ///////////////////////////////////////////
  //   Read Dirichlet values
  ///////////////////////////////////////////

  typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
  FufemFEBasis fufemFeBasis(feBasis);

  BitSetVector<1> allNodes(grid->size(dim));
  allNodes.setAll();
  BoundaryPatch<typename GridType::LeafGridView> dirichletBoundary(grid->leafGridView(), allNodes);

  BitSetVector<blocksize> dirichletNodes;
  constructBoundaryDofs(dirichletBoundary,fufemFeBasis,dirichletNodes);

  ////////////////////////////
  //   Initial iterate
  ////////////////////////////

  // Read initial iterate into a PythonFunction
  typedef PythonFunction<FieldVector<double, dim>, TargetSpace::CoordinateType> FBase;

  Python::Module module = Python::import(parameterSet.get<std::string>("initialIterate"));
  auto pythonInitialIterate = module.get("f").toC<std::shared_ptr<FBase>>();

  std::vector<TargetSpace::CoordinateType> v;
  ::Functions::interpolate(fufemFeBasis, v, *pythonInitialIterate);

  SolutionType x(feBasis.indexSet().size());

  for (size_t i=0; i<x.size(); i++)
    x[i] = v[i];

  //////////////////////////////////////////////////////////////
  //   Create an assembler for the harmonic energy functional
  //////////////////////////////////////////////////////////////

  // Assembler using ADOL-C
  typedef TargetSpace::rebind<adouble>::other ATargetSpace;
  std::shared_ptr<LocalGeodesicFEStiffness<FEBasis,ATargetSpace> > localEnergy;
  localEnergy.reset(new HarmonicEnergyLocalStiffness<FEBasis, ATargetSpace>);

  LocalGeodesicFEADOLCStiffness<FEBasis,TargetSpace> localGFEADOLCStiffness(localEnergy.get());

  GeodesicFEAssembler<FEBasis,TargetSpace> assembler(feBasis, &localGFEADOLCStiffness);

  ///////////////////////////////////////////////////
  //   Create a Riemannian trust-region solver
  ///////////////////////////////////////////////////

  RiemannianTrustRegionSolver<FEBasis,TargetSpace> solver;
  solver.setup(*grid,
               &assembler,
               x,
               dirichletNodes,
               tolerance,
               maxTrustRegionSteps,
               initialTrustRegionRadius,
               multigridIterations,
               mgTolerance,
               mu, nu1, nu2,
               baseIterations,
               baseTolerance,
               false);   // instrumentation

  ///////////////////////////////////////////////////////
  //   Solve!
  ///////////////////////////////////////////////////////

  solver.setInitialIterate(x);
  solver.solve();

  x = solver.getSol();

  ////////////////////////////////
  //   Output result
  ////////////////////////////////

  typedef BlockVector<TargetSpace::CoordinateType> EmbeddedVectorType;
  EmbeddedVectorType xEmbedded(x.size());
  for (size_t i=0; i<x.size(); i++)
    xEmbedded[i] = x[i].globalCoordinates();

  auto xFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<TargetSpace::CoordinateType>(feBasis,
                                                                                                 TypeTree::hybridTreePath(),
                                                                                                 xEmbedded);

  SubsamplingVTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView(),0);
  vtkWriter.addVertexData(xFunction, VTK::FieldInfo("orientation", VTK::FieldInfo::Type::scalar, xEmbedded[0].size()));
  vtkWriter.write("gradientflow_result");

  return 0;
}
catch (Exception e)
{
  std::cout << e << std::endl;
  return 1;
}
