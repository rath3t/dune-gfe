#include <config.h>
#include <memory>
#include <array>
#include <math.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/typetraits.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/gridfunctions/composedgridfunction.hh>
#include <dune/functions/functionspacebases/cubichermitebasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/dunepython.hh>
#include <dune/fufem/discretizationerror.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/functions/discretekirchhoffbendingisometry.hh>
#include <dune/gfe/functions/embeddedglobalgfefunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/assemblers/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/assemblers/geodesicfeassembler.hh>
#include <dune/gfe/assemblers/discretekirchhoffbendingenergy.hh>
#include <dune/gfe/assemblers/forceenergy.hh>
#include <dune/gfe/assemblers/sumenergy.hh>
#include <dune/gfe/bendingisometryhelper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>

#include <dune/gmsh4/gmsh4reader.hh>
#include <dune/gmsh4/gridcreators/lagrangegridcreator.hh>

#include <dune/vtk/vtkwriter.hh>
#include <dune/vtk/writers/unstructuredgridwriter.hh>
#include <dune/vtk/datacollectors/continuousdatacollector.hh>
#include <dune/vtk/datacollectors/discontinuousdatacollector.hh>
#include <dune/vtk/datacollectors/quadraticdatacollector.hh>
#include <dune/vtk/datacollectors/lagrangedatacollector.hh>

const int dim = 2;

using namespace Dune;

/**
    Source file for the computation of bending isometries.

    Usage:
           ./bending-isometries <python path> <python module without extension>
 */

int main(int argc, char *argv[])
{
  MPIHelper::instance(argc, argv);
  Dune::Timer globalTimer;

  if (argc < 3)
    DUNE_THROW(Exception, "Usage: ./bending-isometries <python path> <python module without extension>");

  // Start Python interpreter
  Python::start();
  auto pyMain = Python::main();
  pyMain.runStream()
    << std::endl << "import math"
    << std::endl << "import sys"
    << std::endl << "sys.path.append('" << argv[1] << "')"  << std::endl;
  auto pyModule = pyMain.import(argv[2]);

  std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  std::filesystem::path file_path = (__FILE__);
  std::cout<< "File path: " << file_path<<std::endl;
  std::cout << "dir_path: " << file_path.parent_path() << std::endl;
  std::string dir_path  = file_path.parent_path();

  // parse data file
  ParameterTree parameterSet;
  pyModule.get("parameterSet").toC(parameterSet);

  // read possible further parameters from the command line
  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // Print all parameters, to make them appear in the log file
  std::cout << "Executable: bending-isometries, with parameters:" << std::endl;
  parameterSet.report();

  bool PRINT_DEBUG = parameterSet.get<bool>("print_debug", 0);

  /////////////////////////////////////////
  //   Create the grid
  /////////////////////////////////////////
  using GridType = UGGrid<dim>;

  std::shared_ptr<GridType> grid;
  FieldVector<double,dim> lower(0), upper(1);
  std::array<unsigned int,dim> elementsArray;

  std::string structuredGridType = parameterSet["structuredGrid"];
  if (structuredGridType != "false" )
  {
    lower = parameterSet.get<FieldVector<double,dim> >("lower");
    upper = parameterSet.get<FieldVector<double,dim> >("upper");
    elementsArray = parameterSet.get<std::array<unsigned int,dim> >("elements");

    if (structuredGridType == "simplex")
      grid = StructuredGridFactory<GridType>::createSimplexGrid(lower, upper, elementsArray);
    else if (structuredGridType == "cube")
      grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elementsArray);
    else
      DUNE_THROW(Exception, "Unknown structured grid type '" << structuredGridType << "' found!");
  } else {
    std::cout << "Read GMSH grid." << std::endl;
    std::string gridPath = parameterSet.get<std::string>("gridPath");
    std::string gridFile = parameterSet.get<std::string>("gridFile");
    GridFactory<GridType> factory;
    Gmsh4::LagrangeGridCreator creator{factory};
    Gmsh4Reader reader{creator};
    reader.read(gridPath + "/" + gridFile);
    grid = factory.createGrid();
  }

  const int macroGridLevel = parameterSet.get<int>("macroGridLevel");
  grid->globalRefine(macroGridLevel-1);

  using GridView = typename GridType::LeafGridView;
  GridView gridView = grid->leafGridView();

  ///////////////////////////////////////////////////////
  //  Set up the function spaces
  ///////////////////////////////////////////////////////
  // General coefficient vector of a Discrete Kirchhoff deformation function
  using VectorSpaceCoefficients = BlockVector<FieldVector<double,3> >;

  // Coefficient vector of a Discrete Kirchhoff deformation function that is constrained to be an isometry.
  // we need both 'double' and 'adouble' versions.
  using Coefficient = GFE::ProductManifold<GFE::RealTuple<double,3>, GFE::Rotation<double,3> >;
  using IsometryCoefficients = std::vector<Coefficient>;
  using ACoefficient = typename Coefficient::template rebind<adouble>::other;
  using AIsometryCoefficients = std::vector<ACoefficient>;

  using namespace Functions::BasisFactory;
  auto deformationBasis = makeBasis(gridView,
                                    power<3>(reducedCubicHermite(),
                                             blockedInterleaved()));
  using DeformationBasis = decltype(deformationBasis);


  // The next basis is used to assign (nonlinear) degrees of freedom to the grid vertices.
  // The actual basis function values are never used.
  using CoefficientBasis = Functions::LagrangeBasis<GridView, 1>;
  CoefficientBasis coefficientBasis(gridView);

  // A basis for the tangent space (used to set DirichletNodes)
  auto tangentBasis = makeBasis(gridView,
                                power<Coefficient::TangentVector::dimension>(
                                  lagrange<1>(),
                                  blockedInterleaved()));


  // Print some information on the grid and degrees of freedom.
  std::cout << "Coefficient::TangentVector::dimension: " << Coefficient::TangentVector::dimension<< std::endl;
  std::cout << "Number of Elements in the grid: " << gridView.size(0)<< std::endl;
  std::cout << "Number of Nodes in the grid: "    << gridView.size(dim)<< std::endl;
  std::cout << "deformationBasis.size(): "        << deformationBasis.size() << std::endl;
  std::cout << "deformationBasis.dimension(): "   << deformationBasis.dimension() << std::endl;
  std::cout << "Degrees of Freedom: "             << deformationBasis.dimension() << std::endl;

  ///////////////////////////////////////////
  //   Read Dirichlet values
  ///////////////////////////////////////////
  BitSetVector<1> dirichletVertices(gridView.size(dim), false);
  const typename GridView::IndexSet &indexSet = gridView.indexSet();
  BitSetVector<Coefficient::TangentVector::dimension> dirichletNodes(tangentBasis.size(), false); //tangentBasis.size()=coefficientBasis.size()

  // Make Python function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  auto dirichletIndicatorFunction = Python::make_function<bool>(pyModule.get("dirichlet_indicator"));

  // If we want to clamp DOFs inside the domain, we cannot use 'BoundaryPatch'
  // and 'constructBoundaryDofs'. This is a workaround for now.
  for (auto &&vertex : vertices(gridView))
  {
    dirichletVertices[indexSet.index(vertex)] = dirichletIndicatorFunction(vertex.geometry().corner(0));

    if(dirichletIndicatorFunction(vertex.geometry().corner(0)))
    {
      dirichletNodes[indexSet.index(vertex)] = true;
      if(PRINT_DEBUG)
        std::cout << "Dirichlet Vertex with coordinates:" << vertex.geometry().corner(0) << std::endl;
    }
  }

  ///////////////////////////////////////////
  //   Get initial Iterate
  ///////////////////////////////////////////
  auto pythonInitialIterate = Python::makeDifferentiableFunction<FieldVector<double,3>(FieldVector<double,2>)>(pyModule.get("f"), pyModule.get("df"));
  VectorSpaceCoefficients x(deformationBasis.size());
  interpolate(deformationBasis, x, pythonInitialIterate);



  // We need to setup DiscreteKirchhoffBendingIsometry with a coefficient
  // vector of ctype 'adouble' while the solver gets a coefficient vector
  // of ctype 'double'.
  IsometryCoefficients isometryCoefficients(coefficientBasis.size());
  AIsometryCoefficients isometryCoefficients_adouble(coefficientBasis.size());

  using namespace Dune::GFE::Impl;
  // Copy the current iterate into a data type that encapsulates the isometry constraint
  // i.e. convert coefficient data structure from 'VectorSpaceCoefficients' to 'IsometryCoefficients'
  vectorToIsometryCoefficientMap(deformationBasis,coefficientBasis,x,isometryCoefficients);
  vectorToIsometryCoefficientMap(deformationBasis,coefficientBasis,x,isometryCoefficients_adouble);

  // Create a DiscreteKirchhoffBendingIsometry.
  // This serves as the deformation function.
  using LocalDKFunction = GFE::DiscreteKirchhoffBendingIsometry<DeformationBasis, CoefficientBasis, AIsometryCoefficients>;
  LocalDKFunction localDKFunction(deformationBasis, coefficientBasis, isometryCoefficients_adouble);

  // Read the force term.
  auto pythonForce = Python::make_function<FieldVector<double,3> >(pyModule.get("force"));
  auto forceGVF  = Dune::Functions::makeGridViewFunction(pythonForce, gridView);
  auto localForce = localFunction(forceGVF);

  ////////////////////////////////////////////////////////////////////////
  //   Create an assembler for the Discrete Kirchhoff Energy Functional  (using ADOL-C)
  ////////////////////////////////////////////////////////////////////////

  // Setup nonconforming energy and assembler - only option for this minimal example.
  auto forceEnergy = std::make_shared<GFE::ForceEnergy<CoefficientBasis, LocalDKFunction, decltype(localForce), ACoefficient> >(localDKFunction, localForce);
  auto localEnergy_nonconforming = std::make_shared<GFE::DiscreteKirchhoffBendingEnergy<CoefficientBasis, LocalDKFunction, decltype(localForce), ACoefficient> >(localDKFunction);

  auto sumEnergy = std::make_shared<GFE::SumEnergy<CoefficientBasis, GFE::RealTuple<adouble,3>, GFE::Rotation<adouble,3> > >();
  sumEnergy->addLocalEnergy(localEnergy_nonconforming);
  sumEnergy->addLocalEnergy(forceEnergy);

  auto localGFEADOLCStiffness_nonconforming= std::make_shared<Dune::GFE::LocalGeodesicFEADOLCStiffness<CoefficientBasis, Coefficient> >(sumEnergy);
  std::shared_ptr<Dune::GFE::GeodesicFEAssembler<CoefficientBasis, Coefficient> > assembler_nonconforming;
  assembler_nonconforming = std::make_shared<Dune::GFE::GeodesicFEAssembler<CoefficientBasis, Coefficient> >(coefficientBasis, localGFEADOLCStiffness_nonconforming);

  // Create a solver:
  // * Riemannian Newton with Hessian modification
  // * Riemannian Trust-region
  Dune::GFE::RiemannianProximalNewtonSolver<CoefficientBasis, Coefficient> RNHMsolver;
  Dune::GFE::RiemannianTrustRegionSolver<CoefficientBasis, Coefficient> RTRsolver;

  std::string Solver_name = parameterSet.get<std::string>("Solver", "RNHM");
  double numerical_energy; //final discrete energy

  if(Solver_name == "RNHM")
  {
    RNHMsolver.setup(*grid,
                     &(*assembler_nonconforming),
                     isometryCoefficients,
                     dirichletNodes,
                     parameterSet);

    RNHMsolver.solve();
    isometryCoefficients = RNHMsolver.getSol();
    numerical_energy = RNHMsolver.getStatistics().finalEnergy;
  } else if (Solver_name =="RiemannianTR")
  {
    std::cout << "Using Riemannian Trust-region method for energy minimization." << std::endl;
    RTRsolver.setup(*grid,
                    &(*assembler_nonconforming),
                    isometryCoefficients,
                    dirichletNodes,
                    parameterSet);

    RTRsolver.solve();
    isometryCoefficients = RTRsolver.getSol();
    numerical_energy = RTRsolver.getStatistics().finalEnergy;
  } else
    DUNE_THROW(Dune::Exception, "Unknown Solver type for bending isometries.");


  /** Convert coefficient data structure from 'IsometryCoefficients' back to 'VectorSpaceCoefficients'  */
  VectorSpaceCoefficients x_out(deformationBasis.size());
  isometryToVectorCoefficientMap(deformationBasis,coefficientBasis,x_out,isometryCoefficients);

  ////////////////////////////////
  //   Output result
  ////////////////////////////////
  std::string baseNameDefault = "bending-isometries-";
  std::string baseName = parameterSet.get("baseName", baseNameDefault);
  std::string resultFileName =  parameterSet.get("resultPath", "")
                               + "/" + baseName
                               + "-level" + std::to_string(parameterSet.get<int>("macroGridLevel"));


  if (parameterSet.get<bool>("writeVTK", 1))
  {
    std::cout << "write VTK to Filename: " << resultFileName << std::endl;

    // Compute the displacement from the deformation.
    // interpolate the identity and subtract from the coefficient vector.
    VectorSpaceCoefficients identity(deformationBasis.size());
    IdentityGridEmbedding<double> identityGridEmbedding;
    interpolate(deformationBasis, identity, identityGridEmbedding);

    // Compute the displacement
    auto displacement = x_out;
    displacement -= identity;

    auto deformationFunction = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(deformationBasis, x_out);
    auto displacementFunction = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(deformationBasis, displacement);

    // Use VTK writer from dune-vtk.
    // Setup a DataCollector of order 3.
    Dune::Vtk::LagrangeDataCollector<GridView, 3> lagrangeDataCollector(gridView);
    Dune::Vtk::UnstructuredGridWriter duneVTKwriter(lagrangeDataCollector, Dune::Vtk::FormatTypes::ASCII, Vtk::DataTypes::FLOAT32);

    // Write discrete displacement
    duneVTKwriter.addPointData(displacementFunction, Dune::Vtk::FieldInfo{"Displacement",3, Vtk::RangeTypes::VECTOR});

    duneVTKwriter.write(resultFileName);
  }

  std::cout << "Total time elapsed: " << globalTimer.elapsed() << std::endl;
  return 0;
}
