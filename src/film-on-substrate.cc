#define MIXED_SPACE 0
#include <iostream>
#include <fstream>

#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/indices.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>
#include <dune/common/tuplevector.hh>
#include <dune/common/version.hh>

#include <dune/elasticity/materials/exphenckydensity.hh>
#include <dune/elasticity/materials/henckydensity.hh>
#include <dune/elasticity/materials/mooneyrivlindensity.hh>
#include <dune/elasticity/materials/neohookedensity.hh>
#include <dune/elasticity/materials/stvenantkirchhoffdensity.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/localintegralenergy.hh>
#include <dune/gfe/mixedgfeassembler.hh>
#include <dune/gfe/mixedlocalgfeadolcstiffness.hh>
#include <dune/gfe/neumannenergy.hh>
#include <dune/gfe/surfacecosseratenergy.hh>
#include <dune/gfe/sumenergy.hh>
#include <dune/gfe/vertexnormals.hh>

#if MIXED_SPACE
#include <dune/gfe/mixedriemanniantrsolver.hh>
#else
#include <dune/gfe/geodesicfeassemblerwrapper.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/rigidbodymotion.hh>
#endif

#include <dune/istl/multitypeblockvector.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

// grid dimension
#ifndef WORLD_DIM
#  define WORLD_DIM 3
#endif
const int dim = WORLD_DIM;

const int targetDim = WORLD_DIM;

const int displacementOrder = 2;
const int rotationOrder = 2;

#if !MIXED_SPACE
static_assert(displacementOrder==rotationOrder, "displacement and rotation order do not match!");
#endif

#if DUNE_VERSION_LT(DUNE_COMMON, 2, 7)
template<>
struct Dune::MathematicalConstants<adouble>
{
  static const adouble pi ()
  {
    using std::acos;
    static const adouble pi = acos( adouble( -1 ) );
    return pi;
  }
};
#endif

//differentiation method
typedef adouble ValueType;

using namespace Dune;

int main (int argc, char *argv[]) try
{
  Dune::Timer overallTimer;
  // initialize MPI, finalize is done automatically on exit
  Dune::MPIHelper& mpiHelper = MPIHelper::instance(argc, argv);

  if (mpiHelper.rank()==0) {
    std::cout << "DISPLACEMENTORDER = " << displacementOrder << ", ROTATIONORDER = " << rotationOrder << std::endl;
#if MIXED_SPACE
    std::cout << "MIXED_SPACE = 1" << std::endl;
#else
    std::cout << "MIXED_SPACE = 0" << std::endl;
#endif
  }

  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  //feenableexcept(FE_INVALID);
  Python::runStream()
        << std::endl << "import sys"
        << std::endl << "import os"
        << std::endl << "sys.path.append(os.getcwd() + '/../../problems/')"
        << std::endl;

  // parse data file
  ParameterTree parameterSet;

  ParameterTreeParser::readINITree(argv[1], parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read solver settings
  int numLevels                         = parameterSet.get<int>("numLevels");
  int numHomotopySteps                  = parameterSet.get<int>("numHomotopySteps");
  const double tolerance                = parameterSet.get<double>("tolerance");
  const int maxSolverSteps              = parameterSet.get<int>("maxSolverSteps");
  const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
  const double initialRegularization    = parameterSet.get<double>("initialRegularization");
  const int multigridIterations         = parameterSet.get<int>("numIt");
  const int nu1                         = parameterSet.get<int>("nu1");
  const int nu2                         = parameterSet.get<int>("nu2");
  const int mu                          = parameterSet.get<int>("mu");
  const int baseIterations              = parameterSet.get<int>("baseIt");
  const double mgTolerance              = parameterSet.get<double>("mgTolerance");
  const double baseTolerance            = parameterSet.get<double>("baseTolerance");
  const bool instrumented               = parameterSet.get<bool>("instrumented");
  const bool startFromFile              = parameterSet.get<bool>("startFromFile");
  const std::string resultPath          = parameterSet.get("resultPath", "");

  /////////////////////////////////////////////////////////////
  //                      CREATE THE GRID
  /////////////////////////////////////////////////////////////
  typedef UGGrid<dim> GridType;

  std::shared_ptr<GridType> grid;

  FieldVector<double,dim> lower(0), upper(1);


  if (parameterSet.get<bool>("structuredGrid")) {

    lower = parameterSet.get<FieldVector<double,dim> >("lower");
    upper = parameterSet.get<FieldVector<double,dim> >("upper");

    std::array<unsigned int,dim> elements = parameterSet.get<std::array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

  } else {
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");
    grid = std::shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->setRefinementType(GridType::RefinementType::COPY);

  // Make Python function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  PythonFunction<FieldVector<double,dim>, FieldVector<bool,dim>> pythonDirichletVertices(Python::evaluate(lambda));

  // Same for the Neumann boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
  PythonFunction<FieldVector<double,dim>, bool> pythonNeumannVertices(Python::evaluate(lambda));

  // Same for the Surface Shell Boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("surfaceShellVerticesPredicate", "0") + std::string(")");
  PythonFunction<FieldVector<double,dim>, bool> pythonSurfaceShellVertices(Python::evaluate(lambda));

  while (numLevels > 0) {
    for (auto&& e : elements(grid->leafGridView())){
      bool isSurfaceShell = false;
      for (int i = 0; i < e.geometry().corners(); i++) {
          isSurfaceShell = isSurfaceShell || pythonSurfaceShellVertices(e.geometry().corner(i));
      }
      grid->mark(isSurfaceShell ? 1 : 0,e);
    }

    grid->adapt();

    numLevels--;
  }

  grid->loadBalance();

  if (mpiHelper.rank()==0)
    std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  /////////////////////////////////////////////////////////////
  //                      DATA TYPES 
  /////////////////////////////////////////////////////////////

  using namespace Dune::Indices;

  typedef std::vector<RealTuple<double,dim> > DisplacementVector;
  typedef std::vector<Rotation<double,dim>> RotationVector;
  const int dimRotation = Rotation<double,dim>::embeddedDim;
  typedef TupleVector<DisplacementVector, RotationVector> SolutionType;

  /////////////////////////////////////////////////////////////
  //                      FUNCTION SPACE
  /////////////////////////////////////////////////////////////

  using namespace Dune::Functions::BasisFactory;
  auto compositeBasis = makeBasis(
    gridView,
    composite(
      power<dim>(
        lagrange<displacementOrder>()
      ),
      power<dimRotation>(
        lagrange<rotationOrder>()
      )
  ));

  auto deformationPowerBasis = makeBasis(
    gridView,
    power<dim>(
        lagrange<displacementOrder>()
  ));
  typedef Dune::Functions::LagrangeBasis<GridView,displacementOrder> DeformationFEBasis;
  typedef Dune::Functions::LagrangeBasis<GridView,rotationOrder> OrientationFEBasis;
  DeformationFEBasis deformationFEBasis(gridView);
  OrientationFEBasis orientationFEBasis(gridView);

  using CompositeBasis = decltype(compositeBasis);
  using LocalView = typename CompositeBasis::LocalView;

  /////////////////////////////////////////////////////////////
  //                      BOUNDARY DATA
  /////////////////////////////////////////////////////////////

  BitSetVector<1> dirichletVerticesX(gridView.size(dim), false);
  BitSetVector<1> dirichletVerticesY(gridView.size(dim), false);
  BitSetVector<1> dirichletVerticesZ(gridView.size(dim), false);
  BitSetVector<1> neumannVertices(gridView.size(dim), false);
  BitSetVector<1> surfaceShellVertices(gridView.size(dim), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  for (auto&& v : vertices(gridView))
  {
    FieldVector<bool,dim> isDirichlet = pythonDirichletVertices(v.geometry().corner(0));
    dirichletVerticesX[indexSet.index(v)] = isDirichlet[0];
    dirichletVerticesY[indexSet.index(v)] = isDirichlet[1];
    dirichletVerticesZ[indexSet.index(v)] = isDirichlet[2];

    bool isNeumann = pythonNeumannVertices(v.geometry().corner(0));
    neumannVertices[indexSet.index(v)] = isNeumann;

    bool isSurfaceShell = pythonSurfaceShellVertices(v.geometry().corner(0));
    surfaceShellVertices[indexSet.index(v)] = isSurfaceShell;
  }

  BoundaryPatch<GridView> dirichletBoundaryX(gridView, dirichletVerticesX);
  BoundaryPatch<GridView> dirichletBoundaryY(gridView, dirichletVerticesY);
  BoundaryPatch<GridView> dirichletBoundaryZ(gridView, dirichletVerticesZ);
  auto neumannBoundary = std::make_shared<BoundaryPatch<GridView>>(gridView, neumannVertices);
  BoundaryPatch<GridView> surfaceShellBoundary(gridView, surfaceShellVertices);

  std::cout << "On rank " << mpiHelper.rank() << ": Dirichlet boundary has [" << dirichletBoundaryX.numFaces() <<
                                                                         ", " << dirichletBoundaryY.numFaces() <<
                                                                         ", " << dirichletBoundaryZ.numFaces() <<"] faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Neumann boundary has " << neumannBoundary->numFaces() << " faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Shell boundary has " << surfaceShellBoundary.numFaces() << " faces\n";

  BitSetVector<1> dirichletNodesX(compositeBasis.size({0}),false);
  BitSetVector<1> dirichletNodesY(compositeBasis.size({0}),false);
  BitSetVector<1> dirichletNodesZ(compositeBasis.size({0}),false);
  BitSetVector<1> surfaceShellNodes(compositeBasis.size({1}),false);
  constructBoundaryDofs(dirichletBoundaryX,deformationFEBasis,dirichletNodesX);
  constructBoundaryDofs(dirichletBoundaryY,deformationFEBasis,dirichletNodesY);
  constructBoundaryDofs(dirichletBoundaryZ,deformationFEBasis,dirichletNodesZ);
  constructBoundaryDofs(surfaceShellBoundary,orientationFEBasis,surfaceShellNodes);

  //Create BitVector matching the tangential space
  const int dimRotationTangent = Rotation<double,dim>::TangentVector::dimension;
  typedef MultiTypeBlockVector<std::vector<FieldVector<double,dim> >, std::vector<FieldVector<double,dimRotationTangent>> > VectorForBit;
  typedef Solvers::DefaultBitVector_t<VectorForBit> BitVector;

  BitVector dirichletDofs;
  dirichletDofs[_0].resize(compositeBasis.size({0}));
  dirichletDofs[_1].resize(compositeBasis.size({1}));
  for (size_t i = 0; i < compositeBasis.size({0}); i++) {
    dirichletDofs[_0][i][0] = dirichletNodesX[i][0];
    dirichletDofs[_0][i][1] = dirichletNodesY[i][0];
    dirichletDofs[_0][i][2] = dirichletNodesZ[i][0];
  }
  for (size_t i = 0; i < compositeBasis.size({1}); i++) {
    for (int j = 0; j < dimRotationTangent; j++){
      dirichletDofs[_1][i][j] = not surfaceShellNodes[i][0];
    }
  }

  /////////////////////////////////////////////////////////////
  //                      INITIAL DATA
  /////////////////////////////////////////////////////////////
  
  SolutionType x;
  x[_0].resize(compositeBasis.size({0}));
  x[_1].resize(compositeBasis.size({1}));

  //Initial deformation of the underlying substrate
  BlockVector<FieldVector<double,dim> > displacement(compositeBasis.size({0}));
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
  PythonFunction<FieldVector<double,dim>, FieldVector<double,dim> > pythonInitialDeformation(Python::evaluate(lambda));
  Dune::Functions::interpolate(deformationPowerBasis, displacement, pythonInitialDeformation);

  BlockVector<FieldVector<double,dim> > identity(compositeBasis.size({0}));
  Dune::Functions::interpolate(deformationPowerBasis, identity, [](FieldVector<double,dim> x){ return x; });

  for (int i = 0; i < displacement.size(); i++) {
    x[_0][i] = displacement[i]; //Copy over the initial deformation to the deformation part of x
    displacement[i] -= identity[i]; //Subtract identity to get the initial displacement as a function
  }
  auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim>>(deformationPowerBasis, displacement);
  //  We need to subsample, because VTK cannot natively display real second-order functions
  SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(displacementOrder-1));
  vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriter.write(resultPath + "finite-strain_homotopy_" + parameterSet.get<std::string>("energy") + "_0");
  
  /////////////////////////////////////////////////////////////
  //               INITIAL SURFACE SHELL DATA
  /////////////////////////////////////////////////////////////

  typedef MultiLinearGeometry<double, dim-1, dim> ML;
  std::unordered_map<GridType::GlobalIdSet::IdType, ML> geometriesOnShellBoundary;
  
  auto& idSet = grid->globalIdSet();

  // Read in the grid deformation
  if (startFromFile) {
    // Create a basis of order 1 in order to deform the geometries on the surface shell boundary
    const std::string pathToGridDeformationFile = parameterSet.get("pathToGridDeformationFile", "");
    // for this, we create a basis of order 1 in order to deform the geometries on the surface shell boundary
    auto feBasisOrder1 = makeBasis(
      gridView,
      power<dim>(
        lagrange<1>(),
        blockedInterleaved()
    ));
    GlobalIndexSet<GridView> globalVertexIndexSet(gridView,dim);

    std::unordered_map<std::string, FieldVector<double,3>> deformationMap;
    std::string line, displacement, entry;
    if (mpiHelper.rank() == 0) 
      std::cout << "Reading in deformation file: " << pathToGridDeformationFile + parameterSet.get<std::string>("gridDeformationFile") << std::endl;
    // Read grid deformation information from the file specified in the parameter set via gridDeformationFile
    std::ifstream file(pathToGridDeformationFile + parameterSet.get<std::string>("gridDeformationFile"), std::ios::in);
    if (file.is_open()) {
      while (std::getline(file, line)) {
        size_t j = 0;
        size_t pos = line.find(":");
        displacement = line.substr(pos + 1);
        line.erase(pos);
        std::stringstream entries(displacement);
        FieldVector<double,3> displacementVector(0);
        while(entries >> entry)
          displacementVector[j++] = std::stod(entry);
        deformationMap.insert({line,displacementVector});
      }
      if (mpiHelper.rank() == 0)
        std::cout << "... done: The grid has " << globalVertexIndexSet.size(dim) << " vertices and the defomation file has " << deformationMap.size() << " entries." << std::endl;
      if (deformationMap.size() != globalVertexIndexSet.size(dim))
        DUNE_THROW(Exception, "Error: Grid and deformation vector do not match!");
      file.close();
    } else {
      DUNE_THROW(Exception, "Error: Could not open the file containing the deformation vector!");
    }
  
    BlockVector<FieldVector<double,dim>> gridDeformationFromFile;
    Dune::Functions::interpolate(feBasisOrder1, gridDeformationFromFile, [](FieldVector<double,dim> x){ return x; });

    for (auto& entry : gridDeformationFromFile) {
      std::stringstream stream;
      stream << entry;
      entry = deformationMap.at(stream.str()); //Look up the deformation for this vertex in the deformationMap
    }

    auto gridDeformationFromFileFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim>>(feBasisOrder1, gridDeformationFromFile);
    auto localGridDeformationFromFileFunction = localFunction(gridDeformationFromFileFunction);

    //Write out the stress-free geometries that were read in
    SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(0));
    vtkWriter.addVertexData(localGridDeformationFromFileFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
    vtkWriter.write("stress-free-geometries");

    //Iterate over boundary, each facet on the boundary has an element (boundaryElement.inside()) with a unique global id (idSet.subId);
    //we store the new geometry in the map with this id as reference
    for (auto boundaryElement : surfaceShellBoundary) {
      localGridDeformationFromFileFunction.bind(boundaryElement.inside());
      std::vector<Dune::FieldVector<double,dim>> corners;
      for (int i = 0; i < boundaryElement.geometry().corners(); i++) {
        auto corner = boundaryElement.geometry().corner(i);
        corner += localGridDeformationFromFileFunction(boundaryElement.inside().geometry().local(boundaryElement.geometry().corner(i)));
        corners.push_back(corner);
      }
      localGridDeformationFromFileFunction.unbind();
      GridType::GlobalIdSet::IdType id = idSet.subId(boundaryElement.inside(), boundaryElement.indexInInside(), 1);
      ML boundaryGeometry(boundaryElement.geometry().type(), corners);
      geometriesOnShellBoundary.insert({id, boundaryGeometry});
    }
  } else {
    // Read grid deformation from deformation function
    auto gridDeformationLambda = std::string("lambda x: (") + parameterSet.get<std::string>("gridDeformation") + std::string(")");
    PythonFunction<FieldVector<double,dim>, FieldVector<double,dim> > gridDeformation(Python::evaluate(gridDeformationLambda));

    //Iterate over boundary, each facet on the boundary has an element (boundaryElement.inside()) with a unique global id (idSet.subId);
    //we store the new geometry in the map with this id as reference
    for (auto boundaryElement : surfaceShellBoundary) {
      std::vector<Dune::FieldVector<double,dim>> corners;
      for (int i = 0; i < boundaryElement.geometry().corners(); i++) {
        auto corner = gridDeformation(boundaryElement.geometry().corner(i));
        corners.push_back(corner);
      }
      GridType::GlobalIdSet::IdType id = idSet.subId(boundaryElement.inside(), boundaryElement.indexInInside(), 1);
      ML boundaryGeometry(boundaryElement.geometry().type(), corners);
      geometriesOnShellBoundary.insert({id, boundaryGeometry});
    }
  }

  /////////////////////////////////////////////////////////////
  //                      MAIN HOMOTOPY LOOP
  /////////////////////////////////////////////////////////////
  FieldVector<double,dim> neumannValues {0,0,0};
  if (parameterSet.hasKey("neumannValues"))
    neumannValues = parameterSet.get<FieldVector<double,dim> >("neumannValues");
  std::cout << "Neumann values: " << neumannValues << std::endl;

  // Vertex Normals for the 3D-Part
  std::vector<UnitVector<double,dim> > vertexNormals(gridView.size(dim));
  Dune::FieldVector<double,dim> vertexNormalRaw = {0,0,1};
  for (int i = 0; i< vertexNormals.size(); i++) {
    UnitVector vertexNormal(vertexNormalRaw);
    vertexNormals[i] = vertexNormal;
  }
  //Function for the Cosserat material parameters
  const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
  Python::Reference surfaceShellClass = Python::import(materialParameters.get<std::string>("surfaceShellParameters"));
  Python::Callable surfaceShellCallable = surfaceShellClass.get("SurfaceShellParameters");
  Python::Reference pythonObject = surfaceShellCallable();
  PythonFunction<Dune::FieldVector<double, dim>, double> fThickness(pythonObject.get("thickness"));
  PythonFunction<Dune::FieldVector<double, dim>, Dune::FieldVector<double, 2>> fLame(pythonObject.get("lame"));

  for (int i = 0; i < numHomotopySteps; i++)
  {
    double homotopyParameter = (i+1)*(1.0/numHomotopySteps);

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////


      // A constant vector-valued function, for simple Neumann boundary values
    std::shared_ptr<std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<double,dim>)>> neumannFunctionPtr;
    neumannFunctionPtr = std::make_shared<std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<double,dim>)>>([&](FieldVector<double,dim> ) {
      return neumannValues * (-homotopyParameter);
    });

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    if (mpiHelper.rank() == 0)
    {
      std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;
      std::cout << "Material parameters:" << std::endl;
      materialParameters.report();
    }

    std::cout << "Selected energy is: " << parameterSet.get<std::string>("energy") << std::endl;

    // /////////////////////////////////////////////////
    //   Create the energy functional
    // /////////////////////////////////////////////////

    std::shared_ptr<Elasticity::LocalDensity<dim,ValueType>> elasticDensity;
    if (parameterSet.get<std::string>("energy") == "stvenantkirchhoff")
      elasticDensity = std::make_shared<Elasticity::StVenantKirchhoffDensity<dim,ValueType>>(materialParameters);
    if (parameterSet.get<std::string>("energy") == "neohooke")
      elasticDensity = std::make_shared<Elasticity::NeoHookeDensity<dim,ValueType>>(materialParameters);
    if (parameterSet.get<std::string>("energy") == "hencky")
      elasticDensity = std::make_shared<Elasticity::HenckyDensity<dim,ValueType>>(materialParameters);
    if (parameterSet.get<std::string>("energy") == "exphencky")
      elasticDensity = std::make_shared<Elasticity::ExpHenckyDensity<dim,ValueType>>(materialParameters);
    if (parameterSet.get<std::string>("energy") == "mooneyrivlin")
      elasticDensity = std::make_shared<Elasticity::MooneyRivlinDensity<dim,ValueType>>(materialParameters);


    if(!elasticDensity)
      DUNE_THROW(Exception, "Error: Selected energy not available!");

    auto elasticEnergy = std::make_shared<GFE::LocalIntegralEnergy<CompositeBasis, RealTuple<ValueType,targetDim>, Rotation<ValueType,dim>>>(elasticDensity);
    auto neumannEnergy = std::make_shared<GFE::NeumannEnergy<CompositeBasis, RealTuple<ValueType,targetDim>, Rotation<ValueType,dim>>>(neumannBoundary,*neumannFunctionPtr);
    auto surfaceCosseratEnergy = std::make_shared<GFE::SurfaceCosseratEnergy<CompositeBasis, RealTuple<ValueType,dim>, Rotation<ValueType,dim> >>(materialParameters, std::move(vertexNormals), &surfaceShellBoundary, std::move(geometriesOnShellBoundary), fThickness, fLame);

    GFE::SumEnergy<CompositeBasis, RealTuple<ValueType,targetDim>, Rotation<ValueType,targetDim>> sumEnergy;
    sumEnergy.addLocalEnergy(neumannEnergy);
    sumEnergy.addLocalEnergy(elasticEnergy);
    sumEnergy.addLocalEnergy(surfaceCosseratEnergy);

    MixedLocalGFEADOLCStiffness<CompositeBasis,
                                RealTuple<double,dim>,
                                Rotation<double,dim> > localGFEADOLCStiffness(&sumEnergy);
    MixedGFEAssembler<CompositeBasis,
                      RealTuple<double,dim>,
                      Rotation<double,dim> > mixedAssembler(compositeBasis, &localGFEADOLCStiffness);

    ////////////////////////////////////////////////////////
    //   Set Dirichlet values
    ////////////////////////////////////////////////////////

    BitSetVector<RealTuple<double,dim>::TangentVector::dimension> deformationDirichletDofs(dirichletDofs[_0].size(), false);
    for(int i = 0; i < dirichletDofs[_0].size(); i++)
      for (int j = 0; j < RealTuple<double,dim>::TangentVector::dimension; j++)
        deformationDirichletDofs[i][j] = dirichletDofs[_0][i][j];
    BitSetVector<Rotation<double,dim>::TangentVector::dimension> orientationDirichletDofs(dirichletDofs[_1].size(), false);
    for(int i = 0; i < dirichletDofs[_1].size(); i++)
      for (int j = 0; j < Rotation<double,dim>::TangentVector::dimension; j++)
        orientationDirichletDofs[i][j] = dirichletDofs[_1][i][j];

    Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("dirichletValues"));
    Python::Callable C = dirichletValuesClass.get("DirichletValues");

    // Call a constructor.
    Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

    // Extract object member functions as Dune functions
    PythonFunction<FieldVector<double,dim>, FieldVector<double,targetDim> >   deformationDirichletValues(dirichletValuesPythonObject.get("deformation"));
    PythonFunction<FieldVector<double,dim>, FieldMatrix<double,targetDim,targetDim> > rotationalDirichletValues(dirichletValuesPythonObject.get("orientation"));

    BlockVector<FieldVector<double,targetDim> > ddV;
    Dune::Functions::interpolate(deformationPowerBasis, ddV, deformationDirichletValues, deformationDirichletDofs);

    BlockVector<FieldMatrix<double,targetDim,targetDim> > dOV;
    Dune::Functions::interpolate(orientationFEBasis, dOV, rotationalDirichletValues, orientationDirichletDofs);

    for (int i = 0; i < compositeBasis.size({0}); i++)
      if (dirichletDofs[_0][i][0])
        x[_0][i] = ddV[i];
    for (int i = 0; i < compositeBasis.size({1}); i++)
      if (dirichletDofs[_1][i][0])
        x[_1][i].set(dOV[i]);

#if !MIXED_SPACE
    //The MixedRiemannianTrustRegionSolver can treat the Displacement and Orientation Space as separate ones
    //The RiemannianTrustRegionSolver can only treat the Displacement and Rotation together in a RigidBodyMotion
    //Therefore, x and the dirichletDofs are converted to a RigidBodyMotion structure, as well as the Hessian and Gradient that are returned by the assembler
    typedef RigidBodyMotion<double, dim> RBM;
    std::vector<RBM> xRBM(compositeBasis.size({0}));
    BitSetVector<RBM::TangentVector::dimension> dirichletDofsRBM(compositeBasis.size({0}), false);
    for (int i = 0; i < compositeBasis.size({0}); i++) {
      for (int j = 0; j < dim; j ++) { // Displacement part
        xRBM[i].r[j] = x[_0][i][j];
        dirichletDofsRBM[i][j] = dirichletDofs[_0][i][j];
      }
      xRBM[i].q = x[_1][i]; // Rotation part
      for (int j = dim; j < RBM::TangentVector::dimension; j ++)
        dirichletDofsRBM[i][j] = dirichletDofs[_1][i][j-dim];
    }
    typedef Dune::GFE::GeodesicFEAssemblerWrapper<CompositeBasis, DeformationFEBasis, RBM, RealTuple<double, dim>, Rotation<double,dim>> GFEAssemblerWrapper;
    GFEAssemblerWrapper assembler(&mixedAssembler, deformationFEBasis);
#endif

    ///////////////////////////////////////////////////
    //   Create the chosen solver and solve!
    ///////////////////////////////////////////////////

    if (parameterSet.get<std::string>("solvertype") == "multigrid") {
#if MIXED_SPACE
      MixedRiemannianTrustRegionSolver<GridType, CompositeBasis, DeformationFEBasis, RealTuple<double,dim>, OrientationFEBasis, Rotation<double,dim>> solver;
      solver.setup(*grid,
                   &mixedAssembler,
                   deformationFEBasis,
                   orientationFEBasis,
                   x,
                   deformationDirichletDofs,
                   orientationDirichletDofs,
                   tolerance,
                   maxSolverSteps,
                   initialTrustRegionRadius,
                   multigridIterations,
                   mgTolerance,
                   mu, nu1, nu2,
                   baseIterations,
                   baseTolerance,
                   instrumented);

      solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));
      solver.setInitialIterate(x);
      solver.solve();
      x = solver.getSol();
#else
      RiemannianTrustRegionSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
      solver.setup(*grid,
                   &assembler,
                   xRBM,
                   dirichletDofsRBM,
                   tolerance,
                   maxSolverSteps,
                   initialTrustRegionRadius,
                   multigridIterations,
                   mgTolerance,
                   mu, nu1, nu2,
                   baseIterations,
                   baseTolerance,
                   instrumented);

      solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));
      solver.setInitialIterate(xRBM);
      solver.solve();
      xRBM = solver.getSol();
      for (int i = 0; i < xRBM.size(); i++) {
        x[_0][i] = xRBM[i].r;
        x[_1][i] = xRBM[i].q;
      }
#endif
    } else { //parameterSet.get<std::string>("solvertype") == "cholmod"

#if MIXED_SPACE
    DUNE_THROW(Exception, "Error: There is no MixedRiemannianProximalNewtonSolver!");
#else
      RiemannianProximalNewtonSolver<DeformationFEBasis, RBM, GFEAssemblerWrapper> solver;
      solver.setup(*grid,
                   &assembler,
                   xRBM,
                   dirichletDofsRBM,
                   tolerance,
                   maxSolverSteps,
                   initialRegularization,
                   instrumented);
      solver.setInitialIterate(xRBM);
      solver.solve();
      xRBM = solver.getSol();
      for (int i = 0; i < xRBM.size(); i++) {
        x[_0][i] = xRBM[i].r;
        x[_1][i] = xRBM[i].q;
      }
#endif
    }

    std::cout << "Overall calculation took " << overallTimer.elapsed() << " sec." << std::endl;

    /////////////////////////////////
    //   Output result
    /////////////////////////////////

    // Compute the displacement
    BlockVector<FieldVector<double,dim> > displacement(compositeBasis.size({0}));
    for (int i = 0; i < compositeBasis.size({0}); i++) {
       for (int j = 0; j  < dim; j++) {
        displacement[i][j] = x[_0][i][j];
      }
      displacement[i] -= identity[i];
    }
    auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim>>(deformationPowerBasis, displacement);

    //  We need to subsample, because VTK cannot natively display real second-order functions
    SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(displacementOrder-1));
    vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
    vtkWriter.write(resultPath + "finite-strain_homotopy_" + parameterSet.get<std::string>("energy") + "_" + std::to_string(neumannValues[0]) + "_" + std::to_string(i+1));
  }
} catch (Exception& e) {
    std::cout << e.what() << std::endl;
}
