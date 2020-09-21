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
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>
#include <dune/common/version.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#if DUNE_VERSION_GTE(DUNE_ELASTICITY, 2, 8)
#include <dune/elasticity/materials/exphenckydensity.hh>
#include <dune/elasticity/materials/henckydensity.hh>
#include <dune/elasticity/materials/mooneyrivlindensity.hh>
#include <dune/elasticity/materials/neohookedensity.hh>
#include <dune/elasticity/materials/stvenantkirchhoffdensity.hh>
#include <dune/elasticity/materials/sumenergy.hh>
#include <dune/elasticity/materials/localintegralenergy.hh>
#include <dune/elasticity/materials/neumannenergy.hh>
#else
#include <dune/elasticity/materials/exphenckyenergy.hh>
#include <dune/elasticity/materials/henckyenergy.hh>
#if !DUNE_VERSION_LT(DUNE_ELASTICITY, 2, 7)
#include <dune/elasticity/materials/mooneyrivlinenergy.hh>
#endif
#include <dune/elasticity/materials/neohookeenergy.hh>
#include <dune/elasticity/materials/neumannenergy.hh>
#include <dune/elasticity/materials/sumenergy.hh>
#include <dune/elasticity/materials/stvenantkirchhoffenergy.hh>
#endif


#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemannianpnsolver.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/vertexnormals.hh>
#include <dune/gfe/surfacecosseratenergy.hh>
#include <dune/gfe/sumcosseratenergy.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

// grid dimension
#ifndef WORLD_DIM
#  define WORLD_DIM 3
#endif
const int dim = WORLD_DIM;
const int order = 2;

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

  if (mpiHelper.rank()==0)
    std::cout << "ORDER = " << order << std::endl;

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

  typedef BlockVector<FieldVector<double,dim> > SolutionType;
  typedef RigidBodyMotion<double,dim> TargetSpace;
  typedef std::vector<TargetSpace> SolutionTypeCosserat;

  const int blocksize = TargetSpace::TangentVector::dimension;

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

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
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
  PythonFunction<FieldVector<double,dim>, bool> pythonDirichletVertices(Python::evaluate(lambda));

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

  // FE basis spanning the FE space that we are working in
  typedef Dune::Functions::LagrangeBasis<GridView, order> FEBasis;
  FEBasis feBasis(gridView);

  // dune-fufem-style FE basis for the transition from dune-fufem to dune-functions
  typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
  FufemFEBasis fufemFEBasis(feBasis);

  // /////////////////////////////////////////
  //   Read Dirichlet values
  // /////////////////////////////////////////

  BitSetVector<1> dirichletVertices(gridView.size(dim), false);
  BitSetVector<1> neumannVertices(gridView.size(dim), false);
  BitSetVector<1> surfaceShellVertices(gridView.size(dim), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  for (auto&& v : vertices(gridView))
  {
    bool isDirichlet = pythonDirichletVertices(v.geometry().corner(0));
    dirichletVertices[indexSet.index(v)] = isDirichlet;

    bool isNeumann = pythonNeumannVertices(v.geometry().corner(0));
    neumannVertices[indexSet.index(v)] = isNeumann;

    bool isSurfaceShell = pythonSurfaceShellVertices(v.geometry().corner(0));
    surfaceShellVertices[indexSet.index(v)] = isSurfaceShell;
  }

  BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
  auto neumannBoundary = std::make_shared<BoundaryPatch<GridView>>(gridView, neumannVertices);
  BoundaryPatch<GridView> surfaceShellBoundary(gridView, surfaceShellVertices);

  std::cout << "On rank " << mpiHelper.rank() << ": Neumann boundary has " << neumannBoundary->numFaces() << " faces\n";
  std::cout << "On rank " << mpiHelper.rank() << ": Shell boundary has " << surfaceShellBoundary.numFaces() << " faces\n";


  BitSetVector<1> dirichletNodes(feBasis.size(), false);
#if DUNE_VERSION_LT(DUNE_FUFEM, 2, 7)
  using FufemFEBasis = DuneFunctionsBasis<FEBasis>;
  FufemFEBasis fufemFeBasis(feBasis);
  constructBoundaryDofs(dirichletBoundary,fufemFeBasis,dirichletNodes);
#else
  constructBoundaryDofs(dirichletBoundary,feBasis,dirichletNodes);
#endif

  BitSetVector<1> neumannNodes(feBasis.size(), false);
#if DUNE_VERSION_LT(DUNE_FUFEM, 2, 7)
  constructBoundaryDofs(*neumannBoundary,fufemFeBasis,neumannNodes);
#else
  constructBoundaryDofs(*neumannBoundary,feBasis,neumannNodes);
#endif

  BitSetVector<1> surfaceShellNodes(feBasis.size(), false);
#if DUNE_VERSION_LT(DUNE_FUFEM, 2, 7)
  constructBoundaryDofs(surfaceShellBoundary,fufemFeBasis,surfaceShellNodes);
#else
  constructBoundaryDofs(surfaceShellBoundary,feBasis,surfaceShellNodes);
#endif

  BitSetVector<blocksize> dirichletDofs(feBasis.size(), false);
  for (size_t i=0; i<feBasis.size(); i++)
    for (int j=0; j<blocksize; j++) {
      if (dirichletNodes[i][0])
        dirichletDofs[i][j] = true;
    }
  for (size_t i=0; i<feBasis.size(); i++)
    for (int j=3; j<blocksize; j++) {
      if (not surfaceShellNodes[i][0])
        dirichletDofs[i][j] = true;
    }

  // //////////////////////////
  //   Initial iterate
  // //////////////////////////
  
  SolutionTypeCosserat x(feBasis.size());

  BlockVector<FieldVector<double,3> > v;
  
  //Initial deformation of the underlying substrate
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
  PythonFunction<FieldVector<double,dim>, FieldVector<double,dim> > pythonInitialDeformation(Python::evaluate(lambda));
  ::Functions::interpolate(fufemFEBasis, v, pythonInitialDeformation);

  //Copy over the initial deformation
  for (size_t i=0; i<x.size(); i++) {
      x[i].r = v[i];
  }

  ////////////////////////////////////////////////////////
  //   Main homotopy loop
  ////////////////////////////////////////////////////////

  using namespace Dune::Functions::BasisFactory;
  auto powerBasis = makeBasis(
    gridView,
    power<dim>(
      lagrange<order>(),
      blockedInterleaved()
  ));

  BlockVector<FieldVector<double,dim>> identity;
  Dune::Functions::interpolate(powerBasis, identity, [](FieldVector<double,dim> x){ return x; });

  BlockVector<FieldVector<double,dim> > displacement(v.size());
  for (int i = 0; i < v.size(); i++) {
    displacement[i] = v[i];
  }
  displacement -= identity;

  auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim>>(feBasis, displacement);
  auto localDisplacementFunction = localFunction(displacementFunction);

  FieldVector<double,dim> neumannValues {0,0,0};

  if (parameterSet.hasKey("neumannValues"))
    neumannValues = parameterSet.get<FieldVector<double,dim> >("neumannValues");
  std::cout << "Neumann values: " << neumannValues << std::endl;

  //  We need to subsample, because VTK cannot natively display real second-order functions
  SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(order-1));
  vtkWriter.addVertexData(localDisplacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriter.write(resultPath + "finite-strain_homotopy_" + parameterSet.get<std::string>("energy") + "_0");

  typedef MultiLinearGeometry<double, dim-1, dim> ML;
  std::unordered_map<GridType::GlobalIdSet::IdType, ML> geometriesOnShellBoundary;
  
  auto& idSet = grid->globalIdSet();

  // Read in the grid deformation
  if (startFromFile) {
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

  const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
  Python::Reference surfaceShellClass = Python::import(materialParameters.get<std::string>("surfaceShellParameters"));
  Python::Callable surfaceShellCallable = surfaceShellClass.get("SurfaceShellParameters");

  Python::Reference pythonObject = surfaceShellCallable();
  PythonFunction<Dune::FieldVector<double, dim>, double> fThickness(pythonObject.get("thickness"));
  PythonFunction<Dune::FieldVector<double, dim>, Dune::FieldVector<double, 2>> fLame(pythonObject.get("lame"));

  for (int i=0; i<numHomotopySteps; i++)
  {
    double homotopyParameter = (i+1)*(1.0/numHomotopySteps);

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////


#if DUNE_VERSION_LT(DUNE_ELASTICITY, 2, 8)
    std::shared_ptr<NeumannFunction> neumannFunction;
    neumannFunction = std::make_shared<NeumannFunction>(neumannValues, homotopyParameter);
#else
      // A constant vector-valued function, for simple Neumann boundary values
    std::shared_ptr<std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<double,dim>)>> neumannFunctionPtr;
    neumannFunctionPtr = std::make_shared<std::function<Dune::FieldVector<double,dim>(Dune::FieldVector<double,dim>)>>([&](FieldVector<double,dim> ) {
      return neumannValues * (-homotopyParameter);
    });
#endif

    if (mpiHelper.rank() == 0)
    {
      std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;
      std::cout << "Material parameters:" << std::endl;
      materialParameters.report();
    }

    // Assembler using ADOL-C
    std::cout << "Selected energy is: " << parameterSet.get<std::string>("energy") << std::endl;
#if DUNE_VERSION_GTE(DUNE_ELASTICITY, 2,8)
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

    auto elasticEnergy = std::make_shared<LocalIntegralEnergy<GridView, FEBasis::LocalView::Tree::FiniteElement, ValueType>>(elasticDensity);
    auto neumannEnergy = std::make_shared<NeumannEnergy<GridView,
                                                        FEBasis::LocalView::Tree::FiniteElement,
                                                        ValueType> >(neumannBoundary,neumannFunctionPtr);

    auto elasticAndNeumann = std::make_shared<Dune::SumEnergy<GridView,
          FEBasis::LocalView::Tree::FiniteElement,
          ValueType>>(elasticEnergy, neumannEnergy);
#else
#if DUNE_VERSION_LT(DUNE_ELASTICITY, 2, 7)
    std::shared_ptr<LocalFEStiffness<GridView,
#else
    std::shared_ptr<Elasticity::LocalEnergy<GridView,
#endif
                                     FEBasis::LocalView::Tree::FiniteElement,
                                     std::vector<Dune::FieldVector<ValueType, dim>> > > elasticEnergy;

    if (parameterSet.get<std::string>("energy") == "stvenantkirchhoff")
      elasticEnergy = std::make_shared<StVenantKirchhoffEnergy<GridView,
                                                               FEBasis::LocalView::Tree::FiniteElement,
                                                               ValueType> >(materialParameters);
#if !DUNE_VERSION_LT(DUNE_ELASTICITY, 2, 7)
    if (parameterSet.get<std::string>("energy") == "mooneyrivlin")
      elasticEnergy = std::make_shared<MooneyRivlinEnergy<GridView,
                                                               FEBasis::LocalView::Tree::FiniteElement,
                                                               ValueType> >(materialParameters);
#endif

    if (parameterSet.get<std::string>("energy") == "neohooke")
      elasticEnergy = std::make_shared<NeoHookeEnergy<GridView,
                                                      FEBasis::LocalView::Tree::FiniteElement,
                                                      ValueType> >(materialParameters);

    if (parameterSet.get<std::string>("energy") == "hencky")
      elasticEnergy = std::make_shared<HenckyEnergy<GridView,
                                                    FEBasis::LocalView::Tree::FiniteElement,
                                                    ValueType> >(materialParameters);

    if (parameterSet.get<std::string>("energy") == "exphencky")
      elasticEnergy = std::make_shared<ExpHenckyEnergy<GridView,
                                                       FEBasis::LocalView::Tree::FiniteElement,
                                                       ValueType> >(materialParameters);

    if(!elasticEnergy)
      DUNE_THROW(Exception, "Error: Selected energy not available!");

    auto neumannEnergy = std::make_shared<NeumannEnergy<GridView,
                                                        FEBasis::LocalView::Tree::FiniteElement,
                                                        ValueType> >(neumannBoundary.get(),neumannFunction.get());

    auto elasticAndNeumann = std::make_shared<SumEnergy<GridView,
          FEBasis::LocalView::Tree::FiniteElement,
          ValueType>>(elasticEnergy, neumannEnergy);
#endif

    using LocalEnergyBase = GFE::LocalEnergy<FEBasis,RigidBodyMotion<adouble, dim> >;

    std::shared_ptr<LocalEnergyBase> surfaceCosseratEnergy;
    std::vector<UnitVector<double,3> > vertexNormals(gridView.size(3));
    Dune::FieldVector<double,3> vertexNormalRaw = {0,0,1};
    for (int i = 0; i< vertexNormals.size(); i++) {
      UnitVector vertexNormal(vertexNormalRaw);
      vertexNormals[i] = vertexNormal;
    }

    surfaceCosseratEnergy = std::make_shared<SurfaceCosseratEnergy<FEBasis,RigidBodyMotion<adouble, dim>, adouble, adouble>>(materialParameters, std::move(vertexNormals), &surfaceShellBoundary, std::move(geometriesOnShellBoundary), fThickness, fLame);

    std::shared_ptr<LocalEnergyBase> totalEnergy;
    totalEnergy = std::make_shared<GFE::SumCosseratEnergy<FEBasis,RigidBodyMotion<adouble, dim>, adouble>> (elasticAndNeumann, surfaceCosseratEnergy);


    LocalGeodesicFEADOLCStiffness<FEBasis,
                                  TargetSpace> localGFEADOLCStiffness(totalEnergy.get());

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(gridView, &localGFEADOLCStiffness);


    ////////////////////////////////////////////////////////
    //   Set Dirichlet values
    ////////////////////////////////////////////////////////

    Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("dirichletValues"));

    Python::Callable C = dirichletValuesClass.get("DirichletValues");

    // Call a constructor.
    Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

    // Extract object member functions as Dune functions
    PythonFunction<FieldVector<double,dim>, FieldVector<double,3> >   deformationDirichletValues(dirichletValuesPythonObject.get("deformation"));
    PythonFunction<FieldVector<double,dim>, FieldMatrix<double,3,3> > orientationDirichletValues(dirichletValuesPythonObject.get("orientation"));

    std::vector<FieldVector<double,3> > ddV;
    ::Functions::interpolate(fufemFEBasis, ddV, deformationDirichletValues, dirichletDofs);

    std::vector<FieldMatrix<double,3,3> > dOV;
    ::Functions::interpolate(fufemFEBasis, dOV, orientationDirichletValues, dirichletDofs);

    for (size_t j=0; j<x.size(); j++)
      if (dirichletNodes[j][0])
      {
        x[j].r = ddV[j];
        x[j].q.set(dOV[j]);
      }
    // /////////////////////////////////////////////////
    //   Create the solver and solve
    // /////////////////////////////////////////////////

    if (parameterSet.get<std::string>("solvertype") == "multigrid") {
      RiemannianTrustRegionSolver<FEBasis,TargetSpace> solver;
      solver.setup(*grid,
                   &assembler,
                   x,
                   dirichletDofs,
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
    } else { //parameterSet.get<std::string>("solvertype") == "cholmod"
      RiemannianProximalNewtonSolver<FEBasis,TargetSpace> solver;
      solver.setup(*grid,
                   &assembler,
                   x,
                   dirichletDofs,
                   tolerance,
                   maxSolverSteps,
                   initialRegularization,
                   instrumented);
      solver.setInitialIterate(x);
      solver.solve();

      x = solver.getSol();
    }
    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    std::cout << "Overall calculation took " << overallTimer.elapsed() << " sec." << std::endl;

    /////////////////////////////////
    //   Output result
    /////////////////////////////////
    for (size_t i=0; i<x.size(); i++)
      v[i] = x[i].r;

    // Compute the displacement
    BlockVector<FieldVector<double,dim> > displacement(v.size());
    for (int i = 0; i < v.size(); i++) {
      displacement[i] = v[i];
    }
    displacement -= identity;

    auto displacementFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim>>(feBasis, displacement);
    auto localDisplacementFunction = localFunction(displacementFunction);

    //  We need to subsample, because VTK cannot natively display real second-order functions
    SubsamplingVTKWriter<GridView> vtkWriter(gridView, Dune::refinementLevels(order-1));
    vtkWriter.addVertexData(localDisplacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
    vtkWriter.write(resultPath + "finite-strain_homotopy_" + parameterSet.get<std::string>("energy") + "_" + std::to_string(neumannValues[0]) + "_" + std::to_string(i+1));
  }
} catch (Exception& e) {
    std::cout << e.what() << std::endl;
}
