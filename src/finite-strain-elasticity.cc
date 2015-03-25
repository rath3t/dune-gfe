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

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/functions/functionspacebases/pq2nodalbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/gridfunctions/discretescalarglobalbasisfunction.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/localadolcstiffness.hh>
#include <dune/gfe/stvenantkirchhoffenergy.hh>
#include <dune/gfe/neumannenergy.hh>
#include <dune/gfe/sumenergy.hh>
#include <dune/gfe/feassembler.hh>
#include <dune/gfe/trustregionsolver.hh>

// grid dimension
const int dim = 3;

using namespace Dune;

/** \brief A constant vector-valued function, for simple Neumann boundary values */
struct NeumannFunction
    : public Dune::VirtualFunction<FieldVector<double,dim>, FieldVector<double,3> >
{
  NeumannFunction(const FieldVector<double,3> values,
                  double homotopyParameter)
  : values_(values),
    homotopyParameter_(homotopyParameter)
  {}

  void evaluate(const FieldVector<double, dim>& x, FieldVector<double,3>& out) const
  {
    out = 0;
    out.axpy(-homotopyParameter_, values_);
  }

  FieldVector<double,3> values_;
  double homotopyParameter_;
};


int main (int argc, char *argv[]) try
{
  // initialize MPI, finalize is done automatically on exit
  Dune::MPIHelper& mpiHelper = MPIHelper::instance(argc, argv);

  // Start Python interpreter
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  //feenableexcept(FE_INVALID);
  Python::runStream()
        << std::endl << "import sys"
        << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/')"
        << std::endl;

  typedef BlockVector<FieldVector<double,dim> > SolutionType;

  // parse data file
  ParameterTree parameterSet;

  ParameterTreeParser::readINITree(argv[1], parameterSet);

  ParameterTreeParser::readOptions(argc, argv, parameterSet);

  // read solver settings
  const int numLevels                   = parameterSet.get<int>("numLevels");
  int numHomotopySteps                  = parameterSet.get<int>("numHomotopySteps");
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

  // ///////////////////////////////////////
  //    Create the grid
  // ///////////////////////////////////////
  typedef UGGrid<dim> GridType;

  shared_ptr<GridType> grid;

  FieldVector<double,dim> lower(0), upper(1);

  if (parameterSet.get<bool>("structuredGrid")) {

    lower = parameterSet.get<FieldVector<double,dim> >("lower");
    upper = parameterSet.get<FieldVector<double,dim> >("upper");

    array<unsigned int,dim> elements = parameterSet.get<array<unsigned int,dim> >("elements");
    grid = StructuredGridFactory<GridType>::createCubeGrid(lower, upper, elements);

  } else {
    std::string path                = parameterSet.get<std::string>("path");
    std::string gridFile            = parameterSet.get<std::string>("gridFile");
    grid = shared_ptr<GridType>(GmshReader<GridType>::read(path + "/" + gridFile));
  }

  grid->globalRefine(numLevels-1);

  grid->loadBalance();

  if (mpiHelper.rank()==0)
    std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  // FE basis spanning the FE space that we are working in
  typedef Dune::Functions::PQ2NodalBasis<GridView> FEBasis;
  FEBasis feBasis(gridView);

  // dune-fufem-style FE basis for the transition from dune-fufem to dune-functions
  typedef DuneFunctionsBasis<FEBasis> FufemFEBasis;
  FufemFEBasis fufemFEBasis(feBasis);

  // /////////////////////////////////////////
  //   Read Dirichlet values
  // /////////////////////////////////////////

  BitSetVector<1> dirichletVertices(gridView.size(dim), false);
  BitSetVector<1> neumannVertices(gridView.size(dim), false);

  const GridView::IndexSet& indexSet = gridView.indexSet();

  // Make Python function that computes which vertices are on the Dirichlet boundary,
  // based on the vertex positions.
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
  PythonFunction<FieldVector<double,dim>, bool> pythonDirichletVertices(Python::evaluate(lambda));

  // Same for the Neumann boundary
  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
  PythonFunction<FieldVector<double,dim>, bool> pythonNeumannVertices(Python::evaluate(lambda));

  for (auto&& v : vertices(gridView))
  {
    bool isDirichlet;
    pythonDirichletVertices.evaluate(v.geometry().corner(0), isDirichlet);
    dirichletVertices[indexSet.index(v)] = isDirichlet;

    bool isNeumann;
    pythonNeumannVertices.evaluate(v.geometry().corner(0), isNeumann);
    neumannVertices[indexSet.index(v)] = isNeumann;
  }

  BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
  BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);

  if (mpiHelper.rank()==0)
    std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";


  BitSetVector<1> dirichletNodes(feBasis.indexSet().size(), false);
  constructBoundaryDofs(dirichletBoundary,fufemFEBasis,dirichletNodes);

  BitSetVector<1> neumannNodes(feBasis.indexSet().size(), false);
  constructBoundaryDofs(neumannBoundary,fufemFEBasis,neumannNodes);

  BitSetVector<dim> dirichletDofs(feBasis.indexSet().size(), false);
  for (size_t i=0; i<feBasis.indexSet().size(); i++)
    if (dirichletNodes[i][0])
      for (int j=0; j<dim; j++)
        dirichletDofs[i][j] = true;

  // //////////////////////////
  //   Initial iterate
  // //////////////////////////

  SolutionType x(feBasis.indexSet().size());

  lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
  PythonFunction<FieldVector<double,dim>, FieldVector<double,3> > pythonInitialDeformation(Python::evaluate(lambda));

  ::Functions::interpolate(fufemFEBasis, x, pythonInitialDeformation);

  ////////////////////////////////////////////////////////
  //   Main homotopy loop
  ////////////////////////////////////////////////////////

  // Output initial iterate (of homotopy loop)
  SolutionType identity;
  Dune::Functions::interpolate(feBasis, identity, [](FieldVector<double,dim> x){ return x; });

  SolutionType displacement = x;
  displacement -= identity;

  Dune::Functions::DiscreteScalarGlobalBasisFunction<FEBasis,SolutionType> displacementFunction(feBasis,displacement);
  auto localDisplacementFunction = localFunction(displacementFunction);

  //  We need to subsample, because VTK cannot natively display real second-order functions
  SubsamplingVTKWriter<GridView> vtkWriter(gridView,2);
  vtkWriter.addVertexData(localDisplacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, 3));
  vtkWriter.write(resultPath + "finite-strain_homotopy_0");

  for (int i=0; i<numHomotopySteps; i++)
  {
    double homotopyParameter = (i+1)*(1.0/numHomotopySteps);
    if (mpiHelper.rank()==0)
      std::cout << "Homotopy step: " << i << ",    parameter: " << homotopyParameter << std::endl;


    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the energy functional
    // ////////////////////////////////////////////////////////////

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    shared_ptr<NeumannFunction> neumannFunction;
    if (parameterSet.hasKey("neumannValues"))
      neumannFunction = make_shared<NeumannFunction>(parameterSet.get<FieldVector<double,3> >("neumannValues"),
                                                     homotopyParameter);

    std::cout << "Neumann values: " << parameterSet.get<FieldVector<double,3> >("neumannValues") << std::endl;

    if (mpiHelper.rank() == 0)
    {
      std::cout << "Material parameters:" << std::endl;
      materialParameters.report();
    }

    // Assembler using ADOL-C
    auto elasticEnergy = std::make_shared<StVenantKirchhoffEnergy<GridView,
                                                                  FEBasis::LocalView::Tree::FiniteElement,
                                                                  adouble> >(materialParameters);

    auto neumannEnergy = std::make_shared<NeumannEnergy<GridView,
                                                        FEBasis::LocalView::Tree::FiniteElement,
                                                        adouble> >(&neumannBoundary,neumannFunction.get());

    SumEnergy<GridView,
              FEBasis::LocalView::Tree::FiniteElement,
              adouble> totalEnergy(elasticEnergy, neumannEnergy);

    LocalADOLCStiffness<GridView,
                        FEBasis::LocalView::Tree::FiniteElement,
                        SolutionType> localADOLCStiffness(&totalEnergy);

    FEAssembler<FufemFEBasis,SolutionType> assembler(fufemFEBasis, &localADOLCStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    TrustRegionSolver<FufemFEBasis,SolutionType> solver;
    solver.setup(*grid,
                 &assembler,
                 x,
                 dirichletDofs,
                 tolerance,
                 maxTrustRegionSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance
                );

    ////////////////////////////////////////////////////////
    //   Set Dirichlet values
    ////////////////////////////////////////////////////////

    // The python class that implements the Dirichlet values
    Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("problem") + "-dirichlet-values");

    // The member method 'DirichletValues' of that class
    Python::Callable C = dirichletValuesClass.get("DirichletValues");

    // Call a constructor.
    Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

    // Extract object member functions as Dune functions
    PythonFunction<FieldVector<double,dim>, FieldVector<double,3> >   dirichletValues(dirichletValuesPythonObject.get("dirichletValues"));

    ::Functions::interpolate(fufemFEBasis, x, dirichletValues, dirichletDofs);

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();

    /////////////////////////////////
    //   Output result
    /////////////////////////////////

    // Compute the displacement
    auto displacement = x;
    displacement -= identity;

    Dune::Functions::DiscreteScalarGlobalBasisFunction<FEBasis,SolutionType> displacementFunction(feBasis,displacement);
    auto localDisplacementFunction = localFunction(displacementFunction);

    //  We need to subsample, because VTK cannot natively display real second-order functions
    SubsamplingVTKWriter<GridView> vtkWriter(gridView,2);
    vtkWriter.addVertexData(localDisplacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, 3));
    vtkWriter.write(resultPath + "finite-strain_homotopy_" + std::to_string(i+1));
  }

} catch (Exception e) {
    std::cout << e << std::endl;
}
