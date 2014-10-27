#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/gfe/adolcnamespaceinjections.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functionspacebases/p2nodalbasis.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/localadolcstiffness.hh>
#include <dune/gfe/stvenantkirchhoffenergy.hh>
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

    void evaluate(const FieldVector<double, dim>& x, FieldVector<double,3>& out) const {
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

    typedef std::vector<FieldVector<double,dim> > SolutionType;

    // parse data file
    ParameterTree parameterSet;
//     if (argc != 2)
//       DUNE_THROW(Exception, "Usage: ./hencky-material <parameter file>");

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
    const bool instrumented               = parameterSet.get<bool>("instrumented");
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

    typedef P1NodalBasis<GridView,double> FEBasis;
    FEBasis feBasis(gridView);

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> dirichletVertices(gridView.size(dim), false);
    BitSetVector<1> neumannVertices(gridView.size(dim), false);

    GridType::Codim<dim>::LeafIterator vIt    = gridView.begin<dim>();
    GridType::Codim<dim>::LeafIterator vEndIt = gridView.end<dim>();

    const GridView::IndexSet& indexSet = gridView.indexSet();

    // Make Python function that computes which vertices are on the Dirichlet boundary,
    // based on the vertex positions.
    std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("dirichletVerticesPredicate") + std::string(")");
    PythonFunction<FieldVector<double,dim>, bool> pythonDirichletVertices(Python::evaluate(lambda));

    // Same for the Neumann boundary
    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("neumannVerticesPredicate", "0") + std::string(")");
    PythonFunction<FieldVector<double,dim>, bool> pythonNeumannVertices(Python::evaluate(lambda));

    for (; vIt!=vEndIt; ++vIt) {

        bool isDirichlet;
        pythonDirichletVertices.evaluate(vIt->geometry().corner(0), isDirichlet);
        dirichletVertices[indexSet.index(*vIt)] = isDirichlet;

        bool isNeumann;
        pythonNeumannVertices.evaluate(vIt->geometry().corner(0), isNeumann);
        neumannVertices[indexSet.index(*vIt)] = isNeumann;

    }

    BoundaryPatch<GridView> dirichletBoundary(gridView, dirichletVertices);
    BoundaryPatch<GridView> neumannBoundary(gridView, neumannVertices);

    if (mpiHelper.rank()==0)
      std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";


    BitSetVector<1> dirichletNodes(feBasis.size(), false);
    constructBoundaryDofs(dirichletBoundary,feBasis,dirichletNodes);

    BitSetVector<1> neumannNodes(feBasis.size(), false);
    constructBoundaryDofs(neumannBoundary,feBasis,neumannNodes);

    BitSetVector<dim> dirichletDofs(feBasis.size(), false);
    for (size_t i=0; i<feBasis.size(); i++)
      if (dirichletNodes[i][0])
        for (int j=0; j<dim; j++)
          dirichletDofs[i][j] = true;

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    SolutionType x(feBasis.size());

    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
    PythonFunction<FieldVector<double,dim>, FieldVector<double,3> > pythonInitialDeformation(Python::evaluate(lambda));

    std::vector<FieldVector<double,3> > v;
    Functions::interpolate(feBasis, v, pythonInitialDeformation);

    for (size_t i=0; i<x.size(); i++)
      x[i] = v[i];

    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("identity") + std::string(")");
    PythonFunction<FieldVector<double,dim>, FieldVector<double,3> > pythonIdentity(Python::evaluate(lambda));

    SolutionType identity;
    Functions::interpolate(feBasis, identity, pythonIdentity);

    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    // Output initial iterate (of homotopy loop)
    VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView());
    BlockVector<FieldVector<double,3> > displacement(x.size());
    for (auto it = grid->leafGridView().template begin<dim>(); it != grid->leafGridView().template end<dim>(); ++it)
    {
      size_t idx = grid->leafGridView().indexSet().index(*it);
      displacement[idx] = x[idx] - it->geometry().corner(0);

      //std::cout << "idx: " << idx << "   coordinate: " << it->geometry().corner(0) << std::endl;
    }

    Dune::shared_ptr<VTKBasisGridFunction<FEBasis,BlockVector<FieldVector<double,3> > > > vtkDisplacement
               = Dune::make_shared<VTKBasisGridFunction<FEBasis,BlockVector<FieldVector<double,3> > > >
                                  (feBasis, displacement, "Displacement");
    vtkWriter.addVertexData(vtkDisplacement);
    vtkWriter.write(resultPath + "hencky_homotopy_0");

    for (int i=0; i<numHomotopySteps; i++) {

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

        if (mpiHelper.rank() == 0) {
            std::cout << "Material parameters:" << std::endl;
            materialParameters.report();
        }

    // Assembler using ADOL-C

    StVenantKirchhoffEnergy<GridView,
                 FEBasis::LocalFiniteElement,
                 adouble> henckyEnergy(materialParameters,
                                         &neumannBoundary,
                                         neumannFunction.get());

    LocalADOLCStiffness<GridView,
                        FEBasis::LocalFiniteElement,
                        SolutionType> localADOLCStiffness(&henckyEnergy);

    FEAssembler<FEBasis,SolutionType> assembler(gridView, &localADOLCStiffness);

    std::vector<FieldVector<double,3> > pointLoads(x.size());
    std::fill(pointLoads.begin(), pointLoads.end(), 0);
    pointLoads[1372] = parameterSet.get<FieldVector<double,3> >("neumannValues");
    pointLoads[1372] *= 0.5;

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    TrustRegionSolver<GridType,SolutionType> solver;
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
                 baseTolerance,
                 pointLoads
                );

    solver.identity_ = identity;

        ////////////////////////////////////////////////////////
        //   Set Dirichlet values
        ////////////////////////////////////////////////////////

        Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("problem") + "-dirichlet-values");

        Python::Callable C = dirichletValuesClass.get("DirichletValues");

        // Call a constructor.
        Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

        // Extract object member functions as Dune functions
        PythonFunction<FieldVector<double,dim>, FieldVector<double,3> >   dirichletValues(dirichletValuesPythonObject.get("dirichletValues"));

        std::vector<FieldVector<double,3> > ddV;
        Functions::interpolate(feBasis, ddV, dirichletValues, dirichletDofs);

        for (size_t j=0; j<x.size(); j++)
          if (dirichletNodes[j][0])
            x[j] = ddV[j];

        // /////////////////////////////////////////////////////
        //   Solve!
        // /////////////////////////////////////////////////////

        solver.setInitialIterate(x);
        solver.solve();

        x = solver.getSol();

        // Output result of each homotopy step
        VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView());
        BlockVector<FieldVector<double,3> > displacement(x.size());
        for (auto it = grid->leafGridView().template begin<dim>(); it != grid->leafGridView().template end<dim>(); ++it)
        {
          size_t idx = grid->leafGridView().indexSet().index(*it);
          displacement[idx] = x[idx] - it->geometry().corner(0);
        }


        Dune::shared_ptr<VTKBasisGridFunction<FEBasis,BlockVector<FieldVector<double,3> > > > vtkDisplacement
               = Dune::make_shared<VTKBasisGridFunction<FEBasis,BlockVector<FieldVector<double,3> > > >
                                  (feBasis, displacement, "Displacement");
        vtkWriter.addVertexData(vtkDisplacement);
        vtkWriter.write(resultPath + "hencky_homotopy_" + std::to_string(i+1));

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    // finally: compute the average deformation of the Neumann boundary
    // That is what we need for the locking tests
    FieldVector<double,3> averageDef(0);
    for (size_t i=0; i<x.size(); i++)
        if (neumannNodes[i][0])
        {
            averageDef += x[i];
            std::cout << "i: " << i << ",  pos: " << x[i] << std::endl;
        }
    averageDef /= neumannNodes.count();

    if (mpiHelper.rank()==0)
    {
      std::cout << "Neumann value = " << parameterSet.get<std::string>("neumannValues") << std::endl;
      std::cout << "Neumann value = " << parameterSet.get<FieldVector<double,dim> >("neumannValues") << "  "
                << ",  average deflection: " << averageDef << std::endl;
    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
