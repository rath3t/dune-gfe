#include <config.h>

#include <fenv.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <dune/gfe/adolcnamespaceinjections.hh>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/discretizationerror.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/unitvector.hh>
#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/chiralskyrmionenergy.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/embeddedglobalgfefunction.hh>

// grid dimension
const int dim = 2;

// Image space of the geodesic fe functions
// typedef Rotation<double,2> TargetSpace;
// typedef Rotation<double,3> TargetSpace;
// typedef UnitVector<double,2> TargetSpace;
typedef UnitVector<double,3> TargetSpace;
// typedef UnitVector<double,4> TargetSpace;
// typedef RealTuple<double,1> TargetSpace;

// Tangent vector of the image space
const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;


int main (int argc, char *argv[]) try
{
    //feenableexcept(FE_INVALID);
    // Start Python interpreter
    Python::start();
    Python::Reference main = Python::import("__main__");
    Python::run("import math");

    //feenableexcept(FE_INVALID);
    Python::runStream()
        << std::endl << "import sys"
        << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/src')"
        << std::endl;


    typedef std::vector<TargetSpace> SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc < 2)
      DUNE_THROW(Exception, "Usage: ./harmonicmaps <parameter file>");

    ParameterTreeParser::readINITree(argv[1], parameterSet);

    ParameterTreeParser::readOptions(argc, argv, parameterSet);

    // read solver settings
    const int numLevels                   = parameterSet.get<int>("numLevels");
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
    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

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

    //////////////////////////////////////////////////////////////////////////////////
    //  Construct the scalar function space basis corresponding to the GFE space
    //////////////////////////////////////////////////////////////////////////////////

    typedef P1NodalBasis<typename GridType::LeafGridView,double> FEBasis;
    FEBasis feBasis(grid->leafGridView());

    SolutionType x(feBasis.size());

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid->size(dim));
    allNodes.setAll();
    BoundaryPatch<typename GridType::LeafGridView> dirichletBoundary(grid->leafGridView(), allNodes);

    BitSetVector<blocksize> dirichletNodes;
    constructBoundaryDofs(dirichletBoundary,feBasis,dirichletNodes);

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    // Read initial iterate into a PythonFunction
    typedef VirtualDifferentiableFunction<FieldVector<double, dim>, TargetSpace::CoordinateType> FBase;

    Python::Module module = Python::import(parameterSet.get<std::string>("initialIterate"));
    auto pythonInitialIterate = module.get("fdf").toC<std::shared_ptr<FBase>>();

    std::vector<TargetSpace::CoordinateType> v;
    Functions::interpolate(feBasis, v, *pythonInitialIterate);

    for (size_t i=0; i<x.size(); i++)
      x[i] = v[i];

    // backup for error measurement later
    SolutionType initialIterate = x;

    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    // Assembler using ADOL-C
    typedef TargetSpace::rebind<adouble>::other ATargetSpace;
    std::shared_ptr<LocalGeodesicFEStiffness<GridType::LeafGridView,FEBasis::LocalFiniteElement,ATargetSpace> > localEnergy;

    std::string energy = parameterSet.get<std::string>("energy");
    if (energy == "harmonic")
    {

      localEnergy.reset(new HarmonicEnergyLocalStiffness<GridType::LeafGridView, FEBasis::LocalFiniteElement, ATargetSpace>);

    } else if (energy == "chiral_skyrmion")
    {

      localEnergy.reset(new GFE::ChiralSkyrmionEnergy<GridType::LeafGridView, FEBasis::LocalFiniteElement, adouble>(parameterSet.sub("energyParameters")));

    } else
      DUNE_THROW(Exception, "Unknown energy type '" << energy << "'");

    LocalGeodesicFEADOLCStiffness<GridType::LeafGridView,
                                  FEBasis::LocalFiniteElement,
                                  TargetSpace> localGFEADOLCStiffness(localEnergy.get());

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(grid->leafGridView(), &localGFEADOLCStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

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
                 false);   // instrumented

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    std::cout << "Energy: " << assembler.computeEnergy(x) << std::endl;
    //exit(0);

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    typedef BlockVector<FieldVector<double,3> > EmbeddedVectorType;
    EmbeddedVectorType xEmbedded(x.size());
    for (size_t i=0; i<x.size(); i++) {
        xEmbedded[i][0] = x[i].globalCoordinates()[0];
        xEmbedded[i][1] = x[i].globalCoordinates()[1];
        xEmbedded[i][2] = x[i].globalCoordinates()[2];
    }

    VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView());
    Dune::shared_ptr<VTKBasisGridFunction<FEBasis,EmbeddedVectorType> > vtkVectorField
        = Dune::shared_ptr<VTKBasisGridFunction<FEBasis,EmbeddedVectorType> >
               (new VTKBasisGridFunction<FEBasis,EmbeddedVectorType>(feBasis, xEmbedded, "orientation"));
    vtkWriter.addVertexData(vtkVectorField);

    vtkWriter.write(resultPath + "_" + energy + "_result");

    /////////////////////////////////////////////////////////////////
    //   Measure the discretization error, if requested
    /////////////////////////////////////////////////////////////////

    if (parameterSet.get<std::string>("discretizationErrorMode")=="analytical")
    {
      // Read reference solution and its derivative into a PythonFunction
      typedef VirtualDifferentiableFunction<FieldVector<double, dim>, TargetSpace::CoordinateType> FBase;

      Python::Module module = Python::import(parameterSet.get<std::string>("referenceSolution"));
      auto referenceSolution = module.get("fdf").toC<std::shared_ptr<FBase>>();

      // The numerical solution, as a grid function
      GFE::EmbeddedGlobalGFEFunction<FEBasis, TargetSpace> numericalSolution(feBasis, x);

      // QuadratureRule for the integral of the L^2 error
      QuadratureRuleKey quadKey(dim,6);

      // Compute the embedded L^2 error
      double l2Error = DiscretizationError<GridType::LeafGridView>::computeL2Error(&numericalSolution,
                                                                                   referenceSolution.get(),
                                                                                   quadKey);

      // Compute the embedded H^1 error
      double h1Error = DiscretizationError<GridType::LeafGridView>::computeH1HalfNormDifferenceSquared(grid->leafGridView(),
                                                                                                       &numericalSolution,
                                                                                                       referenceSolution.get(),
                                                                                                       quadKey);

      std::cout << "levels: " << numLevels
                << "      "
                << "L^2 error: " << l2Error
                << "      ";
      std::cout << "H^1 error: " << std::sqrt(l2Error*l2Error + h1Error) << std::endl;

    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
