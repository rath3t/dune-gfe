#include <config.h>

#define SECOND_ORDER

#include <fenv.h>

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
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/grid/io/file/gmshreader.hh>

#include <dune/functions/common/tuplevector.hh>
#include <dune/functions/functionspacebases/pqknodalbasis.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>

#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/dunepython.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/mixedlocalgfeadolcstiffness.hh>
#include <dune/gfe/mixedcosseratenergy.hh>
#include <dune/gfe/cosseratvtkwriter.hh>
#include <dune/gfe/mixedgfeassembler.hh>
#include <dune/gfe/mixedriemanniantrsolver.hh>

// grid dimension
const int dim = 2;

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
        out.axpy(homotopyParameter_, values_);
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
        << std::endl << "sys.path.append('/home/sander/dune/dune-gfe/problems/')"
        << std::endl;

    using namespace Dune::TypeTree::Indices;
    typedef Dune::Functions::TupleVector<std::vector<RealTuple<double,3> >,
                                         std::vector<Rotation<double,3> > > SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc < 2)
      DUNE_THROW(Exception, "Usage: ./mixed-cosserat-continuum <parameter file>");

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

    grid->loadBalance();

    if (mpiHelper.rank()==0)
      std::cout << "There are " << grid->leafGridView().comm().size() << " processes" << std::endl;

    typedef GridType::LeafGridView GridView;
    GridView gridView = grid->leafGridView();

    using namespace Dune::Functions::BasisBuilder;

    auto compositeBasis = makeBasis(
      gridView,
      composite(
          pq<2>(),
          pq<1>()
      )
    );

    typedef Dune::Functions::PQkNodalBasis<GridView,2> DeformationFEBasis;
    typedef Dune::Functions::PQkNodalBasis<GridView,1> OrientationFEBasis;

    DeformationFEBasis deformationFEBasis(gridView);
    OrientationFEBasis orientationFEBasis(gridView);

    // Construct fufem-style function space bases to ease the transition to dune-functions
    typedef DuneFunctionsBasis<DeformationFEBasis> FufemDeformationFEBasis;
    FufemDeformationFEBasis fufemDeformationFEBasis(deformationFEBasis);

    typedef DuneFunctionsBasis<OrientationFEBasis> FufemOrientationFEBasis;
    FufemOrientationFEBasis fufemOrientationFEBasis(orientationFEBasis);



    std::cout << "Deformation: " << deformationFEBasis.indexSet().size() << ",   orientation: " << orientationFEBasis.indexSet().size() << std::endl;

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

    BitSetVector<1> neumannNodes(deformationFEBasis.indexSet().size(), false);
    constructBoundaryDofs(neumannBoundary,fufemDeformationFEBasis,neumannNodes);


    if (mpiHelper.rank()==0)
      std::cout << "Neumann boundary has " << neumannBoundary.numFaces() << " faces\n";


    BitSetVector<1> deformationDirichletNodes(deformationFEBasis.indexSet().size(), false);
    constructBoundaryDofs(dirichletBoundary,fufemDeformationFEBasis,deformationDirichletNodes);

    BitSetVector<3> deformationDirichletDofs(deformationFEBasis.indexSet().size(), false);
    for (size_t i=0; i<deformationFEBasis.indexSet().size(); i++)
      if (deformationDirichletNodes[i][0])
        for (int j=0; j<3; j++)
          deformationDirichletDofs[i][j] = true;

    BitSetVector<1> orientationDirichletNodes(orientationFEBasis.indexSet().size(), false);
    constructBoundaryDofs(dirichletBoundary,fufemOrientationFEBasis,orientationDirichletNodes);

    BitSetVector<3> orientationDirichletDofs(orientationFEBasis.indexSet().size(), false);
    for (size_t i=0; i<orientationFEBasis.indexSet().size(); i++)
      if (orientationDirichletNodes[i][0])
        for (int j=0; j<3; j++)
          orientationDirichletDofs[i][j] = true;

    // //////////////////////////
    //   Initial iterate
    // //////////////////////////

    SolutionType x;

    x[_0].resize(deformationFEBasis.indexSet().size());

    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
    PythonFunction<FieldVector<double,dim>, FieldVector<double,3> > pythonInitialDeformation(Python::evaluate(lambda));

    std::vector<FieldVector<double,3> > v;
    ::Functions::interpolate(fufemDeformationFEBasis, v, pythonInitialDeformation);

    for (size_t i=0; i<x[_0].size(); i++)
      x[_0][i] = v[i];

    x[_1].resize(orientationFEBasis.indexSet().size());
#if 0
    lambda = std::string("lambda x: (") + parameterSet.get<std::string>("initialDeformation") + std::string(")");
    PythonFunction<FieldVector<double,dim>, FieldVector<double,3> > pythonInitialDeformation(Python::evaluate(lambda));

    std::vector<FieldVector<double,3> > v;
    Functions::interpolate(feBasis, v, pythonInitialDeformation);

    for (size_t i=0; i<x.size(); i++)
      xDisp[i] = v[i];
#endif
    ////////////////////////////////////////////////////////
    //   Main homotopy loop
    ////////////////////////////////////////////////////////

    // Output initial iterate (of homotopy loop)
    CosseratVTKWriter<GridType>::writeMixed<FufemDeformationFEBasis,FufemOrientationFEBasis>(fufemDeformationFEBasis,x[_0],
                                                                                             fufemOrientationFEBasis,x[_1],
                                                                                   resultPath + "mixed-cosserat_homotopy_0");

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


        if (mpiHelper.rank() == 0) {
            std::cout << "Material parameters:" << std::endl;
            materialParameters.report();
        }

    // Assembler using ADOL-C
    MixedCosseratEnergy<decltype(compositeBasis),
                        3,adouble> cosseratEnergyADOLCLocalStiffness(materialParameters,
                                                                     &neumannBoundary,
                                                                     neumannFunction.get());

    MixedLocalGFEADOLCStiffness<decltype(compositeBasis),
                                RealTuple<double,3>,
                                Rotation<double,3> > localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

    MixedGFEAssembler<decltype(compositeBasis),
                      RealTuple<double,3>,
                      Rotation<double,3> > assembler(compositeBasis, &localGFEADOLCStiffness);

    // /////////////////////////////////////////////////
    //   Create a Riemannian trust-region solver
    // /////////////////////////////////////////////////

    MixedRiemannianTrustRegionSolver<GridType,
                                     decltype(compositeBasis),
                                     DeformationFEBasis, RealTuple<double,3>,
                                     OrientationFEBasis, Rotation<double,3> > solver;
    solver.setup(*grid,
                 &assembler,
                 deformationFEBasis,
                 orientationFEBasis,
                 x,
                 deformationDirichletDofs,
                 orientationDirichletDofs,
                 tolerance,
                 maxTrustRegionSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 mgTolerance,
                 mu, nu1, nu2,
                 baseIterations,
                 baseTolerance,
                 instrumented);

        solver.setScaling(parameterSet.get<FieldVector<double,6> >("trustRegionScaling"));

        ////////////////////////////////////////////////////////
        //   Set Dirichlet values
        ////////////////////////////////////////////////////////

        Python::Reference dirichletValuesClass = Python::import(parameterSet.get<std::string>("problem") + "-dirichlet-values");

        Python::Callable C = dirichletValuesClass.get("DirichletValues");

        // Call a constructor.
        Python::Reference dirichletValuesPythonObject = C(homotopyParameter);

        // Extract object member functions as Dune functions
        PythonFunction<FieldVector<double,dim>, FieldVector<double,3> >   deformationDirichletValues(dirichletValuesPythonObject.get("deformation"));
        PythonFunction<FieldVector<double,dim>, FieldMatrix<double,3,3> > orientationDirichletValues(dirichletValuesPythonObject.get("orientation"));

        std::vector<FieldVector<double,3> > ddV;
        ::Functions::interpolate(fufemDeformationFEBasis, ddV, deformationDirichletValues, deformationDirichletDofs);

        std::vector<FieldMatrix<double,3,3> > dOV;
        ::Functions::interpolate(fufemOrientationFEBasis, dOV, orientationDirichletValues, orientationDirichletDofs);

        for (size_t j=0; j<x[_0].size(); j++)
          if (deformationDirichletNodes[j][0])
            x[_0][j] = ddV[j];

        for (size_t j=0; j<x[_1].size(); j++)
          if (orientationDirichletNodes[j][0])
            x[_1][j].set(dOV[j]);

        // /////////////////////////////////////////////////////
        //   Solve!
        // /////////////////////////////////////////////////////

        solver.setInitialIterate(x);
        solver.solve();

        x = solver.getSol();

        // Output result of each homotopy step
        std::stringstream iAsAscii;
        iAsAscii << i+1;
        CosseratVTKWriter<GridType>::writeMixed<FufemDeformationFEBasis,FufemOrientationFEBasis>(fufemDeformationFEBasis,x[_0],
                                                                                       fufemOrientationFEBasis,x[_1],
                                                                                       resultPath + "mixed-cosserat_homotopy_" + iAsAscii.str());

    }

    // //////////////////////////////
    //   Output result
    // //////////////////////////////

    // finally: compute the average deformation of the Neumann boundary
    // That is what we need for the locking tests
    FieldVector<double,3> averageDef(0);
    for (size_t i=0; i<x[_0].size(); i++)
        if (neumannNodes[i][0])
            averageDef += x[_0][i].globalCoordinates();
    averageDef /= neumannNodes.count();

    if (mpiHelper.rank()==0)
    {
      std::cout << "Neumann values = " << parameterSet.get<FieldVector<double, 3> >("neumannValues") << "  "
                << ",  average deflection: " << averageDef << std::endl;
    }

 } catch (Exception e) {

    std::cout << e << std::endl;

 }
