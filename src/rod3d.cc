#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/drivers/drivers.h>
#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <optional>

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/onedgrid.hh>

#include <dune/istl/io.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/energynorm.hh>

#if HAVE_DUNE_VTK
#include <dune/vtk/vtkwriter.hh>
#else
#include <dune/gfe/cosseratvtkwriter.hh>
#endif

#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/rodlocalstiffness.hh>
#include <dune/gfe/rotation.hh>
#include <dune/gfe/riemanniantrsolver.hh>

typedef RigidBodyMotion<double,3> TargetSpace;

const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;

int main (int argc, char *argv[]) try
{
    MPIHelper::instance(argc, argv);

    typedef std::vector<RigidBodyMotion<double,3> > SolutionType;

    // parse data file
    ParameterTree parameterSet;
    if (argc < 2)
      DUNE_THROW(Exception, "Usage: ./rod3d <parameter file>");

    ParameterTreeParser::readINITree(argv[1], parameterSet);

    ParameterTreeParser::readOptions(argc, argv, parameterSet);

    // read solver settings
    const int numLevels        = parameterSet.get<int>("numLevels");
    const double tolerance        = parameterSet.get<double>("tolerance");
    const int maxTrustRegionSteps   = parameterSet.get<int>("maxTrustRegionSteps");
    const double initialTrustRegionRadius = parameterSet.get<double>("initialTrustRegionRadius");
    const int multigridIterations   = parameterSet.get<int>("numIt");
    const int nu1              = parameterSet.get<int>("nu1");
    const int nu2              = parameterSet.get<int>("nu2");
    const int mu               = parameterSet.get<int>("mu");
    const int baseIterations      = parameterSet.get<int>("baseIt");
    const double mgTolerance        = parameterSet.get<double>("mgTolerance");
    const double baseTolerance    = parameterSet.get<double>("baseTolerance");
    const bool instrumented      = parameterSet.get<bool>("instrumented");
    std::string resultPath           = parameterSet.get("resultPath", "");

    // read rod parameter settings
    const double A               = parameterSet.get<double>("A");
    const double J1              = parameterSet.get<double>("J1");
    const double J2              = parameterSet.get<double>("J2");
    const double E               = parameterSet.get<double>("E");
    const double nu              = parameterSet.get<double>("nu");
    const int numRodBaseElements = parameterSet.get<int>("numRodBaseElements");
    
    // ///////////////////////////////////////
    //    Create the grid
    // ///////////////////////////////////////
    typedef OneDGrid GridType;
    GridType grid(numRodBaseElements, 0, 1);

    grid.globalRefine(numLevels-1);

    using GridView = GridType::LeafGridView;
    GridView gridView = grid.leafGridView();

    using FEBasis = Functions::LagrangeBasis<GridView,1>;
    FEBasis feBasis(gridView);

    SolutionType x(feBasis.size());

    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    for (size_t i=0; i<x.size(); i++) {
        x[i].r[0] = 0;
        x[i].r[1] = 0;
        x[i].r[2] = double(i)/(x.size()-1);
        x[i].q    = Rotation<double,3>::identity();
    }

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////
    x.back().r = parameterSet.get<FieldVector<double,3> >("dirichletValue");

    auto axis = parameterSet.get<FieldVector<double,3> >("dirichletAxis");
    double angle = parameterSet.get<double>("dirichletAngle");

    x.back().q = Rotation<double,3>(axis, M_PI*angle/180);

    // backup for error measurement later
    SolutionType initialIterate = x;

    std::cout << "Left boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[0].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[0].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[0].q.director(2) << std::endl;
    std::cout << std::endl;
    std::cout << "Right boundary orientation:" << std::endl;
    std::cout << "director 0:  " << x[x.size()-1].q.director(0) << std::endl;
    std::cout << "director 1:  " << x[x.size()-1].q.director(1) << std::endl;
    std::cout << "director 2:  " << x[x.size()-1].q.director(2) << std::endl;

    BitSetVector<blocksize> dirichletNodes(feBasis.size());
    dirichletNodes.unsetAll();
        
    dirichletNodes[0] = true;
    dirichletNodes.back() = true;
    
    //////////////////////////////////////////////
    //  Create the stress-free configuration
    //////////////////////////////////////////////

    auto localRodEnergy = std::make_shared<RodLocalStiffness<GridView,adouble> >(gridView,
                                                                                 A, J1, J2, E, nu);

    std::vector<RigidBodyMotion<double,3> > referenceConfiguration(gridView.size(1));

    for (const auto vertex : vertices(gridView))
    {
        auto idx = gridView.indexSet().index(vertex);

        referenceConfiguration[idx].r[0] = 0;
        referenceConfiguration[idx].r[1] = 0;
        referenceConfiguration[idx].r[2] = vertex.geometry().corner(0)[0];
        referenceConfiguration[idx].q = Rotation<double,3>::identity();
    }

    localRodEnergy->setReferenceConfiguration(referenceConfiguration);

    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////

    LocalGeodesicFEADOLCStiffness<FEBasis,
                                  TargetSpace> localStiffness(localRodEnergy.get());

    GeodesicFEAssembler<FEBasis,TargetSpace> rodAssembler(gridView, localStiffness);

    RiemannianTrustRegionSolver<FEBasis,RigidBodyMotion<double,3> > rodSolver;

    rodSolver.setup(grid, 
                    &rodAssembler,
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
                    instrumented);

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    std::cout << "Energy: " << rodAssembler.computeEnergy(x) << std::endl;
    
    rodSolver.setInitialIterate(x);
    rodSolver.solve();

    x = rodSolver.getSol();
        
    // //////////////////////////////
    //   Output result
    // //////////////////////////////
#if HAVE_DUNE_VTK
    VtkUnstructuredGridWriter<GridView> vtkWriter(gridView, Vtk::ASCII);

    // Make basis for R^3-valued data
    using namespace Functions::BasisFactory;

    auto worldBasis = makeBasis(
      gridView,
      power<3>(lagrange<1>())
    );

    // The rod displacement field
    BlockVector<FieldVector<double,3> > displacement(worldBasis.size());
    for (std::size_t i=0; i<x.size(); i++)
      displacement[i] = x[i].r;

    std::vector<double> xEmbedding;
    Functions::interpolate(feBasis, xEmbedding, [](FieldVector<double,1> x){ return x; });

    BlockVector<FieldVector<double,3> > gridEmbedding(xEmbedding.size());
    for (std::size_t i=0; i<gridEmbedding.size(); i++)
      gridEmbedding[i] = {xEmbedding[i], 0, 0};

    displacement -= gridEmbedding;

    auto displacementFunction = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(worldBasis, displacement);
    vtkWriter.addPointData(displacementFunction, "displacement", 3);

    // The three director fields
    using FunctionType = decltype(displacementFunction);
    std::array<std::optional<FunctionType>, 3> directorFunction;

    for (int i=0; i<3; i++)
    {
      BlockVector<FieldVector<double, 3> > director(worldBasis.size());
      for (std::size_t j=0; j<x.size(); j++)
        director[j] = x[j].q.director(i);

      directorFunction[i] = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,3> >(worldBasis, std::move(director));
      vtkWriter.addPointData(*directorFunction[i], "director " + std::to_string(i), 3);
    }

    vtkWriter.write(resultPath + "rod3d-result");
#else
    std::cout << "Falling back to legacy file writing.  Get dune-vtk for better results" << std::endl;
    // Fall-back solution for users without dune-vtk
    CosseratVTKWriter<GridType>::write<FEBasis>(feBasis,x, resultPath + "rod3d-result");
#endif

} catch (Exception& e)
{
    std::cout << e.what() << std::endl;
}
