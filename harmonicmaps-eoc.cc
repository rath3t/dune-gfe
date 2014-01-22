#include <config.h>

//#define HARMONIC_ENERGY_FD_GRADIENT
//#define HARMONIC_ENERGY_FD_INNER_GRADIENT
//#define SECOND_ORDER
//#define THIRD_ORDER
const int order = 1;

#include <dune/common/bitsetvector.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/onedgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/io/file/amirameshreader.hh>
#include <dune/grid/io/file/gmshreader.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functionspacebases/p2nodalbasis.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>

#include <dune/solvers/solvers/iterativesolver.hh>
#include <dune/solvers/norms/h1seminorm.hh>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/geodesicfeassembler.hh>
#include <dune/gfe/riemanniantrsolver.hh>
#include <dune/gfe/geodesicfefunctionadaptor.hh>
#include <dune/gfe/gfedifferencenormsquared.hh>

// grid dimension
const int dim = 2;

typedef UnitVector<double,3> TargetSpace;
typedef std::vector<TargetSpace> SolutionType;

const int blocksize = TargetSpace::TangentVector::dimension;

using namespace Dune;
using std::string;

struct DirichletFunction
    : public Dune::VirtualFunction<FieldVector<double,dim>, TargetSpace::CoordinateType >
{
    void evaluate(const FieldVector<double, dim>& x, TargetSpace::CoordinateType& out) const {

#if 0
        FieldVector<double,3> axis;
        axis[0] = x[0];  axis[1] = x[1]; axis[2] = 1;
        Rotation<double,3> rotation(axis, x.two_norm()*M_PI*3);

        FieldMatrix<double,3,3> rMat;
        rotation.matrix(rMat);
        out = rMat[2];
#endif   
#if 1   // This data seems to have the necessary smoothness to have optimal
        // convergence order even for 3rd-order elements
        double localX = (x[0] + 2)/4;
        double localY = (x[1] + 1)/2;
        double angleX = 0.5 * M_PI * sin(M_PI*localX);
        double angleY = 0.5 * M_PI * sin(M_PI*localY);
        
        // Rotation matrix around the x axis
        FieldMatrix<double,TargetSpace::CoordinateType::dimension,TargetSpace::CoordinateType::dimension> rotationX(0);
        rotationX[0][0] = 1;
        rotationX[1][1] = cos(angleY);
        rotationX[1][2] = -sin(angleY);
        rotationX[2][1] = sin(angleY);
        rotationX[2][2] = cos(angleY);

        // Rotation matrix around the y axis
        FieldMatrix<double,TargetSpace::CoordinateType::dimension,TargetSpace::CoordinateType::dimension> rotationY(0);
        rotationY[0][0] = cos(angleX);
        rotationY[0][2] = -sin(angleX);
        rotationY[1][1] = 1;
        rotationY[2][0] = sin(angleX);
        rotationY[2][2] = cos(angleX);

        //angle *= -4*x[1]*(x[1]-1);
        TargetSpace::CoordinateType unitVector(0);   unitVector[2] = 1;
        
        TargetSpace::CoordinateType tmp;
        rotationX.mv(unitVector,tmp);
        rotationY.mv(tmp,out);
#endif
    }
};


template <class GridType>
void solve (const shared_ptr<GridType>& grid,
            SolutionType& x, 
            int numLevels,
            const ParameterTree& parameters)
{
    // read solver setting
    const double innerTolerance           = parameters.get<double>("innerTolerance");
    const double tolerance                = parameters.get<double>("tolerance");
    const int maxTrustRegionSteps         = parameters.get<int>("maxTrustRegionSteps");
    const double initialTrustRegionRadius = parameters.get<double>("initialTrustRegionRadius");
    const int multigridIterations         = parameters.get<int>("numIt");

    // /////////////////////////////////////////
    //   Read Dirichlet values
    // /////////////////////////////////////////

    BitSetVector<1> allNodes(grid->size(dim));
    allNodes.setAll();
    BoundaryPatch<typename GridType::LeafGridView> dirichletBoundary(grid->leafGridView(), allNodes);

#if defined THIRD_ORDER
    typedef P3NodalBasis<typename GridType::LeafGridView,double> FEBasis;
#elif defined SECOND_ORDER
    typedef P2NodalBasis<typename GridType::LeafGridView,double> FEBasis;
#else
    typedef P1NodalBasis<typename GridType::LeafGridView,double> FEBasis;
#endif
    FEBasis feBasis(grid->leafGridView());
    
    BitSetVector<blocksize> dirichletNodes;
    constructBoundaryDofs(dirichletBoundary,feBasis,dirichletNodes);
    
    // //////////////////////////
    //   Initial solution
    // //////////////////////////

    x.resize(feBasis.size());
    
    BlockVector<TargetSpace::CoordinateType> dirichletFunctionValues;
    DirichletFunction dirichletFunction;
    Functions::interpolate(feBasis, dirichletFunctionValues, dirichletFunction);

    TargetSpace::CoordinateType innerValue(0);
    innerValue[0] = 1;
    innerValue[1] = 0;
    
    for (size_t i=0; i<x.size(); i++)
        x[i] = (dirichletNodes[i][0]) ? dirichletFunctionValues[i] : innerValue;
    
    // ////////////////////////////////////////////////////////////
    //   Create an assembler for the Harmonic Energy Functional
    // ////////////////////////////////////////////////////////////

    HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,typename FEBasis::LocalFiniteElement, TargetSpace> harmonicEnergyLocalStiffness;

    GeodesicFEAssembler<FEBasis,TargetSpace> assembler(grid->leafGridView(),
                                                       &harmonicEnergyLocalStiffness);

    // ///////////////////////////////////////////
    //   Create a solver for the rod problem
    // ///////////////////////////////////////////

    RiemannianTrustRegionSolver<GridType,TargetSpace> solver;

    solver.setup(*grid, 
                 &assembler,
                 x,
                 dirichletNodes,
                 tolerance,
                 maxTrustRegionSteps,
                 initialTrustRegionRadius,
                 multigridIterations,
                 innerTolerance,
                 1, 3, 3,
                 100,     // iterations of the base solver
                 1e-8,    // base tolerance
                 false);  // instrumentation

    // /////////////////////////////////////////////////////
    //   Solve!
    // /////////////////////////////////////////////////////

    solver.setInitialIterate(x);
    solver.solve();

    x = solver.getSol();
}

int main (int argc, char *argv[]) try
{
    // parse data file
    ParameterTree parameterSet;
    if (argc==2)
        ParameterTreeParser::readINITree(argv[1], parameterSet);
    else
        ParameterTreeParser::readINITree("harmonicmaps-eoc.parset", parameterSet);

    // read solver settings
    const int numLevels        = parameterSet.get<int>("numLevels");

    // grid file
    std::string gridFileName = parameterSet.get<std::string>("gridFile");
    
    // only if a structured grid is used
    const int numBaseElements = parameterSet.get<int>("numBaseElements");
    FieldVector<double,dim> lowerLeft  = parameterSet.get<FieldVector<double,dim> >("lowerLeft");
    FieldVector<double,dim> upperRight = parameterSet.get<FieldVector<double,dim> >("upperRight");
    
    // ///////////////////////////////////////////////////////////
    //   First compute the 'exact' solution on a very fine grid
    // ///////////////////////////////////////////////////////////

    typedef std::conditional<dim==1,OneDGrid,UGGrid<dim> >::type GridType;

    //    Create the reference grid
    shared_ptr<GridType> referenceGrid;
    
    if (parameterSet.get<std::string>("gridType")=="structured") {
        array<unsigned int,dim> elements;
        elements.fill(numBaseElements);
        referenceGrid = StructuredGridFactory<GridType>::createSimplexGrid(lowerLeft,
                                                                           upperRight,
                                                                           elements);
    } else {
        
        if (gridFileName.rfind(".msh")!=std::string::npos)
            referenceGrid = shared_ptr<GridType>(GmshReader<GridType>::read(gridFileName));
        else    
            referenceGrid = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(gridFileName));
    }
    
    referenceGrid->globalRefine(numLevels-1);

    //  Solve the rod Dirichlet problem
    SolutionType referenceSolution;
    solve(referenceGrid, referenceSolution, numLevels, parameterSet);

    BlockVector<TargetSpace::CoordinateType> xEmbedded(referenceSolution.size());
    for (size_t j=0; j<referenceSolution.size(); j++)
        xEmbedded[j] = referenceSolution[j].globalCoordinates();

    //  Set up a basis of the underlying fe space
#ifdef THIRD_ORDER
    typedef P3NodalBasis<GridType::LeafGridView,double> FEBasis;
#elif defined SECOND_ORDER
    typedef P2NodalBasis<GridType::LeafGridView,double> FEBasis;
#else
    typedef P1NodalBasis<GridType::LeafGridView,double> FEBasis;
#endif
    FEBasis referenceBasis(referenceGrid->leafGridView());

#if !defined THIRD_ORDER && ! defined SECOND_ORDER
    VTKWriter<GridType::LeafGridView> vtkWriter(referenceGrid->leafGridView());
    
    shared_ptr<VTKBasisGridFunction<FEBasis,BlockVector<TargetSpace::CoordinateType> > > vtkVectorField
        = Dune::shared_ptr<VTKBasisGridFunction<FEBasis,BlockVector<TargetSpace::CoordinateType> > >
            (new VTKBasisGridFunction<FEBasis,BlockVector<TargetSpace::CoordinateType> >(referenceBasis, xEmbedded, "unit vectors"));

    vtkWriter.addVertexData(vtkVectorField);
    vtkWriter.write("reference_result");
#endif
    
    // //////////////////////////////////////////////////////////////////////
    //   Compute mass matrix and laplace matrix to emulate L2 and H1 norms
    // //////////////////////////////////////////////////////////////////////

    OperatorAssembler<FEBasis,FEBasis> operatorAssembler(referenceBasis, referenceBasis);

    LaplaceAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> laplaceLocalAssembler;
    MassAssembler<GridType, FEBasis::LocalFiniteElement, FEBasis::LocalFiniteElement> massMatrixLocalAssembler;

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrixType;
    ScalarMatrixType laplace, massMatrix;

    operatorAssembler.assemble(laplaceLocalAssembler, laplace);
    operatorAssembler.assemble(massMatrixLocalAssembler, massMatrix);

    // ///////////////////////////////////////////////////////////
    //   Compute on all coarser levels, and compare
    // ///////////////////////////////////////////////////////////
    
    std::ofstream logFile("harmonicmaps-eoc.results");
    logFile << "# mesh size, max-norm, L2-norm, h1-seminorm" << std::endl;
    
    for (int i=1; i<numLevels; i++) {

        shared_ptr<GridType> grid;
        if (parameterSet.get<std::string>("gridType")=="structured") {
            array<unsigned int,dim> elements;
            elements.fill(numBaseElements);
            grid = StructuredGridFactory<GridType>::createSimplexGrid(lowerLeft,
                                                                      upperRight,
                                                                      elements);
        } else {
            if (gridFileName.rfind(".msh")!=std::string::npos)
                grid = shared_ptr<GridType>(GmshReader<GridType>::read(gridFileName));
            else    
                grid = shared_ptr<GridType>(AmiraMeshReader<GridType>::read(gridFileName));
        }

        grid->globalRefine(i-1);

        // compute again
        SolutionType solution;
        solve(grid, solution, i, parameterSet);

        // write solution
        std::stringstream numberAsAscii;
        numberAsAscii << i;

        BlockVector<TargetSpace::CoordinateType> xEmbedded(solution.size());
        for (size_t j=0; j<solution.size(); j++)
            xEmbedded[j] = solution[j].globalCoordinates();

        FEBasis feBasis(grid->leafGridView());
        VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafGridView());
    
        shared_ptr<VTKBasisGridFunction<FEBasis,BlockVector<TargetSpace::CoordinateType> > > vtkVectorField
            = Dune::shared_ptr<VTKBasisGridFunction<FEBasis,BlockVector<TargetSpace::CoordinateType> > >
                (new VTKBasisGridFunction<FEBasis,BlockVector<TargetSpace::CoordinateType> >(feBasis, xEmbedded, "unit vectors"));

        vtkWriter.addVertexData(vtkVectorField);
        vtkWriter.write("harmonic_result_" + numberAsAscii.str());
        
        double newL2 = GFEDifferenceNormSquared<FEBasis,TargetSpace>::compute(*referenceGrid, referenceSolution,
                                                         *grid, solution,
                                                         &massMatrixLocalAssembler);
        newL2 = std::sqrt(newL2);

        double newH1 = GFEDifferenceNormSquared<FEBasis,TargetSpace>::compute(*referenceGrid, referenceSolution,
                                                         *grid, solution,
                                                         &laplaceLocalAssembler);
        newH1 = std::sqrt(newH1);
        
        
        // Prolong solution to the very finest grid
        for (int j=i; j<numLevels; j++) {
            FEBasis basis(grid->leafGridView());
#if defined THIRD_ORDER || defined SECOND_ORDER
            GeodesicFEFunctionAdaptor<FEBasis,TargetSpace>::higherOrderGFEFunctionAdaptor<order>(basis, *grid, solution);
#else
            GeodesicFEFunctionAdaptor<FEBasis,TargetSpace>::geodesicFEFunctionAdaptor(*grid, solution);
#endif
        }

        // Interpret TargetSpace as isometrically embedded into an R^m, because this is
        // how the corresponding Sobolev spaces are defined.

        BlockVector<TargetSpace::CoordinateType> difference(referenceSolution.size());

        for (size_t j=0; j<referenceSolution.size(); j++)
            difference[j] = solution[j].globalCoordinates() - referenceSolution[j].globalCoordinates();

        H1SemiNorm< BlockVector<TargetSpace::CoordinateType> > h1Norm(laplace);
        H1SemiNorm< BlockVector<TargetSpace::CoordinateType> > l2Norm(massMatrix);

        // Compute max-norm difference
        std::cout << "Level: " << i-1 << "   vertices: " << xEmbedded.size() << std::endl;
        std::cout << "h: " << std::pow(0.5, i-1) << std::endl;
        std::cout << "Level: " << i-1 
                  << ",   max-norm error: " << difference.infinity_norm()
                  << std::endl;

        std::cout << "Level: " << i-1 
                  << ",   L2 error: " << l2Norm(difference) << ",   new: " << newL2
                  << std::endl;

        std::cout << "Level: " << i-1 
                  << ",   H1 error: " << h1Norm(difference) << ",   new: " << newH1
                  << std::endl;
                  
        logFile << std::pow(0.5, i-1) << "  " << difference.infinity_norm() 
                << "  " << l2Norm(difference)
                << "  " << h1Norm(difference)
                << std::endl;

    }    
        
 } catch (Exception e) {

    std::cout << e << std::endl;

 }
