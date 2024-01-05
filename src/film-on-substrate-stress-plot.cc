#include <iostream>
#include <fstream>

#include <config.h>

// Includes for the ADOL-C automatic differentiation library
// Need to come before (almost) all others.
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/version.hh>

#include <dune/elasticity/materials/exphenckydensity.hh>
#include <dune/elasticity/materials/henckydensity.hh>
#include <dune/elasticity/materials/mooneyrivlindensity.hh>
#include <dune/elasticity/materials/neohookedensity.hh>
#include <dune/elasticity/materials/stvenantkirchhoffdensity.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/fufem/dunepython.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk.hh>

#include <dune/gfe/filereader.hh>
#include <dune/gfe/assemblers/surfacecosseratstressassembler.hh>
#include <dune/gfe/spaces/rotation.hh>

// grid dimension
#ifndef WORLD_DIM
#  define WORLD_DIM 3
#endif
const int dim = WORLD_DIM;

const int displacementOrder = 2;
const int rotationOrder = 2;

using namespace Dune;
using ValueType = adouble;

int main (int argc, char *argv[]) try
{
  MPIHelper::instance(argc, argv);

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

  // Surface Shell Boundary
  std::string lambda = std::string("lambda x: (") + parameterSet.get<std::string>("surfaceShellVerticesPredicate", "0") + std::string(")");
  auto pythonSurfaceShellVertices = Python::make_function<bool>(Python::evaluate(lambda));

  int numLevels = parameterSet.get<int>("numLevels");

  while (numLevels > 0) {
    for (auto&& e : elements(grid->leafGridView())) {
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

  if (grid->leafGridView().comm().size() > 1)
    DUNE_THROW(Exception,
               std::string("To create a stress plot, please use only one process, now there are ") + std::to_string(grid->leafGridView().comm().size()) + std::string(" procsses."));

  typedef GridType::LeafGridView GridView;
  GridView gridView = grid->leafGridView();

  /////////////////////////////////////////////////////////////
  //                   SURFACE SHELL BOUNDARY
  /////////////////////////////////////////////////////////////

  const GridView::IndexSet& indexSet = gridView.indexSet();
  BitSetVector<1> surfaceShellVertices(gridView.size(dim), false);
  for (auto&& v : vertices(gridView))
  {
    bool isSurfaceShell = pythonSurfaceShellVertices(v.geometry().corner(0));
    surfaceShellVertices[indexSet.index(v)] = isSurfaceShell;
  }
  BoundaryPatch<GridView> surfaceShellBoundary(gridView, surfaceShellVertices);

  /////////////////////////////////////////////////////////////
  //                      FUNCTION SPACE
  /////////////////////////////////////////////////////////////

  using namespace Functions::BasisFactory;
  auto basisOrderD = makeBasis(
    gridView,
    power<dim>(
      lagrange<displacementOrder>()
      ));

  auto basisOrderR = makeBasis(
    gridView,
    power<dim>(
      lagrange<rotationOrder>()
      ));

  /////////////////////////////////////////////////////////////
  //                      INITIAL DATA
  /////////////////////////////////////////////////////////////

  // Read grid deformation information from the file specified in the parameter set via pathToOutput, deformationOutput, rotationOutput and initial deformation
  const std::string pathToOutput = parameterSet.get("pathToOutput", "");

  std::cout << "Reading in deformation file ("  << "order is "  << displacementOrder  << "): " << pathToOutput + parameterSet.get<std::string>("deformationOutput") << std::endl;
  auto deformationMap = Dune::GFE::transformFileToMap<dim>(pathToOutput + parameterSet.get<std::string>("deformationOutput"));
  std::cout << "... done: The basis has " << basisOrderD.size() << " elements and the defomation file has " << deformationMap.size() << " entries." << std::endl;

  const auto dimRotation = Rotation<double,dim>::embeddedDim;
  std::unordered_map<std::string, FieldVector<double,dimRotation> > rotationMap;
  if (parameterSet.hasKey("rotationOutput")) {
    std::cout << "Reading in rotation file ("  << "order is "  << rotationOrder  << "): " << pathToOutput + parameterSet.get<std::string>("rotationOutput") << std::endl;
    rotationMap = Dune::GFE::transformFileToMap<dimRotation>(pathToOutput + parameterSet.get<std::string>("rotationOutput"));
  }
  const bool startFromFile = parameterSet.get<bool>("startFromFile");
  std::unordered_map<std::string, FieldVector<double,dim> > initialDeformationMap;

  auto gridDeformationLambda = std::string("lambda x: (") + parameterSet.get<std::string>("gridDeformation") + std::string(")");
  auto gridDeformation = Python::make_function<FieldVector<double,dim> >(Python::evaluate(gridDeformationLambda));

  if (startFromFile) {
    std::cout << "Reading in file to the stress-free configuration of the shell ("  << "order is "  << displacementOrder  << "): " <<  parameterSet.get("pathToGridDeformationFile", "") + parameterSet.get<std::string>("gridDeformationFile") << std::endl;
    initialDeformationMap = Dune::GFE::transformFileToMap<dim>(parameterSet.get("pathToGridDeformationFile", "") + parameterSet.get<std::string>("gridDeformationFile"));
  }

  using DisplacementVector = std::vector<FieldVector<double,dim> >;
  DisplacementVector x;
  x.resize(basisOrderD.size());
  DisplacementVector xInitial;
  xInitial.resize(basisOrderD.size());
  DisplacementVector displacement;
  displacement.resize(basisOrderD.size());

  Functions::interpolate(basisOrderD, x, [](FieldVector<double,dim> x){
    return x;
  });
  Functions::interpolate(basisOrderD, xInitial, [](FieldVector<double,dim> x){
    return x;
  });

  for (std::size_t i = 0; i < basisOrderD.size(); i++) {
    std::stringstream stream;
    stream << x[i];
    //Look up the displacement for this vertex in the deformationMap
    displacement[i] = deformationMap.at(stream.str());
    x[i] += deformationMap.at(stream.str());
    //In case an a stress-free file was provided: look up the displacement for this vertex in the initialDeformationMap
    if (startFromFile) {
      xInitial[i] += initialDeformationMap.at(stream.str());
    } else {
      xInitial[i] = gridDeformation(xInitial[i]);
    }
  }

  using RotationVector = std::vector<Rotation<double,dim> >;
  RotationVector rot;
  rot.resize(basisOrderR.size());
  DisplacementVector xOrderR;
  xOrderR.resize(basisOrderR.size());
  Functions::interpolate(basisOrderR, xOrderR, [](FieldVector<double,dim> x){
    return x;
  });

  using DirectorVector = std::vector<Dune::FieldVector<double,dim> >;
  std::array<DirectorVector,3> rot_director;
  rot_director[0].resize(basisOrderR.size());
  rot_director[1].resize(basisOrderR.size());
  rot_director[2].resize(basisOrderR.size());

  for (std::size_t i = 0; i < basisOrderR.size(); i++) {
    std::stringstream stream;
    stream << xOrderR[i];
    Rotation<double,dim> rotation(rotationMap.at(stream.str()));
    FieldMatrix<double,dim,dim> rotationMatrix(0);
    rotation.matrix(rotationMatrix);
    rot[i].set(rotationMatrix);
    for (int j = 0; j < 3; j++)
      rot_director[j][i] = rot[i].director(j);
  }

  /////////////////////////////////////////////////////////////
  //                      STRESS ASSEMBLER
  /////////////////////////////////////////////////////////////
  int quadOrder = parameterSet.hasKey("quadOrder") ? parameterSet.get<int>("quadOrder") : 4;

  auto stressAssembler = GFE::SurfaceCosseratStressAssembler<decltype(basisOrderD),decltype(basisOrderR), FieldVector<double,dim>, Rotation<double,dim> >
                           (basisOrderD, basisOrderR);


  std::cout << "Selected energy is: " << parameterSet.get<std::string>("energy") << std::endl;
  std::shared_ptr<Elasticity::LocalDensity<dim,ValueType> > elasticDensity;

  const ParameterTree& materialParameters = parameterSet.sub("materialParameters");

  if (parameterSet.get<std::string>("energy") == "stvenantkirchhoff")
    elasticDensity = std::make_shared<Elasticity::StVenantKirchhoffDensity<dim,ValueType> >(materialParameters);
  if (parameterSet.get<std::string>("energy") == "neohooke")
    elasticDensity = std::make_shared<Elasticity::NeoHookeDensity<dim,ValueType> >(materialParameters);
  if (parameterSet.get<std::string>("energy") == "hencky")
    elasticDensity = std::make_shared<Elasticity::HenckyDensity<dim,ValueType> >(materialParameters);
  if (parameterSet.get<std::string>("energy") == "exphencky")
    elasticDensity = std::make_shared<Elasticity::ExpHenckyDensity<dim,ValueType> >(materialParameters);
  if (parameterSet.get<std::string>("energy") == "mooneyrivlin")
    elasticDensity = std::make_shared<Elasticity::MooneyRivlinDensity<dim,ValueType> >(materialParameters);

  if(!elasticDensity)
    DUNE_THROW(Exception, "Error: Selected energy not available!");

  Python::Reference surfaceShellClass = Python::import(materialParameters.get<std::string>("surfaceShellParameters"));
  Python::Callable surfaceShellCallable = surfaceShellClass.get("SurfaceShellParameters");
  Python::Reference pythonObject = surfaceShellCallable();
  auto fLame = Python::make_function<FieldVector<double, 2> >(pythonObject.get("lame"));

  std::vector<FieldMatrix<double,dim,dim> > stressSubstrate1stPiolaKirchhoffTensor;
  std::vector<FieldMatrix<double,dim,dim> > stressSubstrateCauchyTensor;
  std::cout << "Assemble stress for the substrate.." << std::endl;
  stressAssembler.assembleSubstrateStress<Elasticity::LocalDensity<dim,ValueType> >(x, elasticDensity.get(), quadOrder, stressSubstrate1stPiolaKirchhoffTensor, stressSubstrateCauchyTensor);

  std::vector<FieldMatrix<double,dim,dim> > stressShellBiotTensor;
  std::cout << "Assemble stress for the shell.." << std::endl;
  stressAssembler.assembleShellStress(rot, x, xInitial, fLame,/*mu_c*/ 0, surfaceShellBoundary, quadOrder, stressShellBiotTensor);

  std::vector<double> stressSubstrate1stPiolaKirchhoff(stressSubstrate1stPiolaKirchhoffTensor.size());
  std::vector<double> stressSubstrateCauchy(stressSubstrate1stPiolaKirchhoffTensor.size());
  std::vector<double> stressSubstrateVonMises(stressSubstrate1stPiolaKirchhoffTensor.size());
  std::vector<double> stressShellBiot(stressSubstrate1stPiolaKirchhoffTensor.size());

  for (size_t i = 0; i < stressSubstrate1stPiolaKirchhoffTensor.size(); i++) {
    stressSubstrate1stPiolaKirchhoff[i] = stressSubstrate1stPiolaKirchhoffTensor[i].frobenius_norm();
    stressSubstrateCauchy[i] = stressSubstrateCauchyTensor[i].frobenius_norm();

    double vonMises = 0; //von-Mises-Stress
    for(size_t j=0; j<dim; j++) {
      int jplus1 = (j+1) % dim;
      vonMises += 0.5*(stressSubstrate1stPiolaKirchhoffTensor[i][j][j] - stressSubstrate1stPiolaKirchhoffTensor[i][jplus1][jplus1])*(stressSubstrate1stPiolaKirchhoffTensor[i][j][j] - stressSubstrate1stPiolaKirchhoffTensor[i][jplus1][jplus1]);
      vonMises += 3*stressSubstrate1stPiolaKirchhoffTensor[i][j][jplus1]*stressSubstrate1stPiolaKirchhoffTensor[i][j][jplus1];
    }
    stressSubstrateVonMises[i] = std::sqrt(vonMises);

    stressShellBiot[i] = stressShellBiotTensor[i].frobenius_norm();
  }


  auto displacementFunction = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(basisOrderD, displacement);

  auto director0Function = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(basisOrderR, rot_director[0]);
  auto director1Function = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(basisOrderR, rot_director[1]);
  auto director2Function = Functions::makeDiscreteGlobalBasisFunction<FieldVector<double,dim> >(basisOrderR, rot_director[2]);

  SubsamplingVTKWriter<GridView> vtkWriter(gridView, refinementLevels(displacementOrder-1));
  vtkWriter.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriter.addVertexData(director0Function, VTK::FieldInfo("director0", VTK::FieldInfo::Type::vector, dim));
  vtkWriter.addVertexData(director1Function, VTK::FieldInfo("director1", VTK::FieldInfo::Type::vector, dim));
  vtkWriter.addVertexData(director2Function, VTK::FieldInfo("director2", VTK::FieldInfo::Type::vector, dim));
  vtkWriter.write("stress_plot_" + parameterSet.get<std::string>("energy"));

  VTKWriter<GridView> vtkWriterElement(gridView);
  vtkWriterElement.addCellData(stressShellBiot, "stress-shell-element");
  vtkWriterElement.addCellData(stressShellBiot, "stress-shell-biot");
  vtkWriterElement.addCellData(stressSubstrate1stPiolaKirchhoff, "stress-substrate-1st-piola-kirchhoff");
  vtkWriterElement.addCellData(stressSubstrateCauchy, "stress-substrate-cauchy");
  vtkWriterElement.addCellData(stressSubstrateVonMises, "stress-substrate-von-mises");
  vtkWriterElement.addVertexData(director0Function, VTK::FieldInfo("director0", VTK::FieldInfo::Type::vector, dim));
  vtkWriterElement.addVertexData(director1Function, VTK::FieldInfo("director1", VTK::FieldInfo::Type::vector, dim));
  vtkWriterElement.addVertexData(director2Function, VTK::FieldInfo("director2", VTK::FieldInfo::Type::vector, dim));
  vtkWriterElement.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriterElement.write("stress_plot_" + parameterSet.get<std::string>("energy") + "_element");

  VTKWriter<GridView> vtkWriterElementOnly(gridView);
  vtkWriterElementOnly.addCellData(stressShellBiot, "stress-shell-biot");
  vtkWriterElementOnly.addCellData(stressSubstrate1stPiolaKirchhoff, "stress-substrate-1st-piola-kirchhoff");
  vtkWriterElementOnly.addCellData(stressSubstrateCauchy, "stress-substrate-cauchy");
  vtkWriterElementOnly.addCellData(stressSubstrateVonMises, "stress-substrate-von-mises");
  vtkWriterElementOnly.addVertexData(displacementFunction, VTK::FieldInfo("displacement", VTK::FieldInfo::Type::scalar, dim));
  vtkWriterElementOnly.write("stress_plot_" + parameterSet.get<std::string>("energy") + "_element_only");

}
catch (Exception& e) {
  std::cout << e.what() << std::endl;
}
