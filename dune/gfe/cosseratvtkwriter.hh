#ifndef COSSERAT_VTK_WRITER_HH
#define COSSERAT_VTK_WRITER_HH

#include <dune/common/version.hh>

#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/io/file/vtk/pvtuwriter.hh>

#include <dune/istl/bvector.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/gridfunctions/composedgridfunction.hh>

#include <dune/vtk/vtkwriter.hh>
#include <dune/vtk/datacollectors/lagrangedatacollector.hh>

#include <dune/gfe/vtkfile.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>


namespace Dune::GFE
{

/** \brief Write the configuration of a Cosserat material in VTK format */
template <class GridView>
class CosseratVTKWriter
{

  static const int dim = GridView::dimension;

  template <typename Basis1, typename Basis2>
  static void downsample(const Basis1& basis1, const std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& v1,
                         const Basis2& basis2,       std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& v2)
  {
    // Embed v1 into R^7
    std::vector<Dune::FieldVector<double,7> > v1Embedded(v1.size());
    for (size_t i=0; i<v1.size(); i++)
      v1Embedded[i] = v1[i].globalCoordinates();

    // Interpolate
    auto function = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double,7> >(basis1, v1Embedded);
    std::vector<Dune::FieldVector<double,7> > v2Embedded;
    Dune::Functions::interpolate(basis2, v2Embedded, function);

    // Copy back from R^7 into ProductManifold
    v2.resize(v2Embedded.size());
    for (size_t i=0; i<v2.size(); i++)
      v2[i] = Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> >(v2Embedded[i]);
  }

  template <typename Basis1, typename Basis2>
  static void downsample(const Basis1& basis1, const std::vector<RealTuple<double,3> >& v1,
                         const Basis2& basis2,       std::vector<RealTuple<double,3> >& v2)
  {
    // Copy from RealTuple to FieldVector
    std::vector<Dune::FieldVector<double,3> > v1Embedded(v1.size());
    for (size_t i=0; i<v1.size(); i++)
      v1Embedded[i] = v1[i].globalCoordinates();

    // Interpolate
    auto function = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double,3> >(basis1, v1Embedded);
    std::vector<Dune::FieldVector<double,3> > v2Embedded;
    Dune::Functions::interpolate(basis2, v2Embedded, function);

    // Copy back from FieldVector to RealTuple
    v2.resize(v2Embedded.size());
    for (size_t i=0; i<v2.size(); i++)
      v2[i] = RealTuple<double,3>(v2Embedded[i]);
  }

  /** \brief Extend filename to contain communicator rank and size
   *
   * Copied from dune-grid vtkwriter.hh
   */
  static std::string getParallelPieceName(const std::string& name,
                                          const std::string& path,
                                          int commRank, int commSize)
  {
    std::ostringstream s;
    if(path.size() > 0) {
      s << path;
      if(path[path.size()-1] != '/')
        s << '/';
    }
    s << 's' << std::setw(4) << std::setfill('0') << commSize << '-';
    s << 'p' << std::setw(4) << std::setfill('0') << commRank << '-';
    s << name;
    if(GridView::dimension > 1)
      s << ".vtu";
    else
      s << ".vtp";
    return s.str();
  }

  /** \brief Extend filename to contain communicator rank and size
   *
   * Copied from dune-grid vtkwriter.hh
   */
  static std::string getParallelName(const std::string& name,
                                     const std::string& path,
                                     int commSize)
  {
    std::ostringstream s;
    if(path.size() > 0) {
      s << path;
      if(path[path.size()-1] != '/')
        s << '/';
    }
    s << 's' << std::setw(4) << std::setfill('0') << commSize << '-';
    s << name;
    if(GridView::dimension > 1)
      s << ".pvtu";
    else
      s << ".pvtp";
    return s.str();
  }

public:
  /** \brief Write a configuration given with respect to a scalar function space basis
   */
  template <typename Basis>
  static void write(const Basis& basis,
                    const Dune::TupleVector<std::vector<RealTuple<double,3> >,
                        std::vector<Rotation<double,3> > >& configuration,
                    const std::string& filename)
  {
    using namespace Dune::Indices;
    std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > > xRBM(basis.size());
    for (std::size_t i = 0; i < basis.size(); i++) {
      for (int j = 0; j < 3; j ++)   // Displacement part
        xRBM[i][_0].globalCoordinates()[j] = configuration[_0][i][j];
      xRBM[i][_1] = configuration[_1][i];      // Rotation part
    }
    write(basis,xRBM,filename);
  }
  /** \brief Write a configuration given with respect to a scalar function space basis
   */
  template <typename Basis, typename VectorType>
  static void write(const Basis& basis,
                    const VectorType& configuration,
                    const std::string& filename)
  {
    using namespace Dune::Indices;
    std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > > xRBM(basis.size());
    for (std::size_t i = 0; i < basis.size(); i++) {
      for (int j = 0; j < 3; j ++)   // Displacement part
        xRBM[i][_0].globalCoordinates()[j] = configuration[i][j];
    }
    write(basis,xRBM,filename);
  }

  /** \brief Write a configuration given with respect to a scalar function space basis
   */
  template <typename Basis>
  static void write(const Basis& basis,
                    const std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > >& configuration,
                    const std::string& filename)
  {
    using namespace Dune::Indices;

    assert(basis.size() == configuration.size());
    auto gridView = basis.gridView();

    // Determine order of the basis
    // We check for the order of the first element, and assume it is the same for all others
    auto localView = basis.localView();
    localView.bind(*gridView.template begin<0>());
    const int order = localView.tree().finiteElement().localBasis().order();
    // order of the approximation of the VTK file -- can only be two or one
    const auto vtkOrder = std::min(2,order);

    //  Downsample 3rd-order functions onto a P2-space.  That's all VTK can visualize today.
    if (order>=3)
    {
      using namespace Dune::Functions::BasisFactory;
      auto p2Basis = makeBasis(gridView, lagrange<2>());

      auto blockedP2Basis = makeBasis(
        gridView,
        power<3>(
          lagrange<2>(),
          blockedInterleaved()
          ));

      std::vector<Dune::GFE::ProductManifold<RealTuple<double,3>,Rotation<double,3> > > downsampledConfig;

      downsample(basis, configuration, blockedP2Basis, downsampledConfig);

      write(p2Basis, downsampledConfig, filename);
      return;
    }

    Dune::GFE::VTKFile vtkFile;

    // Count the number of elements of the different types
    std::map<Dune::GeometryType,std::size_t> numElements;
    for (const auto t : gridView.indexSet().types(0))
      numElements[t] = 0;

    for (auto&& t : elements(gridView, Dune::Partitions::interior))
      numElements[t.type()]++;

    std::size_t totalNumElements = 0;
    for (const auto nE : numElements)
      totalNumElements += nE.second;

    // Enter vertex coordinates
    std::vector<Dune::FieldVector<double, 3> > points(configuration.size());
    for (size_t i=0; i<configuration.size(); i++)
      points[i] = configuration[i][_0].globalCoordinates();

    vtkFile.points_ = points;

    // Enter elements
    std::size_t connectivitySize = 0;
    for (const auto nE : numElements)
    {
      if (nE.first.isQuadrilateral())
        connectivitySize += ((vtkOrder==2) ? 8 : 4) * nE.second;
      else if (nE.first.isTriangle())
        connectivitySize += ((vtkOrder==2) ? 6 : 3) * nE.second;
      else if (nE.first.isHexahedron())
        connectivitySize += ((vtkOrder==2) ? 20 : 8) * nE.second;
      else if (nE.first.isLine())
        connectivitySize += ((vtkOrder==2) ? 3 : 2) * nE.second;
      else
        DUNE_THROW(Dune::IOError, "Unsupported element type '" << nE.first << "' found!");
    }
    std::vector<unsigned int> connectivity(connectivitySize);

    size_t i=0;
    for (const auto& element : elements(gridView, Dune::Partitions::interior))
    {
      localView.bind(element);

      if (element.type().isQuadrilateral())
      {
        if (vtkOrder==2)
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(2);
          connectivity[i++] = localView.index(8);
          connectivity[i++] = localView.index(6);

          connectivity[i++] = localView.index(1);
          connectivity[i++] = localView.index(5);
          connectivity[i++] = localView.index(7);
          connectivity[i++] = localView.index(3);
        }
        else      // first order
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(1);
          connectivity[i++] = localView.index(3);
          connectivity[i++] = localView.index(2);
        }
      }
      if (element.type().isTriangle())
      {
        if (vtkOrder==2)
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(2);
          connectivity[i++] = localView.index(5);
          connectivity[i++] = localView.index(1);
          connectivity[i++] = localView.index(4);
          connectivity[i++] = localView.index(3);
        }
        else      // first order
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(1);
          connectivity[i++] = localView.index(2);
        }
      }
      if (element.type().isHexahedron())
      {
        if (vtkOrder==2)
        {
          // Corner dofs
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(2);
          connectivity[i++] = localView.index(8);
          connectivity[i++] = localView.index(6);

          connectivity[i++] = localView.index(18);
          connectivity[i++] = localView.index(20);
          connectivity[i++] = localView.index(26);
          connectivity[i++] = localView.index(24);

          // Edge dofs
          connectivity[i++] = localView.index(1);
          connectivity[i++] = localView.index(5);
          connectivity[i++] = localView.index(7);
          connectivity[i++] = localView.index(3);

          connectivity[i++] = localView.index(19);
          connectivity[i++] = localView.index(23);
          connectivity[i++] = localView.index(25);
          connectivity[i++] = localView.index(21);

          connectivity[i++] = localView.index(9);
          connectivity[i++] = localView.index(11);
          connectivity[i++] = localView.index(17);
          connectivity[i++] = localView.index(15);
        }
        else      // first order
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(1);
          connectivity[i++] = localView.index(3);
          connectivity[i++] = localView.index(2);
          connectivity[i++] = localView.index(4);
          connectivity[i++] = localView.index(5);
          connectivity[i++] = localView.index(7);
          connectivity[i++] = localView.index(6);
        }
      }

      if (element.type().isLine())
      {
        if (vtkOrder==2)
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(2);
          connectivity[i++] = localView.index(1);
        }
        else      // first order
        {
          connectivity[i++] = localView.index(0);
          connectivity[i++] = localView.index(1);
        }
      }
    }

    vtkFile.cellConnectivity_ = connectivity;

    std::vector<int> offsets(totalNumElements);
    i = 0;
    int offsetCounter = 0;
    for (const auto& element : elements(gridView, Dune::Partitions::interior))
    {
      if (element.type().isQuadrilateral())
        offsetCounter += (vtkOrder==2) ? 8 : 4;

      if (element.type().isTriangle())
        offsetCounter += (vtkOrder==2) ? 6 : 3;

      if (element.type().isHexahedron())
        offsetCounter += (vtkOrder==2) ? 20 : 8;

      offsets[i++] += offsetCounter;
    }

    vtkFile.cellOffsets_ = offsets;

    std::vector<int> cellTypes(totalNumElements);
    i = 0;
    for (const auto& element : elements(gridView, Dune::Partitions::interior))
    {
      if (element.type().isQuadrilateral())
        cellTypes[i++] = (vtkOrder==2) ? 23 : 9;

      if (element.type().isTriangle())
        cellTypes[i++] = (vtkOrder==2) ? 22 : 5;

      if (element.type().isHexahedron())
        cellTypes[i++] = (vtkOrder==2) ? 25 : 12;
    }
    vtkFile.cellTypes_ = cellTypes;

    // Z coordinate for better visualization of wrinkles
    std::vector<double> zCoord(points.size());
    for (size_t i=0; i<configuration.size(); i++)
      zCoord[i] = configuration[i][_0].globalCoordinates()[2];

    vtkFile.zCoord_ = zCoord;

    // The three director fields
    for (size_t i=0; i<3; i++)
    {
      vtkFile.directors_[i].resize(configuration.size());
      for (size_t j=0; j<configuration.size(); j++)
        vtkFile.directors_[i][j] = configuration[j][_1].director(i);
    }

    // Actually write the VTK file to disk
    vtkFile.write(filename);
  }

  /** \brief Write a Cosserat configuration as VTK file
   *
   * The microrotation field will be represented as the three director fields
   *
   * \param directorBasis The basis that will be used to represent director fields
   * \param order The polynomial order that the data will have in the VTK file
   */
  template <typename DisplacementFunction, typename OrientationFunction>
  static void write(const GridView& gridView,
                    const DisplacementFunction& displacement,
                    const OrientationFunction& orientation,
                    int order,
                    const std::string& filename)
  {
    using namespace Dune;

    // Create a writer object
#if DUNE_VERSION_GTE(DUNE_VTK, 2, 10)
    auto vtkWriter = Vtk::UnstructuredGridWriter(Vtk::LagrangeDataCollector(gridView,order));
#else
    Vtk::LagrangeDataCollector<GridView> dataCollector(gridView, order);
    auto vtkWriter = VtkUnstructuredGridWriter(dataCollector);
#endif

    // Attach the displacement field
    vtkWriter.addPointData(displacement, Dune::VTK::FieldInfo("displacement", Dune::VTK::FieldInfo::Type::vector, 3));

    // Attach the director fields
    // This lambda takes a unit quaternion and extracts one column
    // of the corresponding rotation matrix
    auto directorExtractor = [](FieldVector<double,4> q,int columnNumber) -> FieldVector<double,3>
                             {
                               FieldMatrix<double,3,3> matrix;
                               Rotation<double,3>(q).matrix(matrix);
                               FieldVector<double,3> column;
                               for (size_t i=0; i<3; ++i)
                                 column[i] = matrix[i][columnNumber];
                               return column;
                             };

    auto director0Function = Functions::makeComposedGridFunction(std::bind(directorExtractor,std::placeholders::_1,0),
                                                                 orientation);

    auto director1Function = Functions::makeComposedGridFunction(std::bind(directorExtractor,std::placeholders::_1,1),
                                                                 orientation);

    auto director2Function = Functions::makeComposedGridFunction(std::bind(directorExtractor,std::placeholders::_1,2),
                                                                 orientation);

    vtkWriter.addPointData(director0Function, Dune::VTK::FieldInfo("director0", Dune::VTK::FieldInfo::Type::vector, 3));
    vtkWriter.addPointData(director1Function, Dune::VTK::FieldInfo("director1", Dune::VTK::FieldInfo::Type::vector, 3));
    vtkWriter.addPointData(director2Function, Dune::VTK::FieldInfo("director2", Dune::VTK::FieldInfo::Type::vector, 3));

    // Write the file
    vtkWriter.write(filename);
  }

};

}  // namespace Dune::GFE

#endif
