#ifndef COSSERAT_VTK_WRITER_HH
#define COSSERAT_VTK_WRITER_HH

#include <dune/common/version.hh>

#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/io/file/vtk/pvtuwriter.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

#include <dune/gfe/vtkfile.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>


/** \brief Write the configuration of a Cosserat material in VTK format */
template <class GridType>
class CosseratVTKWriter
{

  static const int dim = GridType::dimension;

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
    if(GridType::dimension > 1)
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
    if(GridType::dimension > 1)
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
    using namespace Dune::TypeTree::Indices;
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
    using namespace Dune::TypeTree::Indices;
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

  /** \brief Write a configuration given with respect to a scalar function space basis
   */
  template <typename DisplacementBasis, typename OrientationBasis>
  static void writeMixed(const DisplacementBasis& displacementBasis,
                         const std::vector<RealTuple<double,3> >& deformationConfiguration,
                         const OrientationBasis& orientationBasis,
                         const std::vector<Rotation<double,3> >& orientationConfiguration,
                         const std::string& filename)
  {
    assert(displacementBasis.size() == deformationConfiguration.size());
    assert(orientationBasis.size()  == orientationConfiguration.size());
    auto gridView = displacementBasis.gridView();

    // Determine order of the displacement basis
    auto localView = displacementBasis.localView();
    localView.bind(*gridView.template begin<0>());
    int order = localView.tree().finiteElement().localBasis().order();

    // We only do P2 spaces at the moment
    if (order != 2 and order != 3)
    {
      std::cout << "Warning: CosseratVTKWriter only supports P2 spaces -- skipping" << std::endl;
      return;
    }

    std::vector<RealTuple<double,3> > displacementConfiguration = deformationConfiguration;

    if (order == 3)
    {
      using namespace Dune::Functions::BasisFactory;

      auto p2DeformationBasis = makeBasis(
        gridView,
        power<3>(
          lagrange<2>(),
          blockedInterleaved()
          ));

      // resample to 2nd order -- vtk can't do anything higher
      std::vector<RealTuple<double,3> > p2DeformationConfiguration;

      downsample(displacementBasis, displacementConfiguration,
                 p2DeformationBasis, p2DeformationConfiguration);

      displacementConfiguration = p2DeformationConfiguration;
    }

    std::string fullfilename = filename + ".vtu";

    // Prepend rank and communicator size to the filename, if there are more than one process
    if (gridView.comm().size() > 1)
      fullfilename = getParallelPieceName(filename, "", gridView.comm().rank(), gridView.comm().size());

    // Write the pvtu file that ties together the different parts
    if (gridView.comm().size() > 1 && gridView.comm().rank()==0)
    {
      std::ofstream pvtuOutFile(getParallelName(filename, "", gridView.comm().size()));
      Dune::VTK::PVTUWriter writer(pvtuOutFile, Dune::VTK::unstructuredGrid);

      writer.beginMain();

      //writer.beginPointData();
      //writer.addArray<float>("director0", 3);
      //writer.addArray<float>("director1", 3);
      //writer.addArray<float>("director2", 3);
      //writer.addArray<float>("zCoord", 1);
      //writer.endPointData();

      // dump point coordinates
      writer.beginPoints();
      writer.addArray("Coordinates", 3, Dune::VTK::Precision::float32);
      writer.endPoints();

      for (int i=0; i<gridView.comm().size(); i++)
        writer.addPiece(getParallelPieceName(filename, "", i, gridView.comm().size()));

      // finish main section
      writer.endMain();
    }

    /////////////////////////////////////////////////////////////////////////////////
    //  Write the actual vtu file
    /////////////////////////////////////////////////////////////////////////////////

    // Stupid: I can't directly get the number of Interior_Partition elements
    size_t numElements = std::distance(gridView.template begin<0, Dune::Interior_Partition>(),
                                       gridView.template end<0, Dune::Interior_Partition>());

    std::ofstream outFile(fullfilename);

    // Write header
    outFile << "<?xml version=\"1.0\"?>" << std::endl;
    outFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    outFile << "  <UnstructuredGrid>" << std::endl;
    outFile << "    <Piece NumberOfCells=\"" << numElements << "\" NumberOfPoints=\"" << displacementConfiguration.size() << "\">" << std::endl;

    // Write vertex coordinates
    outFile << "      <Points>" << std::endl;
    outFile << "        <DataArray type=\"Float32\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    for (size_t i=0; i<displacementConfiguration.size(); i++)
      outFile << "          " << displacementConfiguration[i] << std::endl;
    outFile << "        </DataArray>" << std::endl;
    outFile << "      </Points>" << std::endl;

    // Write elements
    outFile << "      <Cells>" << std::endl;

    outFile << "         <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;

    for (const auto& element : elements(gridView, Dune::Partitions::interior))
    {
      localView.bind(element);

      outFile << "          ";
      if (element.type().isQuadrilateral())
      {
        outFile << localView.index(0) << " ";
        outFile << localView.index(2) << " ";
        outFile << localView.index(8) << " ";
        outFile << localView.index(6) << " ";

        outFile << localView.index(1) << " ";
        outFile << localView.index(5) << " ";
        outFile << localView.index(7) << " ";
        outFile << localView.index(3) << " ";

        outFile << std::endl;
      }
    }
    outFile << "         </DataArray>" << std::endl;

    outFile << "         <DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    size_t offsetCounter = 0;
    for (const auto& element : elements(gridView))
    {
      outFile << "          ";
      if (element.type().isQuadrilateral())
        offsetCounter += 8;
      outFile << offsetCounter << std::endl;
    }
    outFile << "         </DataArray>" << std::endl;

    outFile << "         <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    for (const auto& element : elements(gridView))
    {
      outFile << "          ";
      if (element.type().isQuadrilateral())
        outFile << "23" << std::endl;
    }
    outFile << "         </DataArray>" << std::endl;

    outFile << "      </Cells>" << std::endl;
#if 0
    // Point data
    outFile << "      <PointData Scalars=\"zCoord\" Vectors=\"director0\">" << std::endl;

    // Z coordinate for better visualization of wrinkles
    outFile << "        <DataArray type=\"Float32\" Name=\"zCoord\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    for (size_t i=0; i<configuration.size(); i++)
      outFile << "          " << configuration[i].r[2] << std::endl;
    outFile << "        </DataArray>" << std::endl;

    // The three director fields
    for (size_t i=0; i<3; i++)
    {
      outFile << "        <DataArray type=\"Float32\" Name=\"director" << i <<"\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
      for (size_t j=0; j<configuration.size(); j++)
        outFile << "          " << configuration[j].q.director(i) << std::endl;
      outFile << "        </DataArray>" << std::endl;
    }

    outFile << "      </PointData>" << std::endl;
#endif
    // Write footer
    outFile << "    </Piece>" << std::endl;
    outFile << "  </UnstructuredGrid>" << std::endl;
    outFile << "</VTKFile>" << std::endl;

  }

};

#endif
