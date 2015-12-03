#ifndef COSSERAT_VTK_WRITER_HH
#define COSSERAT_VTK_WRITER_HH

#include <dune/grid/geometrygrid.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/io/file/vtk/pvtuwriter.hh>

#include <dune/functions/functionspacebases/pqknodalbasis.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functionspacebases/dunefunctionsbasis.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/vtkfile.hh>


/** \brief Write the configuration of a Cosserat material in VTK format */
template <class GridType>
class CosseratVTKWriter
{

    static const int dim = GridType::dimension;

    /** \brief Encapsulates the grid deformation for the GeometryGrid class */
    template <class HostGridView>
    class DeformationFunction
        : public Dune :: DiscreteCoordFunction< double, 3, DeformationFunction<HostGridView> >
    {
        typedef DeformationFunction<HostGridView> This;
        typedef Dune :: DiscreteCoordFunction< double, 3, This > Base;

        static const int dim = HostGridView::dimension;

    public:

        DeformationFunction(const HostGridView& gridView,
                            const std::vector<RigidBodyMotion<double,3> >& deformedPosition)
            : gridView_(gridView),
              deformedPosition_(deformedPosition)
        {}

        void evaluate (const typename HostGridView::template Codim<dim>::Entity& hostEntity, unsigned int corner,
                       Dune::FieldVector<double,3> &y ) const
        {

            const typename HostGridView::IndexSet& indexSet = gridView_.indexSet();

            int idx = indexSet.index(hostEntity);
            y = deformedPosition_[idx].r;
        }

        void evaluate (const typename HostGridView::template Codim<0>::Entity& hostEntity, unsigned int corner,
                       Dune::FieldVector<double,3> &y ) const
        {

            const typename HostGridView::IndexSet& indexSet = gridView_.indexSet();

            int idx = indexSet.subIndex(hostEntity, corner,dim);

            y = deformedPosition_[idx].r;
        }

    private:

        HostGridView gridView_;

        const std::vector<RigidBodyMotion<double,3> > deformedPosition_;

    };

    template <typename Basis1, typename Basis2>
    static void downsample(const Basis1& basis1, const std::vector<RigidBodyMotion<double,3> >& v1,
                           const Basis2& basis2,       std::vector<RigidBodyMotion<double,3> >& v2)
    {
      // Embed v1 into R^7
      std::vector<Dune::FieldVector<double,7> > v1Embedded(v1.size());
      for (size_t i=0; i<v1.size(); i++)
        v1Embedded[i] = v1[i].globalCoordinates();

      // Interpolate
      BasisGridFunction<Basis1, std::vector<Dune::FieldVector<double,7> > > function(basis1, v1Embedded);
      std::vector<Dune::FieldVector<double,7> > v2Embedded;
      Functions::interpolate(basis2, v2Embedded, function);

      // Copy back from R^7 into RigidBodyMotions
      v2.resize(v2Embedded.size());
      for (size_t i=0; i<v2.size(); i++)
        v2[i] = RigidBodyMotion<double,3>(v2Embedded[i]);
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
      BasisGridFunction<Basis1, std::vector<Dune::FieldVector<double,3> > > function(basis1, v1Embedded);
      std::vector<Dune::FieldVector<double,3> > v2Embedded;
      Functions::interpolate(basis2, v2Embedded, function);

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
                      const std::vector<RigidBodyMotion<double,3> >& configuration,
                      const std::string& filename)
    {
        assert(basis.size() == configuration.size());
        auto gridView = basis.getGridView();
#if defined THIRD_ORDER  // No special handling: downsample to first order
        typedef typename GridType::LeafGridView GridView;

        const GridType& grid = basis.getGridView().grid();

        //////////////////////////////////////////////////////////////////////////////////
        //  Downsample the function onto a P1-space.  That's all we can visualize today.
        //////////////////////////////////////////////////////////////////////////////////

        typedef P1NodalBasis<GridView,double> P1Basis;
        P1Basis p1Basis(basis.getGridView());

        std::vector<RigidBodyMotion<double,3> > downsampledConfig;

        downsample(basis, configuration, p1Basis, downsampledConfig);

        //////////////////////////////////////////////////////////////////////////////////
        //  Deform the grid according to the position information in 'configuration'
        //////////////////////////////////////////////////////////////////////////////////

        typedef Dune::GeometryGrid<GridType,DeformationFunction<GridView> > DeformedGridType;

        DeformationFunction<typename GridType::LeafGridView> deformationFunction(grid.leafGridView(), downsampledConfig);

        // stupid, can't instantiate deformedGrid with a const grid
        DeformedGridType deformedGrid(const_cast<GridType&>(grid), deformationFunction);

        typedef P1NodalBasis<typename DeformedGridType::LeafGridView,double> DeformedP1Basis;
        DeformedP1Basis deformedP1Basis(deformedGrid.leafGridView());

        Dune::VTKWriter<typename DeformedGridType::LeafGridView> vtkWriter(deformedGrid.leafGridView());

        // Make three vector fields containing the directors
        typedef std::vector<Dune::FieldVector<double,3> > CoefficientType;

        std::vector<CoefficientType> directors(3);

        for (int i=0; i<3; i++) {

            directors[i].resize(downsampledConfig.size());
            for (size_t j=0; j<downsampledConfig.size(); j++)
                directors[i][j] = downsampledConfig[j].q.director(i);

            std::stringstream iAsAscii;
            iAsAscii << i;

            Dune::shared_ptr<VTKBasisGridFunction<DeformedP1Basis,CoefficientType> > vtkDirector
               = Dune::make_shared<VTKBasisGridFunction<DeformedP1Basis,CoefficientType> >
                                  (deformedP1Basis, directors[i], "director"+iAsAscii.str());
            vtkWriter.addVertexData(vtkDirector);
        }

        // For easier visualization of wrinkles: add z-coordinate as scalar field
        std::vector<double> zCoord(downsampledConfig.size());
        for (size_t i=0; i<zCoord.size(); i++)
          zCoord[i] = downsampledConfig[i].r[2];

        vtkWriter.addVertexData(zCoord, "zCoord");

        // Write the file to disk
        vtkWriter.write(filename);

#else // Write as P2 or as P1 space

        Dune::GFE::VTKFile vtkFile;

        // Count the number of elements of the different types
        std::map<Dune::GeometryType,std::size_t> numElements;
        for (const auto t : gridView.indexSet().types(0))
          numElements[t] = 0;

        for (auto it = gridView.template begin<0,Dune::Interior_Partition>(); it != gridView.template end<0,Dune::Interior_Partition>(); ++it)
          numElements[it.type()]++;

        std::size_t totalNumElements = 0;
        for (const auto nE : numElements)
          totalNumElements += nE.second;

        // Enter vertex coordinates
        std::vector<Dune::FieldVector<double, 3> > points(configuration.size());
        for (size_t i=0; i<configuration.size(); i++)
          points[i] = configuration[i].r;

        vtkFile.points_ = points;

        // Enter elements
        std::size_t connectivitySize = 0;
        for (const auto nE : numElements)
        {
#ifdef SECOND_ORDER
          if (nE.first.isQuadrilateral())
            connectivitySize += 8 * nE.second;
          else if (nE.first.isTriangle())
            connectivitySize += 6 * nE.second;
#else
          if (nE.first.isQuadrilateral())
            connectivitySize += 4 * nE.second;
          else if (nE.first.isTriangle())
            connectivitySize += 3 * nE.second;
#endif
          else
            DUNE_THROW(Dune::IOError, "Unsupported element type '" << nE.first << "' found!");
        }
        std::vector<int> connectivity(connectivitySize);

        size_t i=0;
        for (auto it = gridView.template begin<0,Dune::Interior_Partition>(); it != gridView.template end<0,Dune::Interior_Partition>(); ++it)
        {
          if (it->type().isQuadrilateral())
          {
#ifdef SECOND_ORDER
            connectivity[i++] = basis.index(*it,0);
            connectivity[i++] = basis.index(*it,2);
            connectivity[i++] = basis.index(*it,8);
            connectivity[i++] = basis.index(*it,6);

            connectivity[i++] = basis.index(*it,1);
            connectivity[i++] = basis.index(*it,5);
            connectivity[i++] = basis.index(*it,7);
            connectivity[i++] = basis.index(*it,3);
#else  // first order
            connectivity[i++] = basis.index(*it,0);
            connectivity[i++] = basis.index(*it,1);
            connectivity[i++] = basis.index(*it,3);
            connectivity[i++] = basis.index(*it,2);
#endif
          }
          if (it->type().isTriangle())
          {
#ifdef SECOND_ORDER
            connectivity[i++] = basis.index(*it,0);
            connectivity[i++] = basis.index(*it,2);
            connectivity[i++] = basis.index(*it,5);
            connectivity[i++] = basis.index(*it,1);
            connectivity[i++] = basis.index(*it,4);
            connectivity[i++] = basis.index(*it,3);
#else  // first order
            connectivity[i++] = basis.index(*it,0);
            connectivity[i++] = basis.index(*it,1);
            connectivity[i++] = basis.index(*it,2);
#endif
          }
        }

        vtkFile.cellConnectivity_ = connectivity;

        std::vector<int> offsets(totalNumElements);
        i = 0;
        int offsetCounter = 0;
        for (auto it = gridView.template begin<0,Dune::Interior_Partition>(); it != gridView.template end<0,Dune::Interior_Partition>(); ++it)
        {
          if (it->type().isQuadrilateral())
#ifdef SECOND_ORDER
            offsetCounter += 8;
#else
            offsetCounter += 4;
#endif
          if (it->type().isTriangle())
#ifdef SECOND_ORDER
            offsetCounter += 6;
#else
            offsetCounter += 3;
#endif
          offsets[i++] += offsetCounter;
        }

        vtkFile.cellOffsets_ = offsets;

        std::vector<int> cellTypes(totalNumElements);
        i = 0;
        for (auto it = gridView.template begin<0,Dune::Interior_Partition>(); it != gridView.template end<0,Dune::Interior_Partition>(); ++it)
        {
          if (it->type().isQuadrilateral())
#ifdef SECOND_ORDER
            cellTypes[i++] = 23;
#else
            cellTypes[i++] = 9;
#endif
          if (it->type().isTriangle())
#ifdef SECOND_ORDER
            cellTypes[i++] = 22;
#else
            cellTypes[i++] = 5;
#endif
        }
        vtkFile.cellTypes_ = cellTypes;

        // Z coordinate for better visualization of wrinkles
        std::vector<double> zCoord(points.size());
        for (size_t i=0; i<configuration.size(); i++)
          zCoord[i] = configuration[i].r[2];

        vtkFile.zCoord_ = zCoord;

        // The three director fields
        for (size_t i=0; i<3; i++)
        {
          vtkFile.directors_[i].resize(configuration.size());
          for (size_t j=0; j<configuration.size(); j++)
            vtkFile.directors_[i][j] = configuration[j].q.director(i);
        }

        // Actually write the VTK file to disk
        vtkFile.write(filename);
#endif
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
        auto gridView = displacementBasis.getGridView();

        // Determine order of the displacement basis
        int order = displacementBasis.getLocalFiniteElement(*gridView.template begin<0>()).localBasis().order();

        // We only do P2 spaces at the moment
        if (order != 2 and order != 3)
        {
          std::cout << "Warning: CosseratVTKWriter only supports P2 spaces -- skipping" << std::endl;
          return;
        }

        std::vector<RealTuple<double,3> > displacementConfiguration = deformationConfiguration;
        typedef typename GridType::LeafGridView GridView;
        typedef DuneFunctionsBasis<Dune::Functions::PQkNodalBasis<GridView,2> > P2DeformationBasis;
        P2DeformationBasis p2DeformationBasis(displacementBasis.getGridView());

        if (order == 3)
        {
          // resample to 2nd order -- vtk can't do anything higher
          std::vector<RealTuple<double,3> > p2DeformationConfiguration;

          downsample<DisplacementBasis,P2DeformationBasis>(displacementBasis, displacementConfiguration,
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
          writer.addArray<float>("Coordinates", 3);
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
        size_t numElements = 0;
        for (auto it = gridView.template begin<0,Dune::Interior_Partition>(); it != gridView.template end<0,Dune::Interior_Partition>(); ++it)
          numElements++;

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
        for (auto it = gridView.template begin<0, Dune::Interior_Partition>(); it != gridView.template end<0, Dune::Interior_Partition>(); ++it)
        {
          outFile << "          ";
          if (it->type().isQuadrilateral())
          {
            outFile << p2DeformationBasis.index(*it,0) << " ";
            outFile << p2DeformationBasis.index(*it,2) << " ";
            outFile << p2DeformationBasis.index(*it,8) << " ";
            outFile << p2DeformationBasis.index(*it,6) << " ";

            outFile << p2DeformationBasis.index(*it,1) << " ";
            outFile << p2DeformationBasis.index(*it,5) << " ";
            outFile << p2DeformationBasis.index(*it,7) << " ";
            outFile << p2DeformationBasis.index(*it,3) << " ";
            outFile << std::endl;
          }
        }
        outFile << "         </DataArray>" << std::endl;

        outFile << "         <DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        size_t offsetCounter = 0;
        for (auto it = gridView.template begin<0>(); it != gridView.template end<0>(); ++it)
        {
          outFile << "          ";
          if (it->type().isQuadrilateral())
            offsetCounter += 8;
          outFile << offsetCounter << std::endl;
        }
        outFile << "         </DataArray>" << std::endl;

        outFile << "         <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (auto it = gridView.template begin<0>(); it != gridView.template end<0>(); ++it)
        {
          outFile << "          ";
          if (it->type().isQuadrilateral())
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
