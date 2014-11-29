#ifndef DUNE_GFE_VTKFILE_HH
#define DUNE_GFE_VTKFILE_HH

#include <vector>
#include <fstream>

#if HAVE_TINYXML2
// For VTK file reading
#include <tinyxml2.h>
#endif

#include <dune/common/fvector.hh>

// For parallel infrastructure stuff:
#include <dune/grid/io/file/vtk.hh>

namespace Dune {

  namespace GFE {

    /** \brief A class representing a VTK file, but independent from the Dune grid interface
     *
     * This file is supposed to represent an abstraction layer in between the pure XML used for VTK files,
     * and the VTKWriter from dune-grid, which knows about grids.  In theory, the dune-grid VTKWriter
     * could use this class to simplify its own code.  More importantly, the VTKFile class allows to
     * write files containing second-order geometries, which is currently not possible with the dune-grid
     * VTKWriter.
     */
    class VTKFile
    {

    public:

      /** \brief Write the file to disk */
      void write(const std::string& filename) const
      {
        int argc = 0;
        char** argv;
        Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);

        std::string fullfilename = filename + ".vtu";

        // Prepend rank and communicator size to the filename, if there are more than one process
        if (mpiHelper.size() > 1)
          fullfilename = getParallelPieceName(filename, "", mpiHelper.rank(), mpiHelper.size());

        // Write the pvtu file that ties together the different parts
        if (mpiHelper.size() > 1 && mpiHelper.rank()==0)
        {
          std::ofstream pvtuOutFile(getParallelName(filename, "", mpiHelper.size()));
          Dune::VTK::PVTUWriter writer(pvtuOutFile, Dune::VTK::unstructuredGrid);

          writer.beginMain();

          writer.beginPointData();
          writer.addArray<float>("director0", 3);
          writer.addArray<float>("director1", 3);
          writer.addArray<float>("director2", 3);
          writer.addArray<float>("zCoord", 1);
          writer.endPointData();

          writer.beginCellData();
          writer.addArray<float>("mycelldata", 1);
          writer.endCellData();

          // dump point coordinates
          writer.beginPoints();
          writer.addArray<float>("Coordinates", 3);
          writer.endPoints();

          for (int i=0; i<mpiHelper.size(); i++)
          writer.addPiece(getParallelPieceName(filename, "", i, mpiHelper.size()));

          // finish main section
          writer.endMain();
        }

        std::ofstream outFile(fullfilename);

        // Write header
        outFile << "<?xml version=\"1.0\"?>" << std::endl;
        outFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
        outFile << "  <UnstructuredGrid>" << std::endl;
        outFile << "    <Piece NumberOfCells=\"" << cellOffsets_.size() << "\" NumberOfPoints=\"" << points_.size() << "\">" << std::endl;

        // Write vertex coordinates
        outFile << "      <Points>" << std::endl;
        {  // extra parenthesis to control destruction of the pointsWriter object
          Dune::VTK::AsciiDataArrayWriter<float> pointsWriter(outFile, "Coordinates", 3, Dune::Indent(4));
          for (size_t i=0; i<points_.size(); i++)
            for (int j=0; j<3; j++)
              pointsWriter.write(points_[i][j]);
        }  // destructor of pointsWriter objects writes trailing </DataArray> to file
        outFile << "      </Points>" << std::endl;

        // Write elements
        outFile << "      <Cells>" << std::endl;
        {  // extra parenthesis to control destruction of the cellConnectivityWriter object
          Dune::VTK::AsciiDataArrayWriter<int> cellConnectivityWriter(outFile, "connectivity", 1, Dune::Indent(4));
          for (size_t i=0; i<cellConnectivity_.size(); i++)
            cellConnectivityWriter.write(cellConnectivity_[i]);
        }

        {  // extra parenthesis to control destruction of the writer object
          Dune::VTK::AsciiDataArrayWriter<int> cellOffsetsWriter(outFile, "offsets", 1, Dune::Indent(4));
          for (size_t i=0; i<cellOffsets_.size(); i++)
            cellOffsetsWriter.write(cellOffsets_[i]);
        }

        {  // extra parenthesis to control destruction of the writer object
          Dune::VTK::AsciiDataArrayWriter<unsigned int> cellTypesWriter(outFile, "types", 1, Dune::Indent(4));
          for (size_t i=0; i<cellTypes_.size(); i++)
            cellTypesWriter.write(cellTypes_[i]);
        }

        outFile << "      </Cells>" << std::endl;

        //////////////////////////////////////////////////
        //   Point data
        //////////////////////////////////////////////////
        outFile << "      <PointData Scalars=\"zCoord\" Vectors=\"director0\">" << std::endl;

        // Z coordinate for better visualization of wrinkles
        {  // extra parenthesis to control destruction of the writer object
          Dune::VTK::AsciiDataArrayWriter<float> zCoordWriter(outFile, "zCoord", 1, Dune::Indent(4));
          for (size_t i=0; i<zCoord_.size(); i++)
            zCoordWriter.write(zCoord_[i]);
        }

        // The three director fields
        for (size_t i=0; i<3; i++)
        {
          Dune::VTK::AsciiDataArrayWriter<float> directorWriter(outFile, "director" + std::to_string(i), 3, Dune::Indent(4));
          for (size_t j=0; j<directors_[i].size(); j++)
            for (int k=0; k<3; k++)
              directorWriter.write(directors_[i][j][k]);
        }

        outFile << "      </PointData>" << std::endl;

        //////////////////////////////////////////////////
        //   Cell data
        //////////////////////////////////////////////////

        if (cellData_.size() > 0)
        {
          outFile << "      <CellData>" << std::endl;
          {  // extra parenthesis to control destruction of the writer object
            Dune::VTK::AsciiDataArrayWriter<float> cellDataWriter(outFile, "mycelldata", 1, Dune::Indent(4));
            for (size_t i=0; i<cellData_.size(); i++)
              cellDataWriter.write(cellData_[i]);
          }
          outFile << "      </CellData>" << std::endl;
        }

        //////////////////////////////////////////////////
        //   Write footer
        //////////////////////////////////////////////////
        outFile << "    </Piece>" << std::endl;
        outFile << "  </UnstructuredGrid>" << std::endl;
        outFile << "</VTKFile>" << std::endl;


      }

      /** \brief Read a VTKFile object from a file
       *
       * This method uses TinyXML2.  If you do not have TinyXML2 installed the code will simply throw a NotImplemented exception.
       */
      void read(const std::string& filename)
      {
        int argc = 0;
        char** argv;
        Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc,argv);

        std::string fullfilename = filename + ".vtu";

        // Prepend rank and communicator size to the filename, if there are more than one process
        if (mpiHelper.size() > 1)
          fullfilename = getParallelPieceName(filename, "", mpiHelper.rank(), mpiHelper.size());

#if ! HAVE_TINYXML2
        DUNE_THROW(Dune::NotImplemented, "You need TinyXML2 for vtk file reading!");
#else
        tinyxml2::XMLDocument doc;
        if (int error = doc.LoadFile(fullfilename.c_str()) != tinyxml2::XML_SUCCESS)
        {
          std::cout << "Error: " << error << std::endl;
          DUNE_THROW(Dune::IOError, "Couldn't open the file '" << fullfilename << "'");
        }

        // Get number of cells and number of points
        tinyxml2::XMLElement* pieceElement = doc.FirstChildElement( "VTKFile" )->FirstChildElement( "UnstructuredGrid" )->FirstChildElement( "Piece" );

        int numberOfCells;
        int numberOfPoints;
        pieceElement->QueryIntAttribute( "NumberOfCells", &numberOfCells );
        pieceElement->QueryIntAttribute( "NumberOfPoints", &numberOfPoints );

        //////////////////////////////////
        //  Read point coordinates
        //////////////////////////////////
        points_.resize(numberOfPoints);
        tinyxml2::XMLElement* pointsElement = pieceElement->FirstChildElement( "Points" )->FirstChildElement( "DataArray" );

        std::stringstream coordsStream(pointsElement->GetText());

        double inDouble;
        size_t i=0;
        while (coordsStream >> inDouble)
        {
          points_[i/3][i%3] = inDouble;
          i++;
        }

        ///////////////////////////////////
        //  Read cells
        ///////////////////////////////////
        tinyxml2::XMLElement* cellsElement = pieceElement->FirstChildElement( "Cells" )->FirstChildElement( "DataArray" );

        for (int i=0; i<3; i++)
        {
          int inInt;
          if (cellsElement->Attribute("Name", "connectivity"))
          {
            std::stringstream connectivityStream(cellsElement->GetText());
            while (connectivityStream >> inInt)
              cellConnectivity_.push_back(inInt);
          }
          if (cellsElement->Attribute("Name", "offsets"))
          {
            std::stringstream offsetsStream(cellsElement->GetText());
            while (offsetsStream >> inInt)
              cellOffsets_.push_back(inInt);
          }
          if (cellsElement->Attribute("Name", "types"))
          {
            std::stringstream typesStream(cellsElement->GetText());
            while (typesStream >> inInt)
              cellTypes_.push_back(inInt);
          }

          cellsElement = cellsElement->NextSiblingElement();
        }

        /////////////////////////////////////
        //  Read point data
        /////////////////////////////////////

        tinyxml2::XMLElement* pointDataElement = pieceElement->FirstChildElement( "PointData" )->FirstChildElement( "DataArray" );

        for (int i=0; i<4; i++)
        {
          size_t j=0;
          std::stringstream directorStream(pointDataElement->GetText());
          if (pointDataElement->Attribute("Name", "director0"))
          {
            directors_[0].resize(numberOfPoints);
            while (directorStream >> inDouble)
            {
              directors_[0][j/3][j%3] = inDouble;
              j++;
            }
          }
          if (pointDataElement->Attribute("Name", "director1"))
          {
            directors_[1].resize(numberOfPoints);
            while (directorStream >> inDouble)
            {
              directors_[1][j/3][j%3] = inDouble;
              j++;
            }
          }
          if (pointDataElement->Attribute("Name", "director2"))
          {
            directors_[2].resize(numberOfPoints);
            while (directorStream >> inDouble)
            {
              directors_[2][j/3][j%3] = inDouble;
              j++;
            }
          }

          pointDataElement = pointDataElement->NextSiblingElement();
        }
#endif
      }

      std::vector<Dune::FieldVector<double,3> > points_;

      std::vector<int> cellConnectivity_;

      std::vector<int> cellOffsets_;

      std::vector<int> cellTypes_;

      std::vector<double> zCoord_;

      std::vector<double> cellData_;

      std::array<std::vector<Dune::FieldVector<double,3> >, 3 > directors_;

    private:

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
        if (true) //(GridType::dimension > 1)
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
        if (true) //(GridType::dimension > 1)
          s << ".pvtu";
        else
          s << ".pvtp";
        return s.str();
      }

    };

  }

}

#endif
