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

      /** \brief Constructor taking the communicator rank and size */
      VTKFile(int commRank, int commSize)
      : commRank_(commRank),
        commSize_(commSize)
      {}

      /** \brief Write the file to disk */
      void write(const std::string& filename) const
      {
        std::string fullfilename = filename + ".vtu";

        // Prepend rank and communicator size to the filename, if there are more than one process
        if (commSize_ > 1)
          fullfilename = getParallelPieceName(filename, "", commRank_, commSize_);

        // Write the pvtu file that ties together the different parts
        if (commSize_> 1 && commRank_==0)
        {
          std::ofstream pvtuOutFile(getParallelName(filename, "", commSize_));
          Dune::VTK::PVTUWriter writer(pvtuOutFile, Dune::VTK::unstructuredGrid);

          writer.beginMain();

          writer.beginPointData();
          writer.addArray<float>("director0", 3);
          writer.addArray<float>("director1", 3);
          writer.addArray<float>("director2", 3);
          writer.addArray<float>("zCoord", 1);
          writer.endPointData();

          // dump point coordinates
          writer.beginPoints();
          writer.addArray<float>("Coordinates", 3);
          writer.endPoints();

          for (int i=0; i<commSize_; i++)
          writer.addPiece(getParallelPieceName(filename, "", i, commSize_));

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
        outFile << "        <DataArray type=\"Float32\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
        for (size_t i=0; i<points_.size(); i++)
          outFile << "          " << points_[i] << std::endl;
        outFile << "        </DataArray>" << std::endl;
        outFile << "      </Points>" << std::endl;

        // Write elements
        outFile << "      <Cells>" << std::endl;

        outFile << "         <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (size_t i=0; i<cellConnectivity_.size(); i++)
        {
          if ( (i%8)==0)
            outFile << std::endl << "          ";
          outFile << cellConnectivity_[i] << " ";
        }
        outFile << "         </DataArray>" << std::endl;

        outFile << "         <DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (size_t i=0; i<cellOffsets_.size(); i++)
          outFile << "          " << cellOffsets_[i] << std::endl;
        outFile << "         </DataArray>" << std::endl;

        outFile << "         <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (size_t i=0; i<cellTypes_.size(); i++)
          outFile << "          " << cellTypes_[i] << std::endl;

        outFile << "         </DataArray>" << std::endl;

        outFile << "      </Cells>" << std::endl;

        // Point data
        outFile << "      <PointData Scalars=\"zCoord\" Vectors=\"director0\">" << std::endl;

        // Z coordinate for better visualization of wrinkles
        outFile << "        <DataArray type=\"Float32\" Name=\"zCoord\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (size_t i=0; i<zCoord_.size(); i++)
          outFile << "          " << zCoord_[i] << std::endl;
        outFile << "        </DataArray>" << std::endl;

        // The three director fields
        for (size_t i=0; i<3; i++)
        {
          outFile << "        <DataArray type=\"Float32\" Name=\"director" << i <<"\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
          for (size_t j=0; j<directors_[i].size(); j++)
            outFile << "          " << directors_[i][j] << std::endl;
          outFile << "        </DataArray>" << std::endl;
        }

        outFile << "      </PointData>" << std::endl;

        // Write footer
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
#if ! HAVE_TINYXML2
        DUNE_THROW(Dune::NotImplemented, "You need TinyXML2 for vtk file reading!");
#else
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS)
          DUNE_THROW(Dune::IOError, "Couldn't open the file '" << filename << "'");

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


      int commRank_;

      // Communicator size
      int commSize_;
    };

  }

}

#endif