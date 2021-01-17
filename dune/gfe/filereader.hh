#ifndef DUNE_GFE_FILEREADER_HH
#define DUNE_GFE_FILEREADER_HH

#include <iostream>
#include <fstream>

#include <dune/common/fvector.hh>
// This is a custom file format parser which reads in a grid deformation file
// The file must contain lines of the format <grid vertex>:<displacement or rotation>
// !!! THIS IS A TEMPORARY SOLUTION AND WILL BE REPLACED BY THE Dune::VtkReader in dune-vtk/dune/vtk/vtkreader.hh !!!
namespace Dune {
  namespace GFE {
    // Convert the pairs {grid vertex, vector of dimension d} in the given file to a map
    template <int d>
    static std::unordered_map<std::string, FieldVector<double,d>> transformFileToMap(std::string pathToFile) {

        std::unordered_map<std::string, FieldVector<double,d>> map;

        std::string line, displacement, entry;

        std::ifstream file(pathToFile, std::ios::in);

        if (file.is_open()) {
          while (std::getline(file, line)) {
            size_t j = 0;
            size_t pos = line.find(":");
            displacement = line.substr(pos + 1);
            line.erase(pos);
            std::stringstream entries(displacement);
            FieldVector<double,d> vector(0);
            while(entries >> entry)
              vector[j++] = std::stod(entry);
            map.insert({line,vector});
          }
          file.close();
        } else {
          DUNE_THROW(Exception, "Error: Could not open the file " + pathToFile + " !");
        }
      return map;
    }
  }
}
#endif