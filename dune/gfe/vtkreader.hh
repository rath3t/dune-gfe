// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_VTKREADER_HH
#define DUNE_GFE_VTKREADER_HH

#include <memory>

#include <dune/gfe/vtkfile.hh>

/** \brief Read a grid from a VTK file
 */
template <class GridType>
class VTKReader
{
public:

  /** \brief Read a grid from a VTK file */
  static std::unique_ptr<GridType> read(std::string filename)
  {
    constexpr auto dimworld = GridType::dimensionworld;

    Dune::GFE::VTKFile vtkFile;

    // Read test file from disk
    vtkFile.read(filename);

    Dune::GridFactory<GridType> factory;

    for (const auto& v : vtkFile.points_)
    {
      // use the first dimworld components as vertex coordinate; discard the rest
      Dune::FieldVector<typename GridType::ctype, dimworld> pos;
      for (int i=0; i<dimworld; i++)
        pos[i] = v[i];
      factory.insertVertex(pos);
    }

    for (size_t i=0; i<vtkFile.cellConnectivity_.size(); i+=3)
    {
      factory.insertElement(Dune::GeometryTypes::triangle, {vtkFile.cellConnectivity_[i],
                                       vtkFile.cellConnectivity_[i+1],
                                       vtkFile.cellConnectivity_[i+2]});

    }

    return std::unique_ptr<GridType>(factory.createGrid());
  }

};

#endif
