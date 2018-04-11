// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_VERTEXNORMALS_HH
#define DUNE_GFE_VERTEXNORMALS_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/referenceelements.hh>

#include <dune/matrix-vector/crossproduct.hh>

#include <dune/gfe/unitvector.hh>

/** \brief Compute averaged vertex normals for a 2d-in-3d grid
 */
template <class GridView>
std::vector<UnitVector<typename GridView::ctype,3> > computeVertexNormals(const GridView& gridView)
{
  const auto& indexSet = gridView.indexSet();

  assert(GridView::dimension == 2);

  std::vector<Dune::FieldVector<typename GridView::ctype,3> > unscaledNormals(indexSet.size(2));
  std::fill(unscaledNormals.begin(), unscaledNormals.end(), Dune::FieldVector<typename GridView::ctype,3>(0.0));

  for (const auto& element : elements(gridView))
  {
    for (int i=0; i<element.subEntities(2); i++)
    {
      auto cornerPos = Dune::ReferenceElements<double,2>::general(element.type()).position(i,2);
      auto tangent = element.geometry().jacobianTransposed(cornerPos);
      auto cornerNormal = Dune::MatrixVector::crossProduct(tangent[0], tangent[1]);
      cornerNormal /= cornerNormal.two_norm();

      unscaledNormals[indexSet.subIndex(element,i,2)] += cornerNormal;
    }
  }

  // Normalize
  for (auto& normal : unscaledNormals)
    normal /= normal.two_norm();

  // Convert array of vectors to array of UnitVectors
  std::vector<UnitVector<typename GridView::ctype,3> > result(indexSet.size(2));

  for (std::size_t i=0; i<unscaledNormals.size(); i++)
    result[i] = unscaledNormals[i];

  return result;
}


#endif
