#ifndef COSSERAT_AMIRAMESH_WRITER_HH
#define COSSERAT_AMIRAMESH_WRITER_HH

#include <dune/grid/geometrygrid.hh>
#include <dune/grid/io/file/amirameshwriter.hh>

#include <dune/gfe/rigidbodymotion.hh>


/** \brief Write the configuration of a Cosserat material in AmiraMesh format */
template <class GridType>
class CosseratAmiraMeshWriter
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
                            const std::vector<RigidBodyMotion<3> >& deformedPosition)
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

        const std::vector<RigidBodyMotion<3> > deformedPosition_;

    };

    
public:
    static void write(const GridType& grid,
                      const std::vector<RigidBodyMotion<3> >& configuration,
                      const std::string& filePrefix)
    {

        typedef Dune::GeometryGrid<GridType,DeformationFunction<typename GridType::LeafGridView> > DeformedGridType;
    
        DeformationFunction<typename GridType::LeafGridView> deformationFunction(grid.leafView(), configuration);
    
        // stupid, can't instantiate deformedGrid with a const grid
        DeformedGridType deformedGrid(const_cast<GridType&>(grid), deformationFunction);

        if (dim==2)
            Dune::LeafAmiraMeshWriter<DeformedGridType>::writeSurfaceGrid(deformedGrid.leafView(), filePrefix + "Grid");
        else {
            Dune::LeafAmiraMeshWriter<DeformedGridType> amiramesh(deformedGrid);
            amiramesh.write(filePrefix + "Grid");
        }
    
        // Make three vector fields containing the directors
        // I don't think there is a simpler way to get the data into vanilla Amira
    
        for (int i=0; i<3; i++) {
     
            std::vector<Dune::FieldVector<double,3> > director(configuration.size());
            for (size_t j=0; j<configuration.size(); j++)
                director[j] = configuration[j].q.director(i);
        
            Dune::LeafAmiraMeshWriter<DeformedGridType> amiramesh;
            amiramesh.addVertexData(director, deformedGrid.leafView());
        
            std::stringstream iAsAscii;
            iAsAscii << i;
            amiramesh.write(filePrefix + "Orientation"+iAsAscii.str(), true);
        
        }
    
    }    

};

#endif
