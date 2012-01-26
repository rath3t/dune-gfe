#ifndef COSSERAT_VTK_WRITER_HH
#define COSSERAT_VTK_WRITER_HH

#include <dune/grid/geometrygrid.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>
#include <dune/gfe/rigidbodymotion.hh>


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

    
public:
    static void write(const GridType& grid,
                      const std::vector<RigidBodyMotion<double,3> >& configuration,
                      const std::string& filename)
    {

        typedef Dune::GeometryGrid<GridType,DeformationFunction<typename GridType::LeafGridView> > DeformedGridType;
    
        DeformationFunction<typename GridType::LeafGridView> deformationFunction(grid.leafView(), configuration);
    
        // stupid, can't instantiate deformedGrid with a const grid
        DeformedGridType deformedGrid(const_cast<GridType&>(grid), deformationFunction);

        typedef P1NodalBasis<typename DeformedGridType::LeafGridView,double> P1Basis;
        P1Basis p1Basis(deformedGrid.leafView());

        Dune::VTKWriter<typename DeformedGridType::LeafGridView> vtkWriter(deformedGrid.leafView());
    
        // Make three vector fields containing the directors
        typedef std::vector<Dune::FieldVector<double,3> > CoefficientType;
        
        std::vector<CoefficientType> directors(3);
        
        for (int i=0; i<3; i++) {
     
            directors[i].resize(configuration.size());
            for (size_t j=0; j<configuration.size(); j++)
                directors[i][j] = configuration[j].q.director(i);
        
            std::stringstream iAsAscii;
            iAsAscii << i;
        
            Dune::shared_ptr<VTKBasisGridFunction<P1Basis,CoefficientType> > vtkDirector
               = Dune::shared_ptr<VTKBasisGridFunction<P1Basis,CoefficientType> >
                   (new VTKBasisGridFunction<P1Basis,CoefficientType>(p1Basis, directors[i], "director"+iAsAscii.str()));
            vtkWriter.addVertexData(vtkDirector);
        }
        
        vtkWriter.write(filename);
    
    }    

};

#endif
