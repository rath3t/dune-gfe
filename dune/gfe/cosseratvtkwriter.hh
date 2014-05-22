#ifndef COSSERAT_VTK_WRITER_HH
#define COSSERAT_VTK_WRITER_HH

#include <dune/grid/geometrygrid.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/fufem/functionspacebases/p1nodalbasis.hh>
#include <dune/fufem/functions/vtkbasisgridfunction.hh>
#include <dune/fufem/functiontools/basisinterpolator.hh>
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

public:

    /** \brief Write a Cosserat configuration given as vertex data
     */
    static void write(const GridType& grid,
                      const std::vector<RigidBodyMotion<double,3> >& configuration,
                      const std::string& filename)
    {

        typedef Dune::GeometryGrid<GridType,DeformationFunction<typename GridType::LeafGridView> > DeformedGridType;

        DeformationFunction<typename GridType::LeafGridView> deformationFunction(grid.leafGridView(), configuration);

        // stupid, can't instantiate deformedGrid with a const grid
        DeformedGridType deformedGrid(const_cast<GridType&>(grid), deformationFunction);

        typedef P1NodalBasis<typename DeformedGridType::LeafGridView,double> P1Basis;
        P1Basis p1Basis(deformedGrid.leafGridView());

        Dune::VTKWriter<typename DeformedGridType::LeafGridView> vtkWriter(deformedGrid.leafGridView());

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
               = Dune::make_shared<VTKBasisGridFunction<P1Basis,CoefficientType> >
                                  (p1Basis, directors[i], "director"+iAsAscii.str());
            vtkWriter.addVertexData(vtkDirector);
        }

        vtkWriter.write(filename);

    }

    /** \brief Write a configuration given with respect to a scalar function space basis
     */
    template <typename Basis>
    static void write(const Basis& basis,
                      const std::vector<RigidBodyMotion<double,3> >& configuration,
                      const std::string& filename)
    {
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

    }

};

#endif
