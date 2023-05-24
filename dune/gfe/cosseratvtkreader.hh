#ifndef DUNE_GFE_COSSERATVTKREADER_HH
#define DUNE_GFE_COSSERATVTKREADER_HH

#include <dune/gfe/spaces/rigidbodymotion.hh>

namespace Dune
{
  namespace GFE
  {

    /** \brief Read configurations of Cosserat models from VTK files into memory */
    class CosseratVTKReader
    {
    public:

      static void read(std::vector<RigidBodyMotion<double,3> >& configuration,
                       const std::string& filename)
      {
        VTKFile vtkFile;
        vtkFile.read(filename);

        configuration.resize(vtkFile.points_.size());

        for (size_t i=0; i<configuration.size(); i++)
        {
          configuration[i].r = vtkFile.points_[i];

          FieldMatrix<double,3,3> R;
          for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
              R[j][k] = vtkFile.directors_[k][i][j];

          configuration[i].q.set(R);
        }

      }

    };

  }

}

#endif
