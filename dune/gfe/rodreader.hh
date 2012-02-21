#ifndef ROD_READER_HH
#define ROD_READER_HH

#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>

#include "rigidbodymotion.hh"

class RodReader
{
public:

    /** \brief Read a spatial rod from file
     */
    static void readRod(std::vector<RigidBodyMotion<double,3> >& rod, 
            const std::string& filename)
    {

        // /////////////////////////////////////////////////////
        // Load the AmiraMesh file
        // ////////////////////////////////////////////////

        AmiraMesh* am = AmiraMesh::read(filename.c_str());

        if(!am)
            DUNE_THROW(Dune::IOError, "Could not open AmiraMesh file: " << filename);

        int i, j;

        bool datafound=false;

        // get the translation
        AmiraMesh::Data* am_ValueData =  am->findData("Nodes", HxFLOAT, 3, "Coordinates");
        if (am_ValueData) {
            datafound = true;

            if (rod.size()<am->nElements("Nodes"))
                DUNE_THROW(Dune::IOError, "When reading data from a surface field the "
                        << "array you provide has to have at least the size of the surface!");

            float* am_values_float = (float*) am_ValueData->dataPtr();

            for (i=0; i<am->nElements("Nodes"); i++) {
                for (j=0; j<3; j++)
                    rod[i].r[j] = am_values_float[i*3+j];
            }

        } else
            DUNE_THROW(Dune::IOError, "Couldn't find Coordinates: " << filename);

        // get the rotation
        Dune::BlockVector<Dune::FieldVector<double,3> > dir0(rod.size()),dir1(rod.size());
        am_ValueData =  am->findData("Nodes", HxFLOAT, 3, "Directors0");
        if (am_ValueData) {
            datafound = true;

            if (rod.size()<am->nElements("Nodes"))
                DUNE_THROW(Dune::IOError, "When reading data from a surface field the "
                        << "array you provide has to have at least the size of the surface!");

            float* am_values_float = (float*) am_ValueData->dataPtr();

            for (i=0; i<am->nElements("Nodes"); i++) {
                for (j=0; j<3; j++)
                    dir0[i][j] = am_values_float[i*3+j];
            }

        } else 
            DUNE_THROW(Dune::IOError, "Couldn't find Directors0: " << filename);

        am_ValueData =  am->findData("Nodes", HxFLOAT, 3, "Directors1");
        if (am_ValueData) {
            datafound = true;

            if (rod.size()<am->nElements("Nodes"))
                DUNE_THROW(Dune::IOError, "When reading data from a surface field the "
                        << "array you provide has to have at least the size of the surface!");

            float* am_values_float = (float*) am_ValueData->dataPtr();

            for (i=0; i<am->nElements("Nodes"); i++) {
                for (j=0; j<3; j++)
                    dir1[i][j] = am_values_float[i*3+j];
            }

        } else 
            DUNE_THROW(Dune::IOError, "Couldn't find Directors1: " << filename);


        for (i=0; i<rod.size(); i++) {
            dir0[i] /= dir0[i].two_norm(); 
            dir1[i] /= dir1[i].two_norm(); 
            
            // compute third director as the outer normal of the cross-section
            Dune::FieldVector<double,3> dir2;
            dir2[0] = dir0[i][1]*dir1[i][2]-dir0[i][2]*dir1[i][1];    
            dir2[1] = dir0[i][2]*dir1[i][0]-dir0[i][0]*dir1[i][2];    
            dir2[2] = dir0[i][0]*dir1[i][1]-dir0[i][1]*dir1[i][0];    
            dir2 /= dir2.two_norm();

            // the rotation matrix corresponding to the directors
            Dune::FieldMatrix<double,3,3> rot(0);
            for (j=0;j<3;j++) {
                rot[j][0] = dir0[i][j];
                rot[j][1] = dir1[i][j];
                rot[j][2] = dir2[j];
            }
            rod[i].q.set(rot);
        }

        std::cout << "Rod successfully read from: " << filename << std::endl;

    }
};
#endif
