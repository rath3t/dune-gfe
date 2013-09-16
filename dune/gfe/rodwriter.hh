#ifndef ROD_WRITER_HH
#define ROD_WRITER_HH

#include <fstream>
#include <vector>
#include <ctime>

#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <dune/solvers/common/numproc.hh>

#include "rigidbodymotion.hh"

class RodWriter
{
public:
    
    static void writeBinary(const std::vector<RigidBodyMotion<double,3> >& rod, 
                            const std::string& filename)
    {
        FILE* fpRod = fopen(filename.c_str(), "wb");
        if (!fpRod)
            DUNE_THROW(SolverError, "Couldn't open file " << filename << " for writing");
            
        for (size_t j=0; j<rod.size(); j++) {

            for (int k=0; k<3; k++)
                fwrite(&rod[j].r[k], sizeof(double), 1, fpRod);

            for (int k=0; k<4; k++)  // 3d hardwired here!
                fwrite(&rod[j].q[k], sizeof(double), 1, fpRod);

        }

        fclose(fpRod);
        
    }
    
};

/** \brief Write a planar rod
 */
void writeRod(const std::vector<RigidBodyMotion<double,2> >& rod, 
              const std::string& filename)
{
    int nLines = rod.size() + 1 + 3*rod.size();

    // One point for each center line vertex and two for a little director at each vertex
    int nPoints = 3*rod.size();

    double directorLength = 1/((double)rod.size());

    // /////////////////////
    //   Write header
    // /////////////////////
    
    time_t rawtime;
    time (&rawtime);

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: " << ctime(&rawtime) << std::endl;
    outfile << std::endl;
    outfile << std::endl;
    outfile << "define Lines " << nLines << std::endl;
    outfile << "nVertices " << nPoints << std::endl;
    outfile << std::endl;
    outfile << "Parameters {" << std::endl;
    outfile << "    ContentType \"HxLineSet\"" << std::endl;
    outfile << "}" << std::endl;
    outfile << std::endl;
    outfile << "Lines { int LineIdx } @1" << std::endl;
    outfile << "Vertices { float[3] Coordinates } @2" << std::endl;
    outfile << std::endl;
    outfile << "# Data section follows" << std::endl;
    outfile << "@1" << std::endl;

    // ///////////////////////////////////////
    //   write lines
    // ///////////////////////////////////////

    // The center axis
    for (size_t i=0; i<rod.size(); i++)
        outfile << i << std::endl;

    outfile << "-1" << std::endl;

    // The directors
    for (size_t i=0; i<rod.size(); i++) {
        outfile << rod.size()+2*i << std::endl;
        outfile << rod.size()+2*i+1 << std::endl;
        outfile << "-1" << std::endl;
    }

    // /////////////////////////////////////// 
    //   Write the vertices
    // ///////////////////////////////////////

    outfile << std::endl << "@2" << std::endl;

    // The center axis
    for (size_t i=0; i<rod.size(); i++)
        outfile << rod[i].r[0] << "  " << rod[i].r[1] << "  0" << std::endl;

    // The directors
    for (size_t i=0; i<rod.size(); i++) {

        Dune::FieldVector<double, 2> director;
        director[0] = -cos(rod[i].q.angle_);
        director[1] = sin(rod[i].q.angle_);
        director *= directorLength;

        outfile << rod[i].r[0]+director[0] << "  " << rod[i].r[1]+director[1] << "  0 " << std::endl;
        outfile << rod[i].r[0]-director[0] << "  " << rod[i].r[1]-director[1] << "  0 " << std::endl;

    }

    std::cout << "Result written successfully to: " << filename << std::endl;

}

/** \brief Write a spatial rod
 */
void writeRod(const std::vector<RigidBodyMotion<double,3> >& rod, 
              const std::string& filename,
              double radius = 1.0)
{
    int nPoints = rod.size();

    // /////////////////////
    //   Write header
    // /////////////////////

    time_t rawtime;
    time (&rawtime);

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: " << ctime(&rawtime) << std::endl;
    outfile << std::endl;
    outfile << std::endl;
    outfile << "nNodes " << nPoints << std::endl;
    outfile << std::endl;
    outfile << "Parameters {" << std::endl;
    outfile << "    ContentType \"Rod\"" << std::endl;
    outfile << "}" << std::endl;
    outfile << std::endl;
    outfile << "Nodes { float[3] Coordinates } @1" << std::endl;
    outfile << "Nodes { float[3] Directors0 } @2" << std::endl;
    outfile << "Nodes { float[3] Directors1 } @3" << std::endl;
    outfile << std::endl;
    outfile << "# Data section follows" << std::endl;

    // /////////////////////////////////////// 
    //   Write the center axis
    // ///////////////////////////////////////
    outfile << "@1" << std::endl;

    for (size_t i=0; i<rod.size(); i++)
        //outfile << rod[i].r[0] << "  " << rod[i].r[1] << "  " << rod[i].r[2] << std::endl;
        outfile << rod[i].r << std::endl;


    // /////////////////////////////////////// 
    //   Write the directors
    // ///////////////////////////////////////
    outfile << std::endl << "@2" << std::endl;

    for (size_t i=0; i<rod.size(); i++) {
        Dune::FieldVector<double,3> dir = rod[i].q.director(0);
        dir *= radius; 
        outfile << dir << std::endl;
    }

    outfile << std::endl << "@3" << std::endl;

    for (size_t i=0; i<rod.size(); i++) {
        Dune::FieldVector<double,3> dir = rod[i].q.director(1);
        dir *= radius; 
        outfile << dir << std::endl;
    }

    std::cout << "Result written successfully to: " << filename << std::endl;

}

/** \brief Write a spatial rod along with a strain or stress field
 */
void writeRodElementData(Dune::BlockVector<Dune::FieldVector<double, 1> >& data,
                         const std::string& filename)
{
    // /////////////////////
    //   Write header
    // /////////////////////

    time_t rawtime;
    time (&rawtime);

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: " << ctime(&rawtime) << std::endl;
    outfile << std::endl;
    outfile << std::endl;
    outfile << "define Segments " << data.size() << std::endl;
    outfile << std::endl;
    outfile << "Parameters {" << std::endl;
    outfile << "    ContentType \"ScalarRodField\"" << std::endl;
    outfile << "    Encoding    \"OnSegments\"" << std::endl;
    outfile << "}" << std::endl;
    outfile << std::endl;
    outfile << "Segments { float Data } @1" << std::endl;
    outfile << std::endl;

    outfile << "# Data section follows" << std::endl;
    outfile << "@1" << std::endl;

    // ///////////////////////////////////////
    //   write data
    // ///////////////////////////////////////

    for (size_t i=0; i<data.size(); i++)
        outfile << data[i] << std::endl;

    std::cout << "Result written successfully to: " << filename << std::endl;

}

void writeRodStress(Dune::BlockVector<Dune::FieldVector<double, 6> >& data,
                    const std::string& basename)
{
    Dune::BlockVector<Dune::FieldVector<double, 1> > scalarStress(data.size());

    // Extract separate stress values and write them
    for (int i=0; i<6; i++) {

        for (size_t j=0; j<data.size(); j++) 
            scalarStress[j] = data[j][i];

        // create a name
        std::stringstream iAsAscii;
        iAsAscii << i;

        // write the scalar field
        writeRodElementData(scalarStress, basename + ".stress" + iAsAscii.str());

    }

}

/** \brief Write a spatial rod bundle
 */
void writeRodBundle(const std::vector<std::vector<RigidBodyMotion<double,3> > >& rods,
              const std::string& filename,
              double radius = 1.0)
{
    int nRods = rods.size();

    int nPoints(0);
    for (int i=0; i<nRods; i++)
        nPoints += rods[i].size();

    // /////////////////////
    //   Write header
    // /////////////////////

    time_t rawtime;
    time (&rawtime);

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: " << ctime(&rawtime) << std::endl;
    outfile << std::endl;
    outfile << std::endl;
    outfile << "define Rods " << nRods << std::endl;
    outfile << "nNodes " << nPoints << std::endl;
    outfile << std::endl;
    outfile << "Parameters {" << std::endl;
    outfile << "    ContentType \"RodBundle\"" << std::endl;
    outfile << "}" << std::endl;
    outfile << std::endl;
    outfile << "Rods { int Vertices } @1" << std::endl;
    outfile << "Nodes { float[3] Coordinates } @2" << std::endl;
    outfile << "Nodes { float[3] Directors0 } @3" << std::endl;
    outfile << "Nodes { float[3] Directors1 } @4" << std::endl;
    outfile << std::endl;
    outfile << "# Data section follows" << std::endl;

    // ///////////////////////////////////////
    //   Write the center axis
    // ///////////////////////////////////////
    outfile << "@1" << std::endl;

    for (size_t i=0; i<rods.size(); i++)
        outfile << rods[i].size() << std::endl;
        //outfile << rod[i].r[0] << "  " << rod[i].r[1] << "  " << rod[i].r[2] << std::endl;

    outfile << std::endl << "@2" << std::endl;
    for (size_t i=0; i<rods.size(); i++)
        for (size_t j=0; j<rods[i].size(); j++)
        outfile << rods[i][j].r << std::endl;


    // ///////////////////////////////////////
    //   Write the directors
    // ///////////////////////////////////////
    outfile << std::endl << "@3" << std::endl;

    for (size_t i=0; i<rods.size(); i++)
        for (size_t j=0; j<rods[i].size(); j++) {
            Dune::FieldVector<double,3> dir = rods[i][j].q.director(0);
            dir *= radius;
            outfile << dir << std::endl;
        }

    outfile << std::endl << "@4" << std::endl;

    for (size_t i=0; i<rods.size(); i++)
        for (size_t j=0; j<rods[i].size(); j++) {
            Dune::FieldVector<double,3> dir = rods[i][j].q.director(1);
            dir *= radius;
            outfile << dir << std::endl;
    }

    std::cout << "Result written successfully to: " << filename << std::endl;

}


#endif
