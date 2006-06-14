#ifndef ROD_WRITER_HH
#define ROD_WRITER_HH

#include <fstream>

#include <dune/common/exceptions.hh>

#include "configuration.hh"

/** \brief Write a planar rod
 */
void writeRod(const Dune::BlockVector<Dune::FieldVector<double,3> >& rod, 
              const std::string& filename)
{
    int nLines = rod.size() + 1 + 3*rod.size();

    // One point for each center line vertex and two for a little director at each vertex
    int nPoints = 3*rod.size();

    double directorLength = 1/((double)rod.size());

    // /////////////////////
    //   Write header
    // /////////////////////

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: Mon Jul 18 17:14:27 2005" << std::endl;
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
    for (int i=0; i<rod.size(); i++)
        outfile << i << std::endl;

    outfile << "-1" << std::endl;

    // The directors
    for (int i=0; i<rod.size(); i++) {
        outfile << rod.size()+2*i << std::endl;
        outfile << rod.size()+2*i+1 << std::endl;
        outfile << "-1" << std::endl;
    }

    // /////////////////////////////////////// 
    //   Write the vertices
    // ///////////////////////////////////////

    outfile << std::endl << "@2" << std::endl;

    // The center axis
    for (int i=0; i<rod.size(); i++)
        outfile << rod[i][0] << "  " << rod[i][1] << "  0" << std::endl;

    // The directors
    for (int i=0; i<rod.size(); i++) {

        Dune::FieldVector<double, 2> director;
        director[0] = -cos(rod[i][2]);
        director[1] = sin(rod[i][2]);
        director *= directorLength;

        outfile << rod[i][0]+director[0] << "  " << rod[i][1]+director[1] << "  0 " << std::endl;
        outfile << rod[i][0]-director[0] << "  " << rod[i][1]-director[1] << "  0 " << std::endl;

    }

    std::cout << "Result written successfully to: " << filename << std::endl;

}

/** \brief Write a spatial rod
 */
void writeRod(const std::vector<Configuration>& rod, 
              const std::string& filename)
{
    int nLines = rod.size() + 1 + 3*3*rod.size();

    // One point for each center line vertex and three for three director endpoints
    int nPoints = 4*rod.size();

    double directorLength = 1/((double)rod.size());

    // /////////////////////
    //   Write header
    // /////////////////////

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: Mon Jul 18 17:14:27 2005" << std::endl;
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
    for (int i=0; i<rod.size(); i++)
        outfile << i << std::endl;

    outfile << "-1" << std::endl;

    // The directors
    for (int i=0; i<rod.size(); i++) {
        outfile << i << std::endl << rod.size()+3*i << std::endl << "-1" << std::endl;

        outfile << i << std::endl << rod.size()+3*i+1 << std::endl << "-1" << std::endl;

        outfile << i << std::endl << rod.size()+3*i+2 << std::endl << "-1" << std::endl;
    }

    // /////////////////////////////////////// 
    //   Write the vertices
    // ///////////////////////////////////////

    outfile << std::endl << "@2" << std::endl;

    // The center axis
    for (int i=0; i<rod.size(); i++)
        outfile << rod[i].r[0] << "  " << rod[i].r[1] << "  " << rod[i].r[2] << std::endl;

    // The directors
    for (int i=0; i<rod.size(); i++) {

        Dune::FieldVector<double, 3> director[3];

        director[0] = rod[i].q.director(0);
        director[1] = rod[i].q.director(1);
        director[2] = rod[i].q.director(2);

        director[0] *= directorLength;
        director[1] *= directorLength;
        director[2] *= directorLength;

        outfile << rod[i].r[0]+director[0][0] << "  " << rod[i].r[1]+director[0][1] << "  " << rod[i].r[2]+director[0][2] << std::endl;
        outfile << rod[i].r[0]+director[1][0] << "  " << rod[i].r[1]+director[1][1] << "  " << rod[i].r[2]+director[1][2] << std::endl;
        outfile << rod[i].r[0]+director[2][0] << "  " << rod[i].r[1]+director[2][1] << "  " << rod[i].r[2]+director[2][2] << std::endl;

    }

    std::cout << "Result written successfully to: " << filename << std::endl;

}


/** \brief Write a spatial rod along with a strain or stress field
 */
template <int ndata>
void writeRod(const std::vector<Configuration>& rod, Dune::BlockVector<Dune::FieldVector<double, ndata> >& data,
              const std::string& filename)
{
    if (data.size() != rod.size()-1)
        DUNE_THROW(Dune::Exception, "Size of data vector doesn't match number of vertices");

    // #lines in center axis + #lines in set of directors
    int nLines = 3*(rod.size()-1) + 3*3*rod.size();

    // Don't ever share vertices.  That way we can simulate per-element data
    int nPoints = nLines/3*2;

    double directorLength = 1/((double)rod.size());

    // /////////////////////
    //   Write header
    // /////////////////////

    std::ofstream outfile(filename.c_str());

    outfile << "# AmiraMesh 3D ASCII 2.0" << std::endl;
    outfile << std::endl;
    outfile << "# CreationDate: Mon Jul 18 17:14:27 2005" << std::endl;
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

    for (int i=0; i<ndata; i++)
        outfile << "Vertices {float Data" << i << " } @" << 3+i << std::endl;

    outfile << std::endl;
    outfile << "# Data section follows" << std::endl;
    outfile << "@1" << std::endl;

    // ///////////////////////////////////////
    //   write lines
    // ///////////////////////////////////////

    for (int i=0; i<nLines/3; i++)
        outfile << 2*i << std::endl << 2*i+1 << std::endl << "-1" << std::endl;

    // /////////////////////////////////////// 
    //   Write the vertices
    // ///////////////////////////////////////

    outfile << std::endl << "@2" << std::endl;

    // The center axis
    for (int i=0; i<rod.size()-1; i++) {
        outfile << rod[i].r[0]   << "  " << rod[i].r[1]   << "  " << rod[i].r[2]   << std::endl;
        outfile << rod[i+1].r[0] << "  " << rod[i+1].r[1] << "  " << rod[i+1].r[2] << std::endl;
    }

    // The directors
    for (int i=0; i<rod.size(); i++) {

        Dune::FieldVector<double, 3> director[3];

        director[0] = rod[i].q.director(0);
        director[1] = rod[i].q.director(1);
        director[2] = rod[i].q.director(2);

        director[0] *= directorLength;
        director[1] *= directorLength;
        director[2] *= directorLength;

        outfile << rod[i].r[0]                << "  " << rod[i].r[1]                << "  " << rod[i].r[2]                << std::endl;
        outfile << rod[i].r[0]+director[0][0] << "  " << rod[i].r[1]+director[0][1] << "  " << rod[i].r[2]+director[0][2] << std::endl;
        outfile << rod[i].r[0]                << "  " << rod[i].r[1]                << "  " << rod[i].r[2]                << std::endl;
        outfile << rod[i].r[0]+director[1][0] << "  " << rod[i].r[1]+director[1][1] << "  " << rod[i].r[2]+director[1][2] << std::endl;
        outfile << rod[i].r[0]                << "  " << rod[i].r[1]                << "  " << rod[i].r[2]                << std::endl;
        outfile << rod[i].r[0]+director[2][0] << "  " << rod[i].r[1]+director[2][1] << "  " << rod[i].r[2]+director[2][2] << std::endl;

    }

    outfile << std::endl;

    // ///////////////////////////////////////
    //   Write data
    // ///////////////////////////////////////
    for (int i=0; i<ndata; i++) {

        outfile << "@" << 3+i << std::endl;
        
        for (int j=0; j<nPoints/2; j++) {
            double value = (j<data.size()) ? data[j][i] : 0;
            outfile << value << std::endl << value << std::endl;
        }

        outfile << std::endl;

    }

    std::cout << "Result written successfully to: " << filename << std::endl;

}

#endif
