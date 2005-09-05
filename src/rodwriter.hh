#ifndef ROD_WRITER_HH
#define ROD_WRITER_HH


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

#endif
