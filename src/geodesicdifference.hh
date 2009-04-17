#ifndef ROD_DIFFERENCE_HH
#define ROD_DIFFERENCE_HH

#include "rigidbodymotion.hh"

Dune::BlockVector<Dune::FieldVector<double,6> > computeGeodesicDifference(const std::vector<RigidBodyMotion<3> >& a,
                                                                          const std::vector<RigidBodyMotion<3> >& b)
{
    if (a.size() != b.size())
        DUNE_THROW(Dune::Exception, "a and b have to have the same length!");

    Dune::BlockVector<Dune::FieldVector<double,6> > result(a.size());

    for (size_t i=0; i<result.size(); i++) {

        // Subtract centerline position
        for (int j=0; j<3; j++)
            result[i][j] = a[i].r[j] - b[i].r[j];
        
        // Subtract orientations on the tangent space of 'a'
        Dune::FieldVector<double,3> v = Rotation<3,double>::difference(a[i].q, b[i].q);

        // Compute difference on T_a SO(3)
        for (int j=0; j<3; j++)
            result[i][j+3] = v[j];

    }

    return result;
}

Dune::BlockVector<Dune::FieldVector<double,3> > computeGeodesicDifference(const std::vector<Rotation<3,double> >& a,
                                                                          const std::vector<Rotation<3,double> >& b)
{
    if (a.size() != b.size())
        DUNE_THROW(Dune::Exception, "a and b have to have the same length!");

    Dune::BlockVector<Dune::FieldVector<double,3> > result(a.size());

    for (size_t i=0; i<result.size(); i++) {

        // Subtract orientations on the tangent space of 'a'
        result[i] = Rotation<3,double>::difference(a[i], b[i]);

    }

    return result;
}

#endif
