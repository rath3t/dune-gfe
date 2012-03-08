#ifndef VALUE_FACTORY_HH
#define VALUE_FACTORY_HH

#include <vector>

#include <dune/gfe/unitvector.hh>
#include <dune/gfe/rotation.hh>
#include <dune/gfe/rigidbodymotion.hh>

/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the generic dummy.  The actual work is done in specializations.
 */
template <class T>
class ValueFactory
{
public:
    static void get(std::vector<T>& values);
    
};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for RealTuple<1>
 */
template <>
class ValueFactory<RealTuple<double,1> >
{
public:
    static void get(std::vector<RealTuple<double,1> >& values) {
     
        int nTestPoints = 5;
        double testPoints[5] = {-3, -1, 0, 2, 4};
    
        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++)
            values[i] = RealTuple<double,1>(testPoints[i]);
        
    }
    
};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for RealTuple<3>
 */
template <>
class ValueFactory<RealTuple<double,3> >
{
public:
    static void get(std::vector<RealTuple<double,3> >& values) {
     
        int nTestPoints = 10;
        double testPoints[10][3] = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
                                    {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
                                    {-0.6,0.1,-0.2},{0.45,0.12,0.517},
                                    {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};

        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++) {
        
            Dune::FieldVector<double,3> w;
            for (int j=0; j<3; j++)
                w[j] = testPoints[i][j];
            values[i] = RealTuple<double,3>(w);

        }
        
    }
    
};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for UnitVector<2>
 */
template <>
class ValueFactory<UnitVector<double,2> >
{
public:
    static void get(std::vector<UnitVector<double,2> >& values) {
     
        int nTestPoints = 10;
        double testPoints[10][2] = {{1,0}, {0.5,0.5}, {0,1}, {-0.5,0.5}, {-1,0}, {-0.5,-0.5}, {0,-1}, {0.5,-0.5}, {0.1,1}, {1,.1}};
    
        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++) {
        
            Dune::array<double,2> w = {{testPoints[i][0], testPoints[i][1]}};
            values[i] = UnitVector<double,2>(w);

        }
        
    }
    
};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for UnitVector<3>
 */
template <>
class ValueFactory<UnitVector<double,3> >
{
public:
    static void get(std::vector<UnitVector<double,3> >& values) {
     
        int nTestPoints = 10;
        double testPoints[10][3] = {{1,0,0}, {0,1,0}, {-0.838114,0.356751,-0.412667},
                                    {-0.490946,-0.306456,0.81551},{-0.944506,0.123687,-0.304319},
                                    {-0.6,0.1,-0.2},{0.45,0.12,0.517},
                                    {-0.1,0.3,-0.1},{-0.444506,0.123687,0.104319},{-0.7,-0.123687,-0.304319}};

        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++) {
        
            Dune::array<double,3> w = {{testPoints[i][0], testPoints[i][1], testPoints[i][2]}};
            values[i] = UnitVector<double,3>(w);

        }
        
    }
    
};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for UnitVector<4>
 */
template <>
class ValueFactory<UnitVector<double,4> >
{
public:
    static void get(std::vector<UnitVector<double,4> >& values) {
     
        int nTestPoints = 10;
        double testPoints[10][4] = {{1,0,0,0}, {0,1,0,0}, {-0.838114,0.356751,-0.412667,0.5},
                                    {-0.490946,-0.306456,0.81551,0.23},{-0.944506,0.123687,-0.304319,-0.7},
                                    {-0.6,0.1,-0.2,0.8},{0.45,0.12,0.517,0},
                                    {-0.1,0.3,-0.1,0.73},{-0.444506,0.123687,0.104319,-0.23},{-0.7,-0.123687,-0.304319,0.72}};
                                  

        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++) {
        
            Dune::array<double,4> w = {{testPoints[i][0], testPoints[i][1], testPoints[i][2], testPoints[i][3]}};
            values[i] = UnitVector<double,4>(w);

        }
        
    }
    
};


/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for Rotation<3>
 */
template <>
class ValueFactory<Rotation<double,3> >
{
public:
    static void get(std::vector<Rotation<double,3> >& values) {
     
        int nTestPoints = 10;
        double testPoints[10][4] = {{1,0,0,0}, {0,1,0,0}, {-0.838114,0.356751,-0.412667,0.5},
                                    {-0.490946,-0.306456,0.81551,0.23},{-0.944506,0.123687,-0.304319,-0.7},
                                    {-0.6,0.1,-0.2,0.8},{0.45,0.12,0.517,0},
                                    {-0.1,0.3,-0.1,0.73},{-0.444506,0.123687,0.104319,-0.23},{-0.7,-0.123687,-0.304319,0.72}};
                                  

        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++) {
        
            Dune::array<double,4> w = {{testPoints[i][0], testPoints[i][1], testPoints[i][2], testPoints[i][3]}};
            values[i] = Rotation<double,3>(w);

        }
        
    }
    
};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for RigidBodyMotion<3>
 */
template <>
class ValueFactory<RigidBodyMotion<double,3> >
{
public:
    static void get(std::vector<RigidBodyMotion<double,3> >& values) {
     
        std::vector<RealTuple<double,3> > rValues;
        ValueFactory<RealTuple<double,3> >::get(rValues);
        
        std::vector<Rotation<double,3> > qValues;
        ValueFactory<Rotation<double,3> >::get(qValues);
                                  
        int nTestPoints = std::min(rValues.size(), qValues.size());
        
        values.resize(nTestPoints);
        
        // Set up elements of S^1
        for (int i=0; i<nTestPoints; i++)
            values[i] = RigidBodyMotion<double,3>(rValues[i].globalCoordinates(),qValues[i]);
        
    }
    
};

/** \brief A class that creates sets of values of various types, to be used in unit tests
 * 
 * This is the specialization for square FieldMatrices
 */
template <class T, int N>
class ValueFactory<Dune::FieldMatrix<T,N,N> >
{
public:
    static void get(std::vector<Dune::FieldMatrix<T,N,N> >& values) {
        
        int nTestPoints = 10;
        values.resize(nTestPoints);
        
        // Set up elements of T^{N \times N}
        for (int i=0; i<nTestPoints; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<N; k++)
                    values[i][j][k] = std::rand();

    }
    
};


#endif