// $Id: localstiffness.hh 560 2009-06-10 09:32:22Z sander $

#ifndef DUNE_LOCALSTIFFNESS_HH
#define DUNE_LOCALSTIFFNESS_HH

#include<iostream>
#include<vector>
#include<set>
#include<map>
#include<stdio.h>
#include<stdlib.h>

#include<dune/common/timer.hh>
#include<dune/common/fvector.hh>
#include<dune/common/fmatrix.hh>
#include<dune/common/array.hh>
#include<dune/common/exceptions.hh>
#include<dune/common/geometrytype.hh>
#include<dune/grid/common/grid.hh>
#include<dune/istl/operators.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/matrix.hh>

/**
 * @file
 * @brief  defines a class for piecewise linear finite element functions
 * @author Peter Bastian
 */

/*! @defgroup DISC_Operators Operators
  @ingroup DISC
  @brief
  
  @section D1 Introduction
  <!--=================-->
  
  To be written
*/

namespace Dune
{
  /** @addtogroup DISC_Operators
   *
   * @{
   */
  /**
   * @brief base class for assembling local stiffness matrices
   *
   */


  /*! @brief Base class for local assemblers

  This class serves as a base class for local assemblers. It provides
  space and access to the local stiffness matrix. The actual assembling is done
  in a derived class via the virtual assemble method.

  \tparam GV    A grid view type
  \tparam RT   The field type used in the elements of the stiffness matrix
  \tparam m    number of degrees of freedom per node (system size)
   */
  template<class GV, class RT, int m>
  class LocalStiffness 
  {
	// grid types
      typedef typename GV::Grid::ctype DT;
      typedef typename GV::template Codim<0>::Entity Entity;

  public:
	// types for matrics, vectors and boundary conditions
	typedef FieldMatrix<RT,m,m> MBlockType;                      // one entry in the stiffness matrix

	virtual ~LocalStiffness () 
	{
	}

	//! print contents of local stiffness matrix
	void print (std::ostream& s, int width, int precision)
	{
	  // set the output format
	  s.setf(std::ios_base::scientific, std::ios_base::floatfield);
	  int oldprec = s.precision();
	  s.precision(precision);

	  for (int i=0; i<currentsize(); i++)
		{
		  s << "FEM";    // start a new row
		  s << " ";      // space in front of each entry
		  s.width(4);    // set width for counter
		  s << i;        // number of first entry in a line
		  for (int j=0; j<currentsize(); j++)
			{
			  s << " ";         // space in front of each entry
			  s.width(width);   // set width for each entry anew
			  s << mat(i,j);     // yeah, the number !
			}
		  s << std::endl;// start a new line
		}
	  

	  // reset the output format
	  s.precision(oldprec);
	  s.setf(std::ios_base::fixed, std::ios_base::floatfield);
	}

	//! access local stiffness matrix
	/*! Access elements of the local stiffness matrix. Elements are
	  undefined without prior call to the assemble method.
	 */
	const MBlockType& mat (int i, int j) const
	{
	  return A[i][j];
	}

	//! set the current size of the local stiffness matrix
	void setcurrentsize (int s)
	{
          A.setSize(s,s);
	}

	//! get the current size of the local stiffness matrix
	int currentsize ()
	{
            return A.N();
	}

  public:
	// assembled data
	Matrix<MBlockType> A;
    
  };

  /** @} */

}
#endif
