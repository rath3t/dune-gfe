# $Id$

# possible options
#LDADD = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS)
#AM_CPPFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS)

# Lapack++
LAPACKPP_CPPFLAGS = -I/home/haile/sander/lapack++-inst/include/lapackpp
LAPACKPP_LDFLAGS  = -L/home/haile/sander/lapack++-inst/lib -llapackpp

IPOPT_DIR = /home/haile/sander/COIN/Ipopt

noinst_PROGRAMS = staticrod staticrod2 rod3d dirneucoupling

staticrod_SOURCES = staticrod.cc
staticrod2_SOURCES = staticrod2.cc
rod3d_SOURCES = rod3d.cc

dirneucoupling_SOURCES  = dirneucoupling.cc linearsolver.cc
dirneucoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(MPI_CPPFLAGS) -I$(IPOPT_DIR)/IPOPT/include \
                          $(LAPACKPP_CPPFLAGS)
dirneucoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) $(MPI_LDFLAGS) \
                          -L$(IPOPT_DIR)/lib -lipopt -llapack -lblas -lg2c \
                          $(LAPACKPP_LDFLAGS)

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
