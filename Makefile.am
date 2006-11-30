# $Id$

# possible options
#LDADD = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS)
#AM_CPPFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS)

IPOPT_DIR = /home/haile/sander/COIN/Ipopt

noinst_PROGRAMS = staticrod staticrod2 rod3d simplecoupling dirneucoupling

staticrod_SOURCES = staticrod.cc
staticrod2_SOURCES = staticrod2.cc
rod3d_SOURCES = rod3d.cc

simplecoupling_SOURCES  = simplecoupling.cc
simplecoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(MPI_CPPFLAGS)
simplecoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) $(MPI_LDFLAGS)

dirneucoupling_SOURCES  = dirneucoupling.cc
dirneucoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(MPI_CPPFLAGS) -I$(IPOPT_DIR)/IPOPT/include
dirneucoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) $(MPI_LDFLAGS) \
                          -L$(IPOPT_DIR)/lib -lipopt -llapack -lblas -lg2c

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
