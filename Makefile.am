# $Id$

# possible options
#LDADD = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS)
#AM_CPPFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS)

# Include and linker paths to the old fortran IPOpt library
#IPOPT_DIR = /home/haile/sander/COIN/Ipopt
#IPOPT_LDADD    = -L$(IPOPT_DIR)/lib -lipopt -llapack -lblas -lg2c
#IPOPT_CPPFLAGS = -I$(IPOPT_DIR)/IPOPT/include -DHAVE_IPOPT

# Include and linker paths to the new C++ IPOpt library
IPOPT_CPP_DIR  = /home/haile/sander/Ipopt-3.3.5/build
IPOPT_LDADD    = -L$(IPOPT_CPP_DIR)/lib -lipopt -llapack -lblas -lgfortran
IPOPT_CPPFLAGS = -I$(IPOPT_CPP_DIR)/include -DHAVE_IPOPT_CPP

noinst_PROGRAMS = staticrod staticrod2 rod3d dirneucoupling

staticrod_SOURCES = staticrod.cc
staticrod2_SOURCES = staticrod2.cc
rod3d_SOURCES = rod3d.cc

dirneucoupling_SOURCES  = dirneucoupling.cc
dirneucoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS)
dirneucoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDADD)

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
