# $Id$

# possible options
#LDADD = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS)
#AM_CPPFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS)

noinst_PROGRAMS = staticrod staticrod2 rod3d dirneucoupling

staticrod_SOURCES = staticrod.cc
staticrod2_SOURCES = staticrod2.cc
rod3d_SOURCES = rod3d.cc

dirneucoupling_SOURCES  = dirneucoupling.cc
dirneucoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS)
dirneucoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS)

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
