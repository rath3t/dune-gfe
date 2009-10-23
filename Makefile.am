# $Id$

# possible options
LDADD = $(IPOPT_LDFLAGS) $(IPOPT_LIBS)
AM_CPPFLAGS += $(IPOPT_CPPFLAGS)

noinst_PROGRAMS = rodobstacle rod3d harmonicmaps dirneucoupling neudircoupling rod-eoc

rodobstacle_SOURCES = rodobstacle.cc
rod3d_SOURCES = rod3d.cc

rod_eoc_SOURCES = rod-eoc.cc

harmonicmaps_SOURCES = harmonicmaps.cc
harmonicmaps_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS)
harmonicmaps_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS)

dirneucoupling_SOURCES  = dirneucoupling.cc
dirneucoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS)
dirneucoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS)

neudircoupling_SOURCES  = neudircoupling.cc
neudircoupling_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS)
neudircoupling_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS)

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
