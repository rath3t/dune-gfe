# $Id$

# we need the module file to be able to build via dunecontrol
EXTRA_DIST=dune.module

SUBDIRS = doc m4 dune test

# possible options
LDADD = $(IPOPT_LDFLAGS) $(IPOPT_LIBS)
AM_CPPFLAGS += $(IPOPT_CPPFLAGS)

ADOLC_CPPFLAGS = -I/usr/include
ADOLC_LDFLAGS = -L/usr/lib
ADOLC_LIBS = -ladolc

noinst_PROGRAMS = cosserat-continuum rodobstacle rod3d harmonicmaps harmonicmaps-eoc rod-eoc

cosserat_continuum_SOURCES  = cosserat-continuum.cc
cosserat_continuum_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS) \
                              $(ADOLC_CPPFLAGS) $(PYTHON_CPPFLAGS)
cosserat_continuum_LDADD    = $(UG_LIBS) $(AMIRAMESH_LIBS) \
                              $(IPOPT_LIBS) $(PSURFACE_LIBS) \
                              $(ADOLC_LIBS) $(PYTHON_LIBS)
cosserat_continuum_LDFLAGS  = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) \
                              $(IPOPT_LDFLAGS) $(PSURFACE_LDFLAGS) \
                              $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS)

rodobstacle_SOURCES = rodobstacle.cc
rod3d_SOURCES = rod3d.cc

rod_eoc_SOURCES = rod-eoc.cc

harmonicmaps_SOURCES = harmonicmaps.cc
harmonicmaps_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS)
harmonicmaps_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS)

harmonicmaps_eoc_SOURCES = harmonicmaps-eoc.cc
harmonicmaps_eoc_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS)
harmonicmaps_eoc_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                          $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS)

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5

include $(top_srcdir)/am/global-rules
include $(top_srcdir)/am/top-rules
