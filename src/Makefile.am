
SUBDIRS =

ADOLC_CPPFLAGS = -I/home/sander/adolc-inst/include
ADOLC_LDFLAGS = -L/home/sander/adolc-inst/lib64
ADOLC_LIBS = -ladolc

noinst_PROGRAMS = cosserat-continuum \
                  harmonicmaps \
                  rod3d

cosserat_continuum_SOURCES  = cosserat-continuum.cc
cosserat_continuum_CXXFLAGS = $(UG_CPPFLAGS) $(IPOPT_CPPFLAGS) \
                              $(ADOLC_CPPFLAGS) $(PYTHON_CPPFLAGS)
cosserat_continuum_LDADD    = $(UG_LIBS) $(IPOPT_LIBS) \
                              $(ADOLC_LIBS) $(PYTHON_LIBS)
cosserat_continuum_LDFLAGS  = $(UG_LDFLAGS) $(IPOPT_LDFLAGS) \
                              $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS)

harmonicmaps_SOURCES = harmonicmaps.cc
harmonicmaps_CXXFLAGS = $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) $(IPOPT_CPPFLAGS) $(PSURFACE_CPPFLAGS) \
                        $(ADOLC_CPPFLAGS) $(PYTHON_CPPFLAGS)
harmonicmaps_LDADD    = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS) \
                        $(IPOPT_LDFLAGS) $(IPOPT_LIBS) $(PSURFACE_LDFLAGS) $(PSURFACE_LIBS) \
                        $(ADOLC_LIBS) $(PYTHON_LIBS) $(ADOLC_LDFLAGS) $(PYTHON_LDFLAGS)


rod3d_SOURCES = rod3d.cc
rod3d_CXXFLAGS = $(IPOPT_CPPFLAGS)
rod3d_LDADD    = $(IPOPT_LIBS)
rod3d_LDFLAGS  = $(IPOPT_LDFLAGS)

# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5

# pass most important options when "make distcheck" is used
DISTCHECK_CONFIGURE_FLAGS = --with-dune-common=$(DUNE_COMMON_ROOT) --with-dune-geometry=$(DUNE_GEOMETRY_ROOT) --with-dune-grid=$(DUNE_GRID_ROOT) --with-dune-istl=$(DUNE_ISTL_ROOT) --with-dune-localfunctions=$(DUNE_LOCALFUNCTIONS_ROOT)  CXX="$(CXX)" CC="$(CC)"

include $(top_srcdir)/am/global-rules

