# $Id: Makefile.am 1867 2007-12-31 15:45:00Z sander@PCPOOL.MI.FU-BERLIN.DE $

# possible options
LDADD = $(UG_LDFLAGS) $(AMIRAMESH_LDFLAGS) $(UG_LIBS) $(AMIRAMESH_LIBS)
AM_CPPFLAGS += $(UG_CPPFLAGS) $(AMIRAMESH_CPPFLAGS) -Wall

check_PROGRAMS = frameinvariancetest rotationtest fdcheck localgeodesicfefunctiontest \
                 harmonicenergytest

frameinvariancetest_SOURCES = frameinvariancetest.cc

rotationtest_SOURCES = rotationtest.cc

fdcheck_SOURCES = fdcheck.cc

localgeodesicfefunctiontest_SOURCES = localgeodesicfefunctiontest.cc

harmonicenergytest_SOURCES = harmonicenergytest.cc

# don't follow the full GNU-standard
# we need automake 1.5
AUTOMAKE_OPTIONS = foreign 1.5
