set(programs compute-disc-error
             gradient-flow
             harmonicmaps
             simofoxshell)
#            rodobstacle)
if (${DUNE_COMMON_VERSION} VERSION_GREATER_EQUAL 2.8)
  set (programs rod3d ${programs})
endif()
if (${DUNE_ELASTICITY_VERSION} VERSION_GREATER_EQUAL 2.8)
  set(programs film-on-substrate-stress-plot ${programs})
endif()
if (dune-parmg_FOUND AND dune-curvedgeometry_FOUND AND ${DUNE_ELASTICITY_VERSION} VERSION_GREATER_EQUAL 2.8)
	set(programs film-on-substrate ${programs})
endif()

foreach(_program ${programs})
  add_executable(${_program} ${_program}.cc)
endforeach()

add_executable("cosserat-continuum-2d-in-2d" cosserat-continuum.cc)
set_property(TARGET "cosserat-continuum-2d-in-2d" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=2; WORLD_DIM=2")
set(programs cosserat-continuum-2d-in-2d ${programs})

if (dune-foamgrid_FOUND)
  add_executable("cosserat-continuum-2d-in-3d" cosserat-continuum.cc)
  set_property(TARGET "cosserat-continuum-2d-in-3d" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=2; WORLD_DIM=3")
  set(programs cosserat-continuum-2d-in-3d ${programs})
endif()

add_executable("cosserat-continuum-3d-in-3d" cosserat-continuum.cc)
set_property(TARGET "cosserat-continuum-3d-in-3d" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=3; WORLD_DIM=3")
set(programs cosserat-continuum-3d-in-3d ${programs})

foreach(_program ${programs})
  target_link_dune_default_libraries(${_program})
  add_dune_pythonlibs_flags(${_program})
  if (${DUNE_COMMON_VERSION} VERSION_GREATER_EQUAL 2.8)
    target_link_libraries(${_program} PUBLIC tinyxml2)
  else()
    target_link_libraries(${_program} tinyxml2)
  endif()
#  target_compile_options(${_program} PRIVATE "-fpermissive")
endforeach()