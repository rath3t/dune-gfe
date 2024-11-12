set(programs compute-disc-error
             cosserat-rod
             gradient-flow
             harmonicmaps
             simofoxshell)
#            rodobstacle)

foreach(_program ${programs})
  add_executable(${_program} ${_program}.cc)
endforeach()

if (dune-parmg_FOUND AND dune-curvedgeometry_FOUND AND dune-elasticity_FOUND)
  add_executable("film-on-substrate" film-on-substrate.cc)
  set_property(TARGET "film-on-substrate" APPEND PROPERTY COMPILE_DEFINITIONS "WORLD_DIM=3;  LFE_ORDER=2; GFE_ORDER=2")
	set(programs film-on-substrate ${programs})

  add_executable("film-on-substrate-stress-plot" film-on-substrate-stress-plot.cc)
  set(programs film-on-substrate-stress-plot ${programs})
endif()

add_executable("cosserat-continuum-2d-in-2d" cosserat-continuum.cc)
set_property(TARGET "cosserat-continuum-2d-in-2d" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=2; WORLD_DIM=2")
set(programs cosserat-continuum-2d-in-2d ${programs})

if (dune-foamgrid_FOUND)
  add_executable("cosserat-continuum-2d-in-3d-2-2" cosserat-continuum.cc)
  set_property(TARGET "cosserat-continuum-2d-in-3d-2-2" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=2; WORLD_DIM=3; LFE_ORDER=2; GFE_ORDER=2")
  set(programs cosserat-continuum-2d-in-3d-2-2 ${programs})

  add_executable("cosserat-continuum-2d-in-3d-2-1" cosserat-continuum.cc)
  set_property(TARGET "cosserat-continuum-2d-in-3d-2-1" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=2; WORLD_DIM=3; LFE_ORDER=2; GFE_ORDER=1")
  set(programs cosserat-continuum-2d-in-3d-2-1 ${programs})

  add_executable("cosserat-continuum-2d-in-3d-1-1" cosserat-continuum.cc)
  set_property(TARGET "cosserat-continuum-2d-in-3d-1-1" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=2; WORLD_DIM=3; LFE_ORDER=1; GFE_ORDER=1")
  set(programs cosserat-continuum-2d-in-3d-1-1 ${programs})
endif()

add_executable("cosserat-continuum-3d-in-3d-2-2" cosserat-continuum.cc)
set_property(TARGET "cosserat-continuum-3d-in-3d-2-2" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=3; WORLD_DIM=3; LFE_ORDER=2; GFE_ORDER=2")
set(programs cosserat-continuum-3d-in-3d-2-2 ${programs})

add_executable("cosserat-continuum-3d-in-3d-2-1" cosserat-continuum.cc)
set_property(TARGET "cosserat-continuum-3d-in-3d-2-1" APPEND PROPERTY COMPILE_DEFINITIONS "GRID_DIM=3; WORLD_DIM=3; LFE_ORDER=2; GFE_ORDER=1")
set(programs cosserat-continuum-3d-in-3d-2-1 ${programs})

foreach(_program ${programs})
  target_link_dune_default_libraries(${_program})
  add_dune_pythonlibs_flags(${_program})
  target_link_libraries(${_program} PUBLIC tinyxml2)
endforeach()
