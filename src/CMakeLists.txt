set(programs compute-disc-error
             cosserat-continuum
             film-on-substrate
             gradient-flow
             harmonicmaps
             rod3d
             rod-eoc)
#            rodobstacle)

foreach(_program ${programs})
  add_executable(${_program} ${_program}.cc)
  target_link_dune_default_libraries(${_program})
  add_dune_pythonlibs_flags(${_program})
  target_link_libraries(${_program} tinyxml2)
#  target_compile_options(${_program} PRIVATE "-fpermissive")
endforeach()
