set(programs compute-disc-error
             cosserat-continuum
             gradient-flow
             harmonicmaps
             mixed-cosserat-continuum
             rod-eoc
             rodobstacle)

foreach(_program ${programs})
  add_executable(${_program} ${_program}.cc)
  add_dune_adolc_flags(${_program})
  add_dune_ipopt_flags(${_program})
  add_dune_pythonlibs_flags(${_program})
  add_dune_ug_flags(${_program})
  add_dune_mpi_flags(${_program})
#  target_compile_options(${_program} PRIVATE "-fpermissive")
endforeach()
