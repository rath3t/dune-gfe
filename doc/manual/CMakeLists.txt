dune_add_latex_document(
  SOURCE dune-gfe-manual.tex
  FATHER_TARGET doc
  BUILD_ON_INSTALL
  INSTALL ${CMAKE_INSTALL_DOCDIR}/manual)
