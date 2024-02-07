#install headers
install(FILES
        chiralskyrmionenergy.hh
        cosseratenergystiffness.hh
        cosseratrodenergy.hh
        geodesicfeassembler.hh
        geodesicfeassemblerwrapper.hh
        harmonicenergy.hh
        l2distancesquaredenergy.hh
        localenergy.hh
        localfirstordermodel.hh
        localgeodesicfeadolcstiffness.hh
        localgeodesicfefdstiffness.hh
        localgeodesicfestiffness.hh
        localintegralenergy.hh
        mixedgfeassembler.hh
        nonplanarcosseratshellenergy.hh
        simofoxenergy.hh
        sumenergy.hh
        surfacecosseratenergy.hh
        surfacecosseratstressassembler.hh
        weightedsumenergy.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/gfe/assemblers)
