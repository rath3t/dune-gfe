FROM ikarusproject/dunebase-clang

# Set working directory
WORKDIR /dune
USER root
RUN apt-get update && apt-get -y --no-install-recommends install libadolc-dev libtinyxml2-dev
ARG UID=1001
USER $UID
# Clone the dune-solvers repository
RUN git clone https://git.imp.fu-berlin.de/agnumpde/dune-solvers.git

# Run dunecontrol commands to build and install dune-solvers
RUN /dune/dune-common/bin/dunecontrol git checkout master && \
    /dune/dune-common/bin/dunecontrol --only=dune-solvers --cmake-opts="$DUNE_MODULE_CMAKE_FLAGS" cmake && \
    /dune/dune-common/bin/dunecontrol --only=dune-solvers  make

# Append the new build directory to CMAKE_PREFIX_PATH environment variable
ENV  CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/dune/dune-solvers/build-cmake:/dune/dune-gmsh4/build-cmake:"

# Set the working directory back to the default
WORKDIR /
