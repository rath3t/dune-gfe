# Master

- The `SimoFoxEnergy` class (formerly known as `SimoFoxEnergyLocalStiffness`)
  does not accept a surface load density anymore. Use the
  `NeumannEnergy` class for assembling surface loads.

- The classes `LocalGeodesicFEFunction` and `LocalProjectedFEFunction`
  now take a `dune-functions` basis as their first arguments, instead
  of a `LocalFiniteElement`.

- The entire code is now in the `Dune::GFE` namespace.

- Remove the files `localgfetestfunctionbasis.hh` and `localtangentfefunction.hh`.
  They were never used for anything at all, I and currently wouldn't know
  what to use them for, either.

- All implementations of functions have been move to a `functions` subdirectory.

- The `LocalDensity` class now has a template parameter `ElementOrIntersection`,
  which replaces the parameter `Position`.  As the name says, this parameter
  has to be a grid element, or a grid intersection.  As it turned out,
  just depending on the position type of the integration domain was not enough.

- The `LocalDensity` class now has a `bind` method, which binds the density
  to a given element or intersection. This is necessary, for example, to in turn
  bind coefficient functions that the density may own.

# Release 2.10 (2024-12-29)

- The module `dune-elasticity` has been downgraded from a required
  dependency to an optional dependency.  It is currently only needed
  by the `film-on-substrate.cc` program.

- `cosserat-rod.cc` (formerly `rod3d.cc`) now reads the reference configuration
  and the initial iterate from the Python file.

- The file `rod3d.cc` has been renamed to `cosserat-rod.cc`,
  to better reflect what it does.

- In the `BulkCosseratDensity` class: Make the type of curvature
  tensor controllable at run-time (previously, ugly preprocessor switches
  where used).  Among its parameters `BulkCosseratDensity` now expects
  a string `curvatureType`, which has to take one of the values `norm`,
  `curl`, or `wryness`.

- Building `dune-gfe` now requires Dune 2.9 or newer. Older configurations
  have not gotten CI-tested for quite a while, anyway.

- The `RigidBodyMotion` class has been removed.  Please use
  `ProductManifold<RealTuple,Rotation>` from now on.

- You can use `std::tuple_element` and `std::tuple_size` with
  `ProductManifold`s now.

- Building the module requires CMake version 3.16 now, to be in line
  with the current core modules.

- All files that implement densities have been put into the
  `densities` subdirectory.

- All files that implement assemblers have been moved to the
  `assemblers` subdirectory.
  
- All files that implement target spaces have been moved to the
  `spaces` subdirectory.

- The file `periodic1dpq1nodalbasis.hh` has been removed.  Use `periodicbasis.hh`
  from the `dune-functions` module in the future.

- Replaced the class `GlobalGeodesicFEFunction` by a new one called
  `GlobalGFEFunction`.  There are two important changes:  First of all,
  the new class implements the `dune-functions` interface rather than
  the deprecated one based on inheritance from `VirtualGridViewFunction`.
  Secondly, the new class does not hard-wire geodesic interpolation
  anymore.  Rather, you can give it interpolation classes which then
  govern how interpolation is done.

- Added dune-gmsh4 as a dependency
- Build cosserat-continuum for different combinations of LFE-orders and
  GFE-orders, the respective program is called
  cosserat-continuum-Xd-in-Xd-LFE_ORDER-GFE_ORDER
  TODO: This is now set during compile time, but shall be changed to be
  set during runtime.

- Do not scale the density functions  with the thickness to avoid confusion
  since some densities need to be scaled and some do not need to be scaled
  with the thickness depending on the dimension of the grid, their direction
  and their kind (Neumann or volume load).

- Fix bug in the `RealTuple::log` method: Calling `log(a,b)`returned `a-b`
  instead of `b-a`.

- Fix the return value of `ProductManifold::log`: It was `TangentVector`,
  but now it is `EmbeddedTangentVector`.

- The `RigidBodyMotion` class has a `log` method now.

- The method `Rotation<3>::log` now returns an `EmbeddedTangentVector`
  instead of a `SkewMatrix`.  This is consistent with the other manifold
  implementations.

- Deprecate the method `RigidBodyMotion::difference`; the method
  `RigidBodyMotion::log`.  Watch out: The `difference` method was buggy!
  See https://gitlab.mn.tu-dresden.de/osander/dune-gfe/-/merge_requests/2
  for details.
