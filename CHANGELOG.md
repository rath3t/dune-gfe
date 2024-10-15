# Master

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
