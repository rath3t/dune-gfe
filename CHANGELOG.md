# Master

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
