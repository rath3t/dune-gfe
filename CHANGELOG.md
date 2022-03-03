# Master

- Fix bug in the `RealTuple::log` method: Calling `log(a,b)`returned `a-b`
  instead of `b-a`.

- Fix the return value of `ProductManifold::log`: It was `TangentVector`,
  but now it is `EmbeddedTangentVector`.
