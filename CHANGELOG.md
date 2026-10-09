# Changelog

## 0.1.2

- Add external inputs through sampled tensors (`u`) or a callback (`u_fn`), with
  the right-hand side called as `f(t, y, u)` when inputs are provided.
- Preserve the existing `from FDEint import FDEint` import and the `f(t, y)`
  interface for equations without inputs.
- Correct the time and input associated with stored right-hand-side evaluations
  and support time grids with a nonzero initial time.
- Validate scalar fractional orders, strictly increasing time grids, and tensor
  shapes; support shared and batched observation grids.
- Support differentiation through initial states, the fractional order,
  sampled inputs, input callbacks, and right-hand-side parameters.
- Add an external-input example, API documentation, and a reproducible package
  build and release workflow.
- Correct the learned-order example's final solve to use its trained fractional
  order and document optional example dependencies.

## 0.1.1

- Previously published PyPI release of the predictor-corrector solver.
