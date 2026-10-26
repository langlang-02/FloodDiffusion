# MoMask G1 reference code

`g1_io.py` and `g1_metrics.py` originate from the user's MoMask repository,
branch `feat/g1-rvq-tracking`, commit `bde792168a14b7317df2c45f233adeb7cae20b26`.
The upstream MIT license is included as `LICENSE`.

The metric module's imports and named-joint address lookup are adapted to be
self-contained here. Feature conversion, rotation conventions, native-space
metrics, and joint ordering retain upstream semantics. No RVQ model is used.
