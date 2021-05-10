# Overview
`realtime-drawing` is a Rust crate (library) for fast drawing aimed at interactive applications and games. 

This crate is work-in-progress and was not published on crates.io yet.

# Features
- Optimized for dynamically generated content.
- Antialiasing of lines using blended strips.
- GPU rendering: output to streamed vertex/index buffers.
- Aggressive batching across primitive types.
- Backend-agnostic. Comes with `MiniquadBatch` that implements [`miniquad`](https://github.com/not-fl3/miniquad)-backend out of the box. Easy integration into existing engines.
- Works with custom vertex format through traits.
- Can be used with 16-bit indices (to reduce memory bandwidth) and update multiple buffers when reaching 65K vertex/index limits.
- Easy to extend with custom traits.
- WebAssembly support.
- Pure rust, no unsafe code.
<!-- * No dependencies in minimal configuration. -->
<!-- * Supports parametrization of various shapes with a lambda function. Easy to add custom colors, UV-s or add third dimension to 2D-primitives. -->
<!-- * SIMD support -->

# Non-goals
* Is not a complete vector-graphics backend. Implements features that can be efficiently performed in realtime. 
  If you are looking to render static SVG you might be better off using `Lyon` or `Skia`.
* No composition. You can do this with your own shaders though.
* No text rendering.
* Not for software rasterization.

<!--
# Examples
## Showcase
## Lines
## Jumping Rabbits
Online demo.
-->

<!-- # Comparison of local antialiasing to MXAA  -->
<!-- # Benchmarks
macroquad, ggez, lyon, piston, ImDrawList -->

<!--
-->
