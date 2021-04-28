# Overview
`realtime_drawing` is library for fast drawing aimed at interactive applications and games. 

# Features
* Optimized for dynamically changed content.
* Local-antialiasing using blended strips.
* GPU rendering: output to streamed vertex/index buffers.
* Everything is batched: single draw-call is a common scenario.
* Agnostic of backends and vertex formats. Has `miniquad` <!-- and wgpu --> backend out-of the box. Easy to integrate with a custom engine.
* Can be used with 16-bit indices (to reduce memory bandwidth) and update multiple buffers when reaching 65K vertex/index limits.
* Easy to extend with custom traits. 
* WebAssembly support.
* Pure rust, no unsafe code.
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

# Supported primitives
| Primitive                 | Local Antialiasing | Parametrization |
|:--------------------------|--------------------|-----------------|
| Line (2D)                 | X                  |                 |
| Polyline (2D)             | X                  |                 |
| Circle Outline (2D)       | X                  |                 |
| Circle (2D)               | X                  |                 |
| Capsule Chains (2D)       | X                  |                 |
| Rectangle (2D)            |                    |                 |
| Rectangle Outline (2D)    |                    |                 |
| Grid (2D)                 |                    |                 |
| Box (3D)                  |                    |                 |

Shapes that miss local antialiasing can still be antialised with full-screen antialiasing methods such as MXAA.

<!-- # Comparison of local antialiasing to MXAA  -->
<!-- # Benchmarks
macroquad, ggez, lyon, piston, ImDrawList -->

<!--
-->
