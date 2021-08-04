//! A collection of fast drawing routines aimed at interactive applications and games.
//!
//! # Features
//! - Optimized for dynamically generated content.
//! - Antialiasing of lines using blended strips.
//! - GPU rendering: output to streamed vertex/index buffers.
//! - Aggressive batching across primitive types.
//! - Backend-agnostic. Comes with [`MiniquadBatch`] that implements [`miniquad`]-backend out of the box. Easy integration into existing engines.
//! - Works with custom vertex format through traits. See [`FromPos2Color`], [`FromPos3Color`] etc.
//! - Can be used with 16-bit indices (to reduce memory bandwidth) and update multiple buffers when reaching 65K vertex/index limits.
//! - Easy to extend with custom traits.
//! - WebAssembly support.
//! - Pure rust, no unsafe code.
//!
//! Individual drawing operations such as [`GeometryBatch::fill_circle_aa`] or
//! [`GeometryBatch::stroke_polyline_aa`] are available in [`GeometryBatch`] implementation.
//!
/// [`miniquad`]: https://docs.rs/miniquad/
mod example;
mod miniquad_batch;
pub use example::VertexPos3UvColor;
pub use miniquad_batch::MiniquadBatch;

use core::default::Default;
use core::iter::Iterator;
use core::marker::Copy;
use glam::vec2;
use glam::Vec2;
use std::f32::consts::PI;
use std::mem::take;
use std::unreachable;

type IndexType = u16;

/// Implements primitive drawing.
///
/// `stroke_`/`fill_`-methods are used to draw individual primitives.
///
/// # Example
///
/// ```
/// // initialization stage
/// let geometry = GeometryBatch::new(1024, 1024);
/// geometry.clear();
/// geometry.fill_circle_aa(vec2(512.0, 512.0), 128.0, 64);
/// geometry.stroke_polyline_aa(&[vec2(384.0, 512.0), vec2(512.0, 512.0), vec2(512.0, 384.0)],
///                               [255, 0, 0, 255], true, 2.0);
///
/// // upload geometry.vertices/indices and render according to geometry.commands
///
/// ```
/// # Internals
/// Drawn primitives are accumulated in [`vertices`](`GeometryBatch::vertices`),
/// [`indices`](`GeometryBatch::indices`), and [`commands`](`GeometryBatch::commands`).
///
/// [`GeometryBatch::finish_commands`] is normally used to finalize command list before drawing.
///
/// This type normally used as a part of backend-specific batch. See: [`MiniquadBatch`]
///
pub struct GeometryBatch<Vertex: Copy> {
    /// Batched vertices.
    pub vertices: Vec<Vertex>,
    /// Batched indices.
    pub indices: Vec<IndexType>,
    /// List of emitted drawing commands.
    pub commands: Vec<GeometryCommand>,

    // setup
    /// Maximum number of vertices per vertex buffer.
    pub max_buffer_vertices: usize,
    /// Maximum number of indices per index buffer.
    pub max_buffer_indices: usize,

    /// Size of pixel in position units. It is used for blended strips of antialiased geometry.
    pub pixel_size: f32,

    // active command
    last_vertices_command: usize,
    last_indices_command: usize,
    draw_indices: usize,
    buffer_vertices_end: usize,
    buffer_indices_end: usize,

    // temporary buffers to avoid stack allocations
    temp_points: Vec<Vec2>,
    temp_normals: Vec<Vec2>,
    temp_path: Vec<Vec2>,
}

impl<Vertex: Copy> GeometryBatch<Vertex> {
    /// Instantiates new GeometryBatch with a given capacity.
    pub fn with_capacity(max_buffer_vertices: usize, max_buffer_indices: usize) -> Self {
        assert!(max_buffer_vertices <= IndexType::MAX as usize + 1);
        let mut commands = Vec::new();
        let last_indices_command = commands.len();
        commands.push(GeometryCommand::Indices { num_indices: 0 });
        let last_vertices_command = commands.len();
        commands.push(GeometryCommand::Vertices { num_vertices: 0 });
        Self {
            vertices: Vec::with_capacity(max_buffer_vertices),
            indices: Vec::with_capacity(max_buffer_indices),
            pixel_size: 1.0,
            max_buffer_vertices,
            max_buffer_indices,
            commands,
            draw_indices: 0,
            buffer_vertices_end: max_buffer_vertices,
            buffer_indices_end: max_buffer_indices,
            last_indices_command,
            last_vertices_command,
            temp_points: Vec::new(),
            temp_normals: Vec::new(),
            temp_path: Vec::new(),
        }
    }

    /// Base method for implementing primitives.
    /// Allocates a slice of vertices and indices.
    /// Returns (vertices_slice, indices_slice, base_index) where
    /// base_index is an index of first vertex in the buffer.
    #[inline(always)]
    pub fn allocate(
        &mut self,
        num_vertices: usize,
        num_indices: usize,
        def: Vertex,
    ) -> (&mut [Vertex], &mut [IndexType], IndexType) {
        assert!(num_vertices < IndexType::MAX as usize);
        let old_vertices = self.vertices.len();
        let old_indices = self.indices.len();
        let new_vertices = old_vertices + num_vertices;
        let new_indices = old_indices + num_indices;
        let indices_overflow = new_indices > self.buffer_indices_end;
        let vertices_overflow = new_vertices > self.buffer_vertices_end;
        let split_draw = indices_overflow || vertices_overflow;
        if split_draw {
            self.commands.push(GeometryCommand::DrawCall {
                num_indices: old_indices - self.draw_indices,
            });
            self.draw_indices = old_indices;
        }
        if indices_overflow {
            // finish last indices command
            self.commands[self.last_indices_command] = GeometryCommand::Indices {
                num_indices: old_indices + self.max_buffer_indices - self.buffer_indices_end,
            };
            self.last_indices_command = self.commands.len();
            self.commands
                .push(GeometryCommand::Indices { num_indices: 0 });
            self.buffer_indices_end = old_indices + self.max_buffer_indices;
        }
        if vertices_overflow {
            self.commands[self.last_vertices_command] = GeometryCommand::Vertices {
                num_vertices: old_vertices + self.max_buffer_vertices - self.buffer_vertices_end,
            };
            self.last_vertices_command = self.commands.len();
            self.commands
                .push(GeometryCommand::Vertices { num_vertices: 0 });
            self.buffer_vertices_end = old_vertices + self.max_buffer_vertices;
        }
        let buffer_vertices_begin = self.buffer_vertices_end - self.max_buffer_vertices;
        let first_index = (old_vertices - buffer_vertices_begin) as IndexType;
        self.vertices.resize(new_vertices, def);
        self.indices.resize(new_indices, 0 as IndexType);
        (
            &mut self.vertices[old_vertices..new_vertices],
            &mut self.indices[old_indices..new_indices],
            first_index,
        )
    }

    /// Returns previously allocated vertex/indices.
    /// Used when exact knowledge of allocation requires second pass but
    /// conservative estimation is available.
    pub fn reclaim_allocation(&mut self, num_vertices: usize, num_indices: usize) {
        self.vertices.truncate(self.vertices.len() - num_vertices);
        self.indices.truncate(self.indices.len() - num_indices);
    }

    /// Clears vertex, index buffers and command list.
    pub fn clear(&mut self) {
        self.indices.clear();
        self.vertices.clear();
        self.commands.clear();
        self.last_indices_command = self.commands.len();
        self.commands
            .push(GeometryCommand::Indices { num_indices: 0 });
        self.last_vertices_command = self.commands.len();
        self.commands
            .push(GeometryCommand::Vertices { num_vertices: 0 });
        self.draw_indices = 0;
        self.buffer_indices_end = self.max_buffer_indices;
        self.buffer_vertices_end = self.max_buffer_vertices;
    }

    /// Finalizes command list. This function is expected to be called before
    /// uploading vertex/index buffers.
    pub fn finish_commands(&mut self) {
        let num_vertices = self.vertices.len();
        let num_indices = self.indices.len();
        if self.buffer_indices_end != num_indices + self.max_buffer_indices {
            self.commands[self.last_indices_command] = GeometryCommand::Indices {
                num_indices: num_indices + self.max_buffer_indices - self.buffer_indices_end,
            };
        }
        if self.buffer_vertices_end != num_vertices + self.max_buffer_vertices {
            self.commands[self.last_vertices_command] = GeometryCommand::Vertices {
                num_vertices: num_vertices + self.max_buffer_vertices - self.buffer_vertices_end,
            };
        }
        if self.draw_indices != num_indices {
            self.commands.push(GeometryCommand::DrawCall {
                num_indices: num_indices - self.draw_indices,
            });
            self.draw_indices = num_indices;
        }
    }
}

impl<Vertex: Copy + Default> GeometryBatch<Vertex> {
    /// Closure arguments are `(pos, alpha, uv)` where `uv` are normalized polar coordinates on the circle.
    #[doc(hidden)]
    #[inline]
    pub fn fill_circle_aa_with<ToVertex: FnMut(Vec2, f32, Vec2) -> Vertex>(
        &mut self,
        center: Vec2,
        radius: f32,
        num_segments: usize,
        mut to_vertex: ToVertex,
    ) {
        let pixel = self.pixel_size;
        let half_pixel = pixel * 0.5;
        let (vs, is, first) =
            self.allocate(2 * num_segments + 1, num_segments * 9, Vertex::default());
        for (i, pair) in vs.chunks_mut(2).enumerate() {
            let u = i as f32 / num_segments as f32;
            let angle = u * 2.0 * std::f32::consts::PI;
            let cos = angle.cos();
            let sin = angle.sin();
            for (v, p) in pair.iter_mut().zip(&[0.0, pixel]) {
                let pos = center + vec2(cos, sin) * (radius - half_pixel + p);
                *v = to_vertex(pos, 1.0 - p, vec2(u, 1.0));
            }
        }
        *vs.last_mut().unwrap() = to_vertex(center, 1.0, vec2(0.0, 0.0));
        let central_vertex = num_segments * 2;

        for (section_i, section) in is.chunks_mut(9).enumerate() {
            let section_n = (section_i + 1) % num_segments;
            let indices = [
                central_vertex,
                section_i * 2 + 0,
                section_n * 2 + 0,
                section_i * 2 + 0,
                section_i * 2 + 1,
                section_n * 2 + 0,
                section_i * 2 + 1,
                section_n * 2 + 1,
                section_n * 2 + 0,
            ];
            for (dest, src) in section.iter_mut().zip(&indices) {
                *dest = first + *src as IndexType;
            }
        }
    }
    
    /// Closure arguments are `(pos, uv)` where `u` is normalized polar coordinate on a circle.
    #[inline]
    #[doc(hidden)]
    pub fn fill_circle_with<ToVertex: FnMut(Vec2, Vec2) -> Vertex>(
        &mut self,
        center: Vec2,
        radius: f32,
        num_segments: usize,
        mut to_vertex: ToVertex,
    ) {
        let (vs, is, first) =
            self.allocate(num_segments + 1, num_segments * 3, Vertex::default());
        for (i, v) in vs.iter_mut().enumerate() {
            let u = i as f32 / num_segments as f32;
            let angle = u * 2.0 * std::f32::consts::PI;
            let cos = angle.cos();
            let sin = angle.sin();
            let pos = center + vec2(cos, sin) * radius;
            *v = to_vertex(pos, vec2(u, 1.0));
        }
        *vs.last_mut().unwrap() = to_vertex(center, vec2(0.0, 0.0));
        let central_vertex = num_segments;

        for (section_i, section) in is.chunks_mut(3).enumerate() {
            let section_n = (section_i + 1) % num_segments;
            section[0] = first + central_vertex as IndexType;
            section[1] = first + section_i as IndexType;
            section[2] = first + section_n as IndexType;
        }
    }
}

impl<Vertex: Copy + Default + FromPos2Color> GeometryBatch<Vertex> {
    /// Adds filled antialiased circle positioned at `center` with `radius` and `thickness`.
    ///
    /// Circle outer edge is constructed out of `num_segments`-linear segments.
    #[inline]
    pub fn fill_circle_aa(
        &mut self,
        center: Vec2,
        radius: f32,
        num_segments: usize,
        color: [u8; 4],
    ) {
        self.fill_circle_aa_with(center, radius, num_segments, move |pos, alpha, _uv| {
            Vertex::from_pos2_color(pos.into(), [color[0], color[1], color[2], (color[3] as f32 * alpha) as u8])
        })
    }
    
    /// Adds filled circle positioned at `center` with `radius` and `thickness`.
    ///
    /// Circle outer edge is constructed out of `num_segments`-linear segments.
    #[inline]
    pub fn fill_circle(
        &mut self,
        center: Vec2,
        radius: f32,
        num_segments: usize,
        color: [u8; 4],
    ) {
        self.fill_circle_with(center, radius, num_segments, move |pos, _uv| {
            Vertex::from_pos2_color(pos.into(), color)
        })
    }
}

impl<Vertex: Copy + Default> GeometryBatch<Vertex> {
    /// Adds an antialiased outline of a circle positioned at `center` with `radius` and `thickness`.
    ///
    /// The circle is constructed out of `num_segments`-linear segments.
    ///
    /// Closure arguments are `(pos, alpha, u)` where u is normalized polar coordinate on a circle.
    #[inline]
    #[doc(hidden)]
    pub fn stroke_circle_aa_with<ToVertex: FnMut(Vec2, f32, f32) -> Vertex>(
        &mut self,
        center: Vec2,
        radius: f32,
        thickness: f32,
        num_segments: usize,
        mut to_vertex: ToVertex,
    ) {
        let pixel_size = self.pixel_size;
        if thickness > pixel_size {
            let (vs, is, first) =
                self.allocate(4 * num_segments, num_segments * 18, Vertex::default());
            let ht = (thickness - pixel_size) * 0.5;
            for (i, pair) in vs.chunks_mut(4).enumerate() {
                let t = i as f32 / num_segments as f32;
                let angle = t * 2.0 * std::f32::consts::PI;
                let cos = angle.cos();
                let sin = angle.sin();
                for (v, p) in pair.iter_mut().zip(&[
                    (-ht - pixel_size, 0.0),
                    (-ht, 1.0),
                    (ht, 1.0),
                    (ht + pixel_size, 0.0),
                ]) {
                    let pos = vec2(
                        (center[0] + cos * (radius + p.0)).into(),
                        (center[1] + sin * (radius + p.0)).into(),
                    );
                    *v = to_vertex(pos, p.1, t);
                }
            }

            for (section_i, section) in is.chunks_mut(18).enumerate() {
                let section_n = (section_i + 1) % num_segments;
                let indices = [
                    section_i * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 2,
                    section_n * 4 + 1,
                    section_i * 4 + 2,
                    section_i * 4 + 3,
                    section_n * 4 + 2,
                    section_i * 4 + 3,
                    section_n * 4 + 3,
                    section_n * 4 + 2,
                ];
                for (dest, src) in section.iter_mut().zip(&indices) {
                    *dest = first + *src as IndexType;
                }
            }
        } else {
            let (vs, is, first) =
                self.allocate(4 * num_segments, num_segments * 12, Vertex::default());
            for (i, pair) in vs.chunks_mut(4).enumerate() {
                let t = i as f32 / num_segments as f32;
                let angle = t * 2.0 * std::f32::consts::PI;
                let cos = angle.cos();
                let sin = angle.sin();
                for (v, p) in
                    pair.iter_mut()
                        .zip(&[(-pixel_size, 0.0), (0.0, thickness), (pixel_size, 0.0)])
                {
                    let pos = vec2(
                        center[0] + cos * (radius + p.0),
                        center[1] + sin * (radius + p.0),
                    );
                    *v = to_vertex(pos, p.1, t);
                }
            }
            for (section_i, section) in is.chunks_mut(12).enumerate() {
                let section_n = (section_i + 1) % num_segments;
                let indices = [
                    section_i * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 2,
                    section_n * 4 + 1,
                ];
                for (dest, src) in section.iter_mut().zip(&indices) {
                    *dest = first + *src as IndexType;
                }
            }
        }
    }
    
    /// Adds an outline of circle positioned at `center` with `radius` and `thickness`.
    ///
    /// The circle is constructed out of `num_segments`-linear segments.
    ///
    /// Closure arguments are `(pos, uv)` where `u` is a normalized angular coordinate on the circle
    /// and `v` has value of 0.0 on the inner edge and 1.0 on the outer edge.
    #[inline]
    #[doc(hidden)]
    pub fn stroke_circle_with<ToVertex: FnMut(Vec2, Vec2) -> Vertex>(
        &mut self,
        center: Vec2,
        radius: f32,
        thickness: f32,
        num_segments: usize,
        mut to_vertex: ToVertex,
    ) {
        let (vs, is, first) =
            self.allocate(2 * num_segments, num_segments * 6, Vertex::default());
        let half_thickness = thickness * 0.5;
        for (i, pair) in vs.chunks_mut(2).enumerate() {
            let t = i as f32 / num_segments as f32;
            let angle = t * 2.0 * PI;
            let cos = angle.cos();
            let sin = angle.sin();

            let inner_radius = radius - half_thickness;
            let outer_radius = radius + half_thickness;
            pair[0] = to_vertex(vec2(center.x + cos * inner_radius, center.y + sin * inner_radius), vec2(t, 0.0));
            pair[1] = to_vertex(vec2(center.x + cos * outer_radius, center.y + sin * outer_radius), vec2(t, 1.0));
        }
        for (section_i, section) in is.chunks_mut(6).enumerate() {
            let section_i2 = first + section_i as IndexType * 2;
            let section_n = (section_i + 1) % num_segments;
            let section_n2 = first + section_n as IndexType * 2;
            section[0] = section_i2 + 0;
            section[1] = section_i2 + 1;
            section[2] = section_n2 + 1;
            section[3] = section_i2 + 0;
            section[4] = section_n2 + 0;
            section[5] = section_n2 + 1;
        }
    }
}

const MODE_THICK_AA: i32 = 0;
const MODE_THIN_AA: i32 = 1;
const MODE_NORMAL: i32 = 2;

impl<Vertex: Copy + Default + FromPos2Color> GeometryBatch<Vertex> {
    /// Adds an antialiased outline of a circle positioned at `center` with `radius`, `thickness`
    /// and `color`.
    ///
    /// The circle is constructed out of `num_segments`-linear segments.
    #[inline]
    pub fn stroke_circle_aa(
        &mut self,
        center: Vec2,
        radius: f32,
        thickness: f32,
        num_segments: usize,
        color: [u8; 4],
    ) {
        self.stroke_circle_aa_with(
            center,
            radius,
            thickness,
            num_segments,
            |pos, alpha, _u| {
                Vertex::from_pos2_color(pos.into(), [color[0], color[1], color[2], (255.0 * alpha) as u8])
            },
        )
    }

    /// Adds an outline of circle positioned at `center` with `radius`, `thickness` and `color`.
    ///
    /// The circle is constructed out of `num_segments`-linear segments.
    #[inline]
    pub fn stroke_circle (
        &mut self,
        center: Vec2,
        radius: f32,
        thickness: f32,
        num_segments: usize,
        color: [u8; 4],
    ) {
        self.stroke_circle_with(
            center,
            radius,
            thickness,
            num_segments,
            |pos, _| Vertex::from_pos2_color(pos.into(), color)
        )
    }
    
    /// Draws an antialiased line from `start` to `finish` of `thickness` and `color`.
    ///
    /// The line is drawn without caps. Caps are not antialiased.
    pub fn stroke_line_aa(
        &mut self,
        start: Vec2,
        end: Vec2,
        thickness: f32,
        color: [u8; 4]
    ) {
        self.stroke_polyline_aa(&[start, end], false, thickness, color);
    }
    
    /// Draws a line from `start` to `finish` of `thickness` and `color`.
    ///
    /// The line is drawn without caps. Caps are not antialiased.
    pub fn stroke_line(
        &mut self,
        start: Vec2,
        end: Vec2,
        thickness: f32,
        color: [u8; 4]
    ) {
        self.stroke_polyline(&[start, end], false, thickness, color);
    }

    // Based on ImDrawList::AddPoyline implementation from Dear ImGui by Omar Cornut 
    // (https://github.com/ocornut/imgui/, MIT license) and Pavel Potoček (https://github.com/potocpav)
    fn stroke_polyline_internal<
        const MODE: i32, // MODE_NORMAL, MODE_THIN_AA, MODE_THICK_AA
    >(
        &mut self,
        points: &[Vec2],
        closed: bool,
        thickness: f32,
        color: [u8; 4],
    ) {
        let points_count = points.len();
        if points_count < 2 {
            return;
        }

        let count = if closed {
            points_count
        } else {
            points_count - 1
        }; // segment count

        let pixel_size = self.pixel_size;
        let color_transparent = [color[0], color[1], color[2], 0];
        let color_thin = [color[0], color[1], color[2], (color[3] as f32 * thickness / pixel_size).min(255.0) as u8];

        // Precise line with bevels on acute angles
        let max_n_vtx = match MODE {
            MODE_THICK_AA => 6,
            MODE_THIN_AA => 4,
            MODE_NORMAL => 3,
            _ => unreachable!()
        };
        let max_n_idx = match MODE {
            MODE_THICK_AA => 3 * 9,
            MODE_THIN_AA => 3 * 7,
            MODE_NORMAL => 3 * 3,
            _ => unreachable!()
        };
        let vtx_count = points_count * max_n_vtx;
        let idx_count = count * max_n_idx;
        let (mut vs, mut is, first) = self.allocate(vtx_count, idx_count, Vertex::default());

        let half_thickness = match MODE {
            MODE_THICK_AA | MODE_THIN_AA => (thickness - pixel_size) * 0.5,
            MODE_NORMAL => thickness * 0.5,
            _ => unreachable!()
        };
        let half_thickness_aa = half_thickness + pixel_size;

        let first_vtx_ptr = first;
        let mut unused_vertices = 0;
        let mut unused_indices = 0;

        let mut sqlen1 = 0.0;
        let mut dx1 = 0.0;
        let mut dy1 = 0.0;
        if closed {
            dx1 = points[0].x - points[points_count - 1].x;
            dy1 = points[0].y - points[points_count - 1].y;
            sqlen1 = dx1 * dx1 + dy1 * dy1;

            let d_len = sqlen1.sqrt();
            if d_len > 0.0 {
                let inv = 1.0 / d_len;
                dx1 *= inv;
                dy1 *= inv;
            }
        }

        let mut vi = first as usize;

        for i1 in 0..points_count {
            let p1 = &points[i1];
            let i2 = if i1 + 1 == points_count { 0 } else { i1 + 1 };
            let p2 = &points[i2];
            let mut dx2 = p1.x - p2.x;
            let mut dy2 = p1.y - p2.y;
            let mut sqlen2 = dx2 * dx2 + dy2 * dy2;

            let d_len = sqlen2.sqrt();
            if d_len > 0.0 {
                let inv = 1.0 / d_len;
                dx2 *= inv;
                dy2 *= inv;
            }

            if !closed && i1 == 0 {
                dx1 = -dx2;
                dy1 = -dy2;
                sqlen1 = sqlen2;
            } else if !closed && i1 == points_count - 1 {
                dx2 = -dx1;
                dy2 = -dy1;
                sqlen2 = sqlen1;
            }

            let miter_l_recip = dx1 * dy2 - dy1 * dx2;
            let bevel = (dx1 * dx2 + dy1 * dy2) > 1e-5;
            let mut d_sqlen = 0.0;
            let mut min_sqlen = 0.0;
            let (mlx, mly, mrx, mry) = match MODE {
                MODE_NORMAL | MODE_THICK_AA => {
                    if miter_l_recip.abs() > 1e-3 {
                        let mut miter_l = half_thickness / miter_l_recip;
                        // Limit (inner) miter so it doesn't shoot away when miter is longer than adjacent line segments on acute angles
                        if bevel {
                            // This is too aggressive (not exactly precise)
                            min_sqlen = sqlen1.min(sqlen2);
                            d_sqlen = (dx1 + dx2) * (dx1 + dx2) + (dy1 + dy2) * (dy1 + dy2);
                            let miter_sqlen = d_sqlen * miter_l * miter_l;
                            if miter_sqlen > min_sqlen {
                                miter_l *= (min_sqlen / miter_sqlen).sqrt();
                            }
                        }
                        (
                            p1.x - (dx1 + dx2) * miter_l,
                            p1.y - (dy1 + dy2) * miter_l,
                            p1.x + (dx1 + dx2) * miter_l,
                            p1.y + (dy1 + dy2) * miter_l,
                        )
                    } else {
                        // Avoid degeneracy for (nearly) straight lines
                        (
                            p1.x + dy1 * half_thickness,
                            p1.y - dx1 * half_thickness,
                            p1.x - dy1 * half_thickness,
                            p1.y + dx1 * half_thickness,
                        )
                    }
                },
                MODE_THIN_AA => {
                    (
                        p1.x,
                        p1.y,
                        0.0,
                        0.0,
                    )
                },
                _ => unreachable!()
            };
            // The two bevel vertices if the angle is right or obtuse
            // miter_sign == 1, if the outer (maybe bevelled) edge is on the right, -1 iff it is on the left
            let miter_sign = if miter_l_recip >= 0.0 { 1.0 } else { 0.0 }
                - if miter_l_recip < 0.0 { 1.0 } else { 0.0 };
            let (b1x, b1y, b2x, b2y) = if bevel {
                (
                    p1.x + (dx1 - dy1 * miter_sign) * half_thickness,
                    p1.y + (dy1 + dx1 * miter_sign) * half_thickness,
                    p1.x + (dx2 + dy2 * miter_sign) * half_thickness,
                    p1.y + (dy2 - dx2 * miter_sign) * half_thickness,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            // Populate vertices, vertex order (looking along the direction of the polyline):
            //
            // MODE_THICK_AA:
            // - left vertex*
            // - right vertex*
            // - left AA-fringe vertex*
            // - right AA-fringe vertex*
            // - extra bevel vertex (if bevel)
            // - extra bevel AA-fringe vertex (if bevel)
            // MODE_THIN_AA:
            // - middle vertex*
            // - left AA-fringe vertex*
            // - right AA-fringe vertex*
            // - extra bevel AA-fringe vertex (if bevel)
            // MODE_NORMAL:
            // - left vertex*
            // - right vertex*
            // - the remaining vertex (if bevel)
            //
            // (*) if there is bevel, these vertices are the ones on the incoming edge. Having all
            // the vertices of the incoming edge in predictable positions is important - we
            // reference them even if we don't know relevant line properties yet
            let vertex_count = match (MODE, bevel) {
                (MODE_THICK_AA, true) => 6,
                (MODE_THICK_AA, false) => 4,
                (MODE_THIN_AA, true) => 4,
                (MODE_THIN_AA, false) => 3,
                (MODE_NORMAL, true) => 3,
                (MODE_NORMAL, false) => 2,
                (_, _) => unreachable!()
            };
            let bevel_l = bevel && miter_sign < 0.0;
            let bevel_r = bevel && miter_sign > 0.0;

            // Outgoing edge bevel vertex index
            let bi = match MODE {
                MODE_THICK_AA => {
                    let (b1ax, b1ay, b2ax, b2ay) = if bevel {
                        (
                            p1.x + (dx1 - dy1 * miter_sign) * half_thickness_aa,
                            p1.y + (dy1 + dx1 * miter_sign) * half_thickness_aa,
                            p1.x + (dx2 + dy2 * miter_sign) * half_thickness_aa,
                            p1.y + (dy2 - dx2 * miter_sign) * half_thickness_aa,
                        )
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    };

                    let mut miter_al = half_thickness_aa / miter_l_recip;
                    if bevel {
                        let miter_sqlen = d_sqlen * miter_al * miter_al;
                        if miter_sqlen > min_sqlen {
                            miter_al *= (min_sqlen / miter_sqlen).sqrt();
                        }
                    }

                    let (mlax, mlay, mrax, mray) = if miter_l_recip.abs() > 1e-3 {
                        (
                            p1.x - (dx1 + dx2) * miter_al,
                            p1.y - (dy1 + dy2) * miter_al,
                            p1.x + (dx1 + dx2) * miter_al,
                            p1.y + (dy1 + dy2) * miter_al,
                        )
                    } else {
                        (
                            p1.x + dy1 * half_thickness_aa,
                            p1.y - dx1 * half_thickness_aa,
                            p1.x - dy1 * half_thickness_aa,
                            p1.y + dx1 * half_thickness_aa,
                        )
                    };
                    vs[0] = Vertex::from_pos2_color( if bevel_l { [b1x, b1y] } else { [mlx, mly] }, color);
                    vs[1] = Vertex::from_pos2_color( if bevel_r { [b1x, b1y] } else { [mrx, mry] }, color);
                    vs[2] = Vertex::from_pos2_color( if bevel_l { [b1ax, b1ay] } else { [mlax, mlay] }, color_transparent);
                    vs[3] = Vertex::from_pos2_color( if bevel_r { [b1ax, b1ay] } else { [mrax, mray] }, color_transparent);
                    if bevel {
                        vs[4] = Vertex::from_pos2_color([b2x, b2y], color);
                        vs[5] = Vertex::from_pos2_color([b2ax, b2ay], color_transparent);
                    }
                    4
                }
                MODE_THIN_AA => {
                    let (b1ax, b1ay, b2ax, b2ay) = if bevel {
                        (
                            p1.x + (dx1 - dy1 * miter_sign) * pixel_size,
                            p1.y + (dy1 + dx1 * miter_sign) * pixel_size,
                            p1.x + (dx2 + dy2 * miter_sign) * pixel_size,
                            p1.y + (dy2 - dx2 * miter_sign) * pixel_size,
                        )
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    };

                    let mut miter_al = pixel_size / miter_l_recip;
                    if bevel {
                        let miter_sqlen = d_sqlen * miter_al * miter_al;
                        if miter_sqlen > min_sqlen {
                            miter_al *= (min_sqlen / miter_sqlen).sqrt();
                        }
                    }

                    let (mlax, mlay, mrax, mray) = if miter_l_recip.abs() > 1e-3 {
                        (
                            p1.x - (dx1 + dx2) * miter_al,
                            p1.y - (dy1 + dy2) * miter_al,
                            p1.x + (dx1 + dx2) * miter_al,
                            p1.y + (dy1 + dy2) * miter_al,
                        )
                    } else {
                        (
                            p1.x + dy1 * pixel_size,
                            p1.y - dx1 * pixel_size,
                            p1.x - dy1 * pixel_size,
                            p1.y + dx1 * pixel_size,
                        )
                    };

                    vs[0] = Vertex::from_pos2_color( [mlx, mly], color_thin); vs[1] = Vertex::from_pos2_color( if bevel_l { [b1ax, b1ay] } else { [mlax, mlay] }, color_transparent);
                    vs[2] = Vertex::from_pos2_color( if bevel_r { [b1ax, b1ay] } else { [mrax, mray] }, color_transparent);
                    if bevel {
                        vs[3] = Vertex::from_pos2_color([b2ax, b2ay], color_transparent);
                    }
                    3
                }
                MODE_NORMAL => {
                    vs[0] = Vertex::from_pos2_color( if bevel_l { [b1x, b1y] } else { [mlx, mly] }, color);
                    vs[1] = Vertex::from_pos2_color( if bevel_r { [b1x, b1y] } else { [mrx, mry] }, color);
                    if bevel {
                        vs[2] = Vertex::from_pos2_color([b2x, b2y], color);
                    }
                    2
                },
                _ => unreachable!(),
            };
            unused_vertices += max_n_vtx - vertex_count;

            vs = &mut vs[vertex_count..];

            // Set the previous line direction so it doesn't need to be recomputed
            dx1 = -dx2;
            dy1 = -dy2;
            sqlen1 = sqlen2;

            if i1 < count {
                let vtx_next_id = if i1 < points_count - 1 {
                    vi + vertex_count
                } else {
                    first_vtx_ptr as usize
                };

                match MODE {
                    MODE_NORMAL => {
                        let l1i = vi + if bevel_l { bi } else { 0 };
                        let r1i = vi + if bevel_r { bi } else { 1 };
                        let l2i = vtx_next_id;
                        let r2i = vtx_next_id + 1;
                        let ebi = vi + if bevel_l { 0 } else { 1 }; // incoming edge bevel vertex index

                        is[0] = l1i as IndexType;
                        is[1] = r1i as IndexType;
                        is[2] = r2i as IndexType;
                        is[3] = l1i as IndexType;
                        is[4] = r2i as IndexType;
                        is[5] = l2i as IndexType;
                        is = &mut is[6..];

                        if bevel {
                            is[0] = l1i as IndexType;
                            is[1] = r1i as IndexType;
                            is[2] = ebi as IndexType;
                            is = &mut is[3..];
                        } else {
                            unused_indices += 3;
                        }
                    }
                    MODE_THIN_AA => {
                        let m1i = vi;
                        let m2i = vtx_next_id;

                        let l1ai = vi + if bevel_l { 3 } else { 1 };
                        let r1ai = vi + if bevel_r { 3 } else { 2 };
                        let l2ai = vtx_next_id + 1;
                        let r2ai = vtx_next_id + 2;

                        is[0] = l1ai as IndexType;
                        is[1] = m1i as IndexType;
                        is[2] = m2i as IndexType;

                        is[3] = l1ai as IndexType;
                        is[4] = m2i as IndexType;
                        is[5] = l2ai as IndexType;

                        is[6] = r1ai as IndexType;
                        is[7] = m1i as IndexType;
                        is[8] = m2i as IndexType;

                        is[9] = r1ai as IndexType;
                        is[10] = m2i as IndexType;
                        is[11] = r2ai as IndexType;

                        is = &mut is[12..];

                        if bevel {
                            is[0] = vi as u16 + if bevel_r { 0 } else { 1 };
                            is[1] = vi as u16 + if bevel_r { 2 } else { 0 };
                            is[2] = vi as u16 + if bevel_r { 3 } else { 3 };
                            is = &mut is[3..];
                        } else {
                            unused_indices += 3;
                        }

                    }
                    MODE_THICK_AA => {
                        let l1i = vi + if bevel_l { bi } else { 0 };
                        let r1i = vi + if bevel_r { bi } else { 1 };
                        let l2i = vtx_next_id;
                        let r2i = vtx_next_id + 1;
                        let ebi = vi + if bevel_l { 0 } else { 1 }; // incoming edge bevel vertex index

                        is[0] = l1i as IndexType;
                        is[1] = r1i as IndexType;
                        is[2] = r2i as IndexType;
                        is[3] = l1i as IndexType;
                        is[4] = r2i as IndexType;
                        is[5] = l2i as IndexType;
                        is = &mut is[6..];

                        if bevel {
                            is[0] = l1i as IndexType;
                            is[1] = r1i as IndexType;
                            is[2] = ebi as IndexType;
                            is = &mut is[3..];
                        } else {
                            unused_indices += 3;
                        }

                        let l1ai = vi + if bevel_l { 5 } else { 2 };
                        let r1ai = vi + if bevel_r { 5 } else { 3 };
                        let l2ai = vtx_next_id + 2;
                        let r2ai = vtx_next_id + 3;

                        is[0] = l1ai as IndexType;
                        is[1] = l1i as IndexType;
                        is[2] = l2i as IndexType;
                        is[3] = l1ai as IndexType;
                        is[4] = l2i as IndexType;
                        is[5] = l2ai as IndexType;
                        is[6] = r1ai as IndexType;
                        is[7] = r1i as IndexType;
                        is[8] = r2i as IndexType;
                        is[9] = r1ai as IndexType;
                        is[10] = r2i as IndexType;
                        is[11] = r2ai as IndexType;
                        is = &mut is[12..];

                        if bevel {
                            is[0] = vi as u16 + if bevel_r { 1 } else { 2 };
                            is[1] = vi as u16 + if bevel_r { 3 } else { 0 };
                            is[2] = vi as u16 + if bevel_r { 5 } else { 4 };

                            is[3] = vi as u16 + if bevel_r { 1 } else { 2 };
                            is[4] = vi as u16 + if bevel_r { 5 } else { 4 };
                            is[5] = vi as u16 + if bevel_r { 4 } else { 5 };
                            is = &mut is[6..];
                        } else {
                            unused_indices += 6;
                        }
                    }
                    _ => unreachable!()
                }
            }
            vi += vertex_count;
        }
        self.reclaim_allocation(unused_vertices, unused_indices);
    }

    /// Draws a connected sequence of antialised line segments passing through `points`.
    ///
    /// When `closed` is set the last and first points are connected as well.
    ///
    /// The line is drawn without caps. Caps are not antialiased.
    pub fn stroke_polyline_aa(
        &mut self,
        points: &[Vec2],
        closed: bool,
        thickness: f32,
        color: [u8; 4],
    ) {
        if thickness > self.pixel_size {
            self.stroke_polyline_internal::<MODE_THICK_AA>(points, closed, thickness, color)
        } else {
            self.stroke_polyline_internal::<MODE_THIN_AA>(points, closed, thickness, color)
        }
    }

    /// Draws a connected sequence of line segments passing through `points`.
    ///
    /// When `closed` is set the last and first points are connected as well.
    ///
    /// The line is drawn without any caps.
    pub fn stroke_polyline(
        &mut self,
        points: &[Vec2],
        closed: bool,
        thickness: f32,
        color: [u8; 4],
    ) {
        self.stroke_polyline_internal::<MODE_NORMAL>(points, closed, thickness, color)
    }

    #[doc(hidden)]
    pub fn stroke_polyline_variable_aa(&mut self, points: &[Vec2], radius: &[f32], color: [u8; 4]) {
        if points.len() < 2 {
            return;
        }
        assert!(points.len() == radius.len());
        let count = points.len() - 1;

        let gradient_size = self.pixel_size;
        let half_gradient = gradient_size * 0.5;
        let alpha_transparent = 0;
        let color_transparent = [color[0], color[1], color[2], alpha_transparent];
        let index_count = count * 18;
        let vertex_count = points.len() * 4;

        // move out temporary buffers as self.allocate borrows self
        let mut temp_normals = take(&mut self.temp_normals);
        let mut temp_points = take(&mut self.temp_points);

        let (vs, is, first) = self.allocate(vertex_count, index_count, Vertex::default());
        temp_normals.resize(points.len(), vec2(0., 0.));
        temp_points.resize(points.len() * 4, vec2(0., 0.));
        for i1 in 0..count {
            let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
            let mut delta = points[i2] - points[i1];
            let len2 = delta.dot(delta);
            if len2 > 0.0 {
                let len = len2.sqrt();
                delta /= len;
            }
            temp_normals[i1] = vec2(delta.y, -delta.x);
        }
        temp_normals[points.len() - 1] = temp_normals[points.len() - 2];

        let half_inner_thickness = radius[0] - half_gradient;
        temp_points[0] = points[0] + temp_normals[0] * (half_inner_thickness + gradient_size);
        temp_points[1] = points[0] + temp_normals[0] * half_inner_thickness;
        temp_points[2] = points[0] - temp_normals[0] * half_inner_thickness;
        temp_points[3] = points[0] - temp_normals[0] * (half_inner_thickness + gradient_size);

        let half_inner_thickness = radius[points.len() - 1] - half_gradient;
        temp_points[(points.len() - 1) * 4 + 0] = points[points.len() - 1]
            + temp_normals[points.len() - 1] * (half_inner_thickness + gradient_size);
        temp_points[(points.len() - 1) * 4 + 1] =
            points[points.len() - 1] + temp_normals[points.len() - 1] * (half_inner_thickness);
        temp_points[(points.len() - 1) * 4 + 2] =
            points[points.len() - 1] - temp_normals[points.len() - 1] * (half_inner_thickness);
        temp_points[(points.len() - 1) * 4 + 3] = points[points.len() - 1]
            - temp_normals[points.len() - 1] * (half_inner_thickness + gradient_size);

        let mut idx1 = first;
        for i1 in 0..count {
            let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
            let idx2 = if (i1 + 1) == points.len() {
                first
            } else {
                idx1 + 4
            };
            let half_inner_thickness = radius[i2] - half_gradient;
            let mut dm = (temp_normals[i1] + temp_normals[i2]) * 0.5;
            let mut dm_len2 = dm.dot(dm);
            if dm_len2 < 0.5 {
                dm_len2 = 0.5;
            }
            let inv_len2 = 1.0 / dm_len2;
            dm *= inv_len2;
            let dm_out = dm * (half_inner_thickness + gradient_size);
            let dm_in = dm * half_inner_thickness;

            // points
            temp_points[i2 * 4 + 0] = points[i2] + dm_out;
            temp_points[i2 * 4 + 1] = points[i2] + dm_in;
            temp_points[i2 * 4 + 2] = points[i2] - dm_in;
            temp_points[i2 * 4 + 3] = points[i2] - dm_out;

            // indices
            is[18 * i1..18 * (i1 + 1)].copy_from_slice(&[
                idx2 + 1,
                idx1 + 1,
                idx1 + 2,
                idx1 + 2,
                idx2 + 2,
                idx2 + 1,
                idx2 + 1,
                idx1 + 1,
                idx1 + 0,
                idx1 + 0,
                idx2 + 0,
                idx2 + 1,
                idx2 + 2,
                idx1 + 2,
                idx1 + 3,
                idx1 + 3,
                idx2 + 3,
                idx2 + 2,
            ]);
            idx1 = idx2;
        }

        for i in 0..points.len() {
            vs[i * 4 + 0] = Vertex::from_pos2_color(temp_points[i * 4 + 0].into(), color_transparent);
            vs[i * 4 + 1] = Vertex::from_pos2_color(temp_points[i * 4 + 1].into(), color);
            vs[i * 4 + 2] = Vertex::from_pos2_color(temp_points[i * 4 + 2].into(), color);
            vs[i * 4 + 3] = Vertex::from_pos2_color(temp_points[i * 4 + 3].into(), color_transparent);
        }

        // return temporary buffers
        self.temp_normals = temp_normals;
        self.temp_points = temp_points;
    }

    #[doc(hidden)]
    pub fn stroke_capsule_chain_aa(&mut self, points: &[Vec2], radius: &[f32], color: [u8; 4]) {
        // TODO: optimal non-overlapping implementation
        self.stroke_polyline_variable_aa(points, radius, color);
        for (&point, &r) in points.iter().zip(radius.iter()) {
            self.fill_circle_aa(point, r, r.ceil() as usize * 3, color);
        }
    }
}

impl<Vertex: Copy + Default + FromPos2Color> GeometryBatch<Vertex> {
    pub fn fill_convex_polygon_aa(&mut self, points: &[Vec2], color: [u8; 4]) {
        let mut temp_normals = take(&mut self.temp_normals);
        let half_pixel_size = self.pixel_size * 0.5;
        let color_trans = [color[0], color[1], color[2], 0];

        let num_vertices = points.len() * 2;
        let num_indices = (points.len() - 2) * 3 + points.len() * 6;
        let (vs, is, first) = self.allocate(num_vertices, num_indices, Vertex::default());

        // indices for inner fill
        for i in 0..points.len() - 2 {
            is[i * 3 + 0] = first as IndexType;
            is[i * 3 + 1] = first + ((i + 1) * 2) as IndexType;
            is[i * 3 + 2] = first + ((i + 2) * 2) as IndexType;
        }

        // calculate normals
        temp_normals.clear();
        temp_normals.reserve(points.len());
        for i in 0..points.len() {
            let p0 = points[i];
            let p1 = points[(i + 1) % points.len()];
            let delta = p1 - p0;
            temp_normals.push(-delta.perp().try_normalize().unwrap_or(vec2(0.0, 0.0)));
        }

        let fringe_base = (points.len() - 2) * 3;
        for i in 0..points.len() {
            let ip = (i + points.len() - 1) % points.len();
            let n0 = temp_normals[ip];
            let n1 = temp_normals[i];
            let hn = (n0 + n1).try_normalize().unwrap_or(vec2(0.0, 0.0)) * half_pixel_size;

            // inner vertex followed by outer vertex
            vs[i * 2 + 0] = Vertex::from_pos2_color([points[i].x - hn.x, points[i].y - hn.y], color);
            vs[i * 2 + 1] = Vertex::from_pos2_color([points[i].x + hn.x, points[i].y + hn.y], color_trans);

            // indices for surrounding blended faceloop
            let base = fringe_base + i * 6;
            is[base + 0] = first + 0 + 2 * i  as IndexType;
            is[base + 1] = first + 0 + 2 * ip as IndexType;
            is[base + 2] = first + 1 + 2 * ip as IndexType;
            is[base + 3] = first + 1 + 2 * ip as IndexType;
            is[base + 4] = first + 1 + 2 * i  as IndexType;
            is[base + 5] = first + 0 + 2 * i  as IndexType;
        }

        self.temp_normals = temp_normals;
    }

    /// Fills a visibility-polygon.
    ///
    /// Visibility polygon is a star-shaped polygon. It does not need to be convex but all
    /// vertices should be directly "visible" from the `center` point.
    pub fn fill_visibility_polygon_aa(&mut self, points: &[Vec2], center: Vec2, color: [u8; 4]) {
        let mut temp_normals = take(&mut self.temp_normals);
        let half_pixel_size = self.pixel_size * 0.5;
        let color_trans = [color[0], color[1], color[2], 0];

        let num_vertices = 1 + points.len() * 2;
        let num_indices = points.len() * 3 + points.len() * 6;
        let (vs, is, first) = self.allocate(num_vertices, num_indices, Vertex::default());

        // indices for inner fill
        let center_index = first + 2 * points.len() as IndexType;
        let num_points2 = points.len() as IndexType * 2;
        for i in 0..points.len() {
            is[i * 3 + 0] = center_index;
            is[i * 3 + 1] = first + ((i as IndexType + 0) * 2);
            is[i * 3 + 2] = first + ((i as IndexType + 1) * 2) % num_points2;
        }

        // calculate normals
        temp_normals.clear();
        temp_normals.reserve(points.len());
        for i in 0..points.len() {
            let p0 = points[i];
            let p1 = points[(i + 1) % points.len()];
            let delta = p1 - p0;
            temp_normals.push(-delta.perp().try_normalize().unwrap_or(vec2(0.0, 0.0)));
        }

        let fringe_base = points.len() * 3;
        for i in 0..points.len() {
            let ip = (i + points.len() - 1) % points.len();
            let n0 = temp_normals[ip];
            let n1 = temp_normals[i];
            let hn = (n0 + n1).try_normalize().unwrap_or(vec2(0.0, 0.0)) * half_pixel_size;

            // inner vertex followed by outer vertex
            vs[i * 2 + 0] = Vertex::from_pos2_color([points[i].x - hn.x, points[i].y - hn.y], color);
            vs[i * 2 + 1] = Vertex::from_pos2_color([points[i].x + hn.x, points[i].y + hn.y], color_trans);

            // indices for surrounding blended faceloop
            let base = fringe_base + i * 6;
            is[base + 0] = first + 0 + 2 * i  as IndexType;
            is[base + 1] = first + 0 + 2 * ip as IndexType;
            is[base + 2] = first + 1 + 2 * ip as IndexType;
            is[base + 3] = first + 1 + 2 * ip as IndexType;
            is[base + 4] = first + 1 + 2 * i  as IndexType;
            is[base + 5] = first + 0 + 2 * i  as IndexType;
        }
        vs[points.len() * 2] = Vertex::from_pos2_color([center.x, center.y], color);

        self.temp_normals = temp_normals;
    }

    pub fn fill_convex_polygon(&mut self, points: &[Vec2], color: [u8; 4]) {
        let num_vertices = points.len();
        let num_indices = (points.len() - 2) * 3;
        let (vs, is, first) = self.allocate(num_vertices, num_indices, Vertex::default());
        for i in 0..points.len() {
            vs[i] = Vertex::from_pos2_color([points[i].x, points[i].y], color);
        }
        for i in 0..points.len() - 2 {
            is[i * 3 + 0] = first as IndexType;
            is[i * 3 + 1] = first + (i + 1) as IndexType;
            is[i * 3 + 2] = first + (i + 2) as IndexType;
        }
    }

    pub fn add_position_indices(
        &mut self,
        positions: &[Vec2],
        indices: &[IndexType],
        color: [u8; 4],
    ) {
        let (vs, is, first) = self.allocate(positions.len(), indices.len(), Vertex::default());
        for (dest, pos) in vs.iter_mut().zip(positions) {
            *dest = Vertex::from_pos2_color([pos.x, pos.y], color);
        }
        for (dest, index) in is.iter_mut().zip(indices) {
            *dest = index + first;
        }
    }
    pub fn fill_rect(&mut self, start: Vec2, end: Vec2, color: [u8; 4]) {
        let (vs, is, first) = self.allocate(4, 6, Vertex::default());
        vs[0] = Vertex::from_pos2_color([start.x, start.y], color);
        vs[1] = Vertex::from_pos2_color([end.x, start.y], color);
        vs[2] = Vertex::from_pos2_color([end.x, end.y], color);
        vs[3] = Vertex::from_pos2_color([start.x, end.y], color);

        is[0] = first + 0;
        is[1] = first + 1;
        is[2] = first + 2;
        is[3] = first + 0;
        is[4] = first + 2;
        is[5] = first + 3;
    }

    pub fn stroke_rect(&mut self, start: Vec2, end: Vec2, thickness: f32, color: [u8; 4]) {
        let (vs, indices, first) = self.allocate(8, 24, Vertex::default());

        let ht = thickness * 0.5;
        let ht = vec2(ht, ht);
        let os = start - ht;
        let oe = end + ht;
        let is = start + ht;
        let ie = end - ht;

        vs[0] = Vertex::from_pos2_color([os.x, os.y], color);
        vs[1] = Vertex::from_pos2_color([oe.x, os.y], color);
        vs[2] = Vertex::from_pos2_color([oe.x, oe.y], color);
        vs[3] = Vertex::from_pos2_color([os.x, oe.y], color);
        vs[4] = Vertex::from_pos2_color([is.x, is.y], color);
        vs[5] = Vertex::from_pos2_color([ie.x, is.y], color);
        vs[6] = Vertex::from_pos2_color([ie.x, ie.y], color);
        vs[7] = Vertex::from_pos2_color([is.x, ie.y], color);

        // 0         1
        //   4     5

        //   7     6
        // 3         2

        indices[0] = first + 0;
        indices[1] = first + 1;
        indices[2] = first + 4;
        indices[3] = first + 4;
        indices[4] = first + 1;
        indices[5] = first + 5;

        indices[6] = first + 1;
        indices[7] = first + 2;
        indices[8] = first + 5;
        indices[9] = first + 5;
        indices[10] = first + 2;
        indices[11] = first + 6;

        indices[12] = first + 2;
        indices[13] = first + 3;
        indices[14] = first + 7;
        indices[15] = first + 2;
        indices[16] = first + 7;
        indices[17] = first + 6;

        indices[18] = first + 3;
        indices[19] = first + 0;
        indices[20] = first + 4;
        indices[21] = first + 3;
        indices[22] = first + 4;
        indices[23] = first + 7;
    }

    /// Adds an antialiased rounded rectangle outline between `start` and `end` of `color`. Corners
    /// are round with radius `corner_radius`. Each corner is built up of `corner_points`-1 linear
    /// segments.
    pub fn stroke_round_rect_aa(&mut self, start: Vec2, end: Vec2, corner_radius: f32, corner_points: usize, thickness: f32, color: [u8; 4]) {
        let mut temp_path = take(&mut self.temp_path);
        temp_path.clear();

        path::add_rounded_rect(&mut temp_path, start, end, corner_radius, corner_points);
        self.stroke_polyline_aa(&temp_path, true, thickness, color);

        self.temp_path = temp_path;
    }
    
    /// Adds a rectangle outline with rounded corners
    pub fn stroke_round_rect(&mut self, start: Vec2, end: Vec2, corner_radius: f32, corner_points: usize, thickness: f32, color: [u8; 4]) {
        let mut temp_path = take(&mut self.temp_path);
        temp_path.clear();

        path::add_rounded_rect(&mut temp_path, start, end, corner_radius, corner_points);
        self.stroke_polyline(&temp_path, true, thickness, color);

        self.temp_path = temp_path;
    }
    
    /// Adds an antialiased rounded rectangle between `start` and `end` of `color`. Corners
    /// are round with radius `corner_radius`. Each corner is built up of `corner_points`-1 linear
    /// segments.
    pub fn fill_round_rect_aa(&mut self, start: Vec2, end: Vec2, corner_radius: f32, corner_points: usize, color: [u8; 4]) {
        let mut temp_path = take(&mut self.temp_path);
        temp_path.clear();

        path::add_rounded_rect(&mut temp_path, start, end, corner_radius, corner_points);
        self.fill_convex_polygon_aa(&temp_path, color);

        self.temp_path = temp_path;
    }
    
    /// Adds a rounded rectangle between `start` and `end` of `color`. Corners
    /// are round with radius `corner_radius`. Each corner is built up of `corner_points`-1 linear
    /// segments.
    pub fn fill_round_rect(&mut self, start: Vec2, end: Vec2, corner_radius: f32, corner_points: usize, color: [u8; 4]) {
        let mut temp_path = take(&mut self.temp_path);
        temp_path.clear();

        path::add_rounded_rect(&mut temp_path, start, end, corner_radius, corner_points);
        self.fill_convex_polygon(&temp_path, color);

        self.temp_path = temp_path;
    }
}

impl<Vertex: Copy + Default + FromPos3Color> GeometryBatch<Vertex> {
    pub fn add_box(&mut self, center: [f32; 3], size: [f32; 3], color: [u8; 4]) {
        let (vs, is, first) = self.allocate(8, 36, Vertex::default());
        for (v, i) in vs.iter_mut().zip(0..8) {
            *v = Vertex::from_pos3_color([
                center[0] + size[0] * ((i & 1) as f32 - 1.0f32),
                center[1] + size[1] * (((i & 2) >> 1) as f32 - 1.0f32),
                center[2] + size[2] * (((i & 4) >> 2) as f32 - 1.0f32),
            ], color);
        }

        let cube_indices: [IndexType; 36] = [
            0, 1, 2, 0, 2, 3, // front
            4, 5, 6, 4, 6, 7, // back
            8, 9, 10, 8, 10, 11, // top
            12, 13, 14, 12, 14, 15, // bottom
            16, 17, 18, 16, 18, 19, // right
            20, 21, 22, 20, 22, 23, // left
        ];
        for (dest, index) in is.iter_mut().zip(cube_indices.iter()) {
            *dest = index + first;
        }
    }

    pub fn add_position3_indices(
        &mut self,
        positions: &[[f32; 3]],
        indices: &[IndexType],
        color: [u8; 4],
    ) {
        let (vs, is, first) = self.allocate(positions.len(), indices.len(), Vertex::default());
        for (dest, pos) in vs.iter_mut().zip(positions) {
            *dest = Vertex::from_pos3_color(*pos, color);
        }
        for (dest, index) in is.iter_mut().zip(indices) {
            *dest = index + first;
        }
    }
}

impl<Vertex: Default + Copy + FromPos2ColorUV> GeometryBatch<Vertex> {
    pub fn fill_rect_uv(&mut self, rect: [f32; 4], uv: [f32; 4], color: [u8; 4]) -> &mut [Vertex] {
        let (vs, is, first) = self.allocate(4, 6, Vertex::default());

        vs[0] = Vertex::from_pos2_color_uv([rect[0], rect[1]], color, [uv[0], uv[1]]);
        vs[1] = Vertex::from_pos2_color_uv([rect[2], rect[1]], color, [uv[2], uv[1]]);
        vs[2] = Vertex::from_pos2_color_uv([rect[2], rect[3]], color, [uv[2], uv[3]]);
        vs[3] = Vertex::from_pos2_color_uv([rect[0], rect[3]], color, [uv[0], uv[3]]);

        is[0] = first + 0;
        is[1] = first + 1;
        is[2] = first + 2;
        is[3] = first + 0;
        is[4] = first + 2;
        is[5] = first + 3;
        vs
    }

    pub fn add_grid(
        &mut self,
        center: [f32; 2],
        extents: [f32; 2],
        divisions: [IndexType; 2],
        uv_rect: [f32; 4],
        color: [u8; 4],
    ) {
        let num_points = (divisions[0] * divisions[1]) as usize;
        let (vs, is, first) = self.allocate(num_points, 6 * num_points, Vertex::default());

        for j in 0..=divisions[1] {
            for i in 0..=divisions[0] {
                let v = &mut vs[(j * (divisions[0] + 1) + i) as usize];
                let x_f = i as f32 / divisions[0] as f32;
                let y_f = j as f32 / divisions[1] as f32;
                *v = Vertex::from_pos2_color_uv([
                    (x_f - 0.5f32) * extents[0] + center[0],
                    (y_f - 0.5f32) * extents[1] + center[1],
                ], color, [
                    uv_rect[0] * x_f + uv_rect[2] * (1.0f32 - x_f),
                    uv_rect[1] * x_f + uv_rect[3] * (1.0f32 - y_f),
                ]);
            }

            for j in 0..divisions[1] {
                for i in 0..divisions[0] {
                    let inds = [
                        first + j * (divisions[0] + 1) + i,
                        first + (j + 1) * (divisions[0] + 1) + i,
                        first + (j + 1) * (divisions[0] + 1) + i + 1,
                        first + j * (divisions[0] + 1) + (i + 1),
                    ];
                    let from = (j * divisions[0] + i) as usize;
                    is[from..from + 6]
                        .copy_from_slice(&[inds[0], inds[1], inds[2], inds[0], inds[2], inds[3]]);
                }
            }
        }
    }
}

/// Routines for constructing paths in `Vec<Vec2>`
pub mod path {
    use super::{Vec2, vec2, PI};

    /// Adds an arc consisting of `num_points`, at `center` and `radius`. Between `angle_start` and `angle_end`.
    pub fn add_arc(path: &mut Vec<Vec2>, center: Vec2, radius: f32, angle_start: f32, angle_end: f32, num_points: usize) {
        if num_points < 2 {
            return;
        }
        path.reserve(num_points);
        for i in 0..num_points {
            let f = angle_start + (angle_end - angle_start) * (i as f32 / (num_points - 1) as f32);
            path.push(center + vec2(f.cos(), f.sin()) * radius);
        }
    }

    /// Adds a rounded rectangle between `start` and `end` of `color`. Corner radius is `corner_radius`, each
    /// corner is built up of linear segments `corner_points`.
    pub fn add_rounded_rect(out: &mut Vec<Vec2>, start: Vec2, end: Vec2, corner_radius: f32, corner_points: usize) {
        if corner_points >= 2 {
            out.reserve(corner_points * 4);
            add_arc(out, start + vec2(corner_radius, corner_radius), corner_radius, -PI, -PI * 0.5, corner_points);
            add_arc(out, vec2(end.x, start.y) + vec2(-corner_radius, corner_radius), corner_radius, -PI * 0.5, 0.0, corner_points);
            add_arc(out, end + vec2(-corner_radius, -corner_radius), corner_radius, 0.0, PI * 0.5, corner_points);
            add_arc(out, vec2(start.x, end.y) + vec2(corner_radius, -corner_radius), corner_radius, PI * 0.5, PI, corner_points);
        } else {
            out.reserve(4);
            out.push(start);
            out.push(vec2(end.x, start.y));
            out.push(end);
            out.push(vec2(start.x, end.y));
        }
    }
}

/// Construct a vertex from 2D-position + color.
///
/// Implement this trait for your vertex type.
pub trait FromPos2Color {
    fn from_pos2_color(pos: [f32; 2], color: [u8; 4])->Self;
}

/// Construct a vertex from 3D-position + color.
///
/// Implement this trait for your vertex type.
pub trait FromPos3Color {
    fn from_pos3_color(pos: [f32; 3], color: [u8; 4])->Self;
}

/// Construct a vertex from 2D-position + color + uv.
///
/// Implement this trait for your vertex type.
pub trait FromPos2ColorUV {
    fn from_pos2_color_uv(pos: [f32; 2], color: [u8; 4], uv: [f32; 2])->Self;
}

/// Individual command in a drawing list.
pub enum GeometryCommand {
    /// Starts a new index buffer
    Indices { num_indices: usize },
    /// Start new vertex buffer
    Vertices { num_vertices: usize },
    /// Performs actual draw call
    DrawCall { num_indices: usize },
}
