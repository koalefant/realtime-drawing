//! A collection of fast drawing routines aimed at interactive applications and games.
//!
//! # Features
//! - Optimized for dynamically generated content.
//! - Local-antialiasing using blended strips.
//! - GPU rendering: output to streamed vertex/index buffers.
//! - Aggressive batching across primitive types.
//! - Backend-agnostic. Comes with [`MiniquadBatch`] that implements [`miniquad`]-backend out of the box. Easy integration into existing engines.
//! - Works with custom vertex format through traits. See [`VertexPos2`], [`VertexPos3`], [`VertexColor`] etc.
//! - Can be used with 16-bit indices (to reduce memory bandwidth) and update multiple buffers when reaching 65K vertex/index limits.
//! - Easy to extend with custom traits.
//! - WebAssembly support.
//! - Pure rust, no unsafe code.
//!
//! Individual drawing operations such as [`GeometryBatch::add_circle`] or
//! [`GeometryBatch::add_polyline`] are available in [`GeometryBatch`] implementation.
//!
/// [`miniquad`]: https://docs.rs/miniquad/
mod example;
mod miniquad_batch;
pub use miniquad_batch::MiniquadBatch;
pub use example::VertexPos3UvColor;

use core::default::Default;
use core::iter::Iterator;
use core::marker::Copy;
use glam::vec2;
use glam::Vec2;

type IndexType = u16;

/// Hosts drawing routines.
///
/// `add_`-methods are used to draw individual primitives.
/// [`GeometryBatch::finish_commands`] is used to finalize buffers and backend-specific
/// call is used for the actual rendering.
///
/// # Example
///
/// ```
/// // initialization stage
/// let geometry = GeometryBatch::new(1024, 1024);
/// geometry.clear();
/// geometry.add_circle::<true>(vec2(512.0, 512.0), 128.0, 64);
/// geometry.add_polyline::<true>(&[vec2(384.0, 512.0), vec2(512.0, 512.0), vec2(512.0, 384.0)],
///                               [255, 0, 0, 255], true, 2.0);
///
/// // upload geometry.vertices/indices and render according to geometry.commands
///
/// ```
/// # Internals
/// Drawn primitives are accumulated in [`vertices`](`GeometryBatch::vertices`),
/// [`indices`](`GeometryBatch::indices`), and [`commands`](`GeometryBatch::commands`).
///
/// See a particlar backend for actual rendering: [`MiniquadBatch`]
///
pub struct GeometryBatch<Vertex: Copy> {
    /// batched vertices
    pub vertices: Vec<Vertex>,
    /// batched indices
    pub indices: Vec<IndexType>,
    /// list of drawing commands
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
}

impl<Vertex: Copy> GeometryBatch<Vertex> {
    pub fn new(max_buffer_vertices: usize, max_buffer_indices: usize) -> Self {
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

impl<Vertex: Copy + Default + VertexPos2 + VertexColor> GeometryBatch<Vertex> {
    #[inline]
    pub fn add_circle<const ANTIALIAS: bool>(
        &mut self,
        center: Vec2,
        radius: f32,
        num_segments: usize,
        def: Vertex,
    ) {
        let pixel = self.pixel_size;
        let half_pixel = pixel * 0.5;
        let alpha = def.alpha() as f32;
        let (vs, is, first) = self.allocate(2 * num_segments + 1, num_segments * 9, def);
        for (i, pair) in vs.chunks_mut(2).enumerate() {
            let angle = i as f32 / num_segments as f32 * 2.0 * std::f32::consts::PI;
            let cos = angle.cos();
            let sin = angle.sin();
            for (v, p) in pair.iter_mut().zip(&[0.0, pixel]) {
                let pos = center + vec2(cos, sin) * (radius - half_pixel + p);
                v.set_pos(pos.into());
                v.set_alpha(((1.0 - p) * alpha) as u8);
            }
        }
        vs.last_mut().unwrap().set_pos([center[0], center[1]]);
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

    #[inline]
    pub fn add_circle_outline<const ANTIALIAS: bool>(
        &mut self,
        center: Vec2,
        radius: f32,
        thickness: f32,
        num_segments: usize,
        def: Vertex,
    ) {
        let pixel_size = self.pixel_size;
        if thickness > pixel_size {
            let (vs, is, first) = self.allocate(4 * num_segments, num_segments * 18, def);
            let ht = (thickness - pixel_size) * 0.5;
            for (i, pair) in vs.chunks_mut(4).enumerate() {
                let angle = i as f32 / num_segments as f32 * 2.0 * std::f32::consts::PI;
                let cos = angle.cos();
                let sin = angle.sin();
                for (v, p) in pair.iter_mut().zip(&[
                    (-ht - pixel_size, 0),
                    (-ht, 255),
                    (ht, 255),
                    (ht + pixel_size, 0),
                ]) {
                    v.set_pos([
                        (center[0] + cos * (radius + p.0)).into(),
                        (center[1] + sin * (radius + p.0)).into(),
                    ]);
                    v.set_alpha(p.1);
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
            let (vs, is, first) = self.allocate(4 * num_segments, num_segments * 12, def);
            for (i, pair) in vs.chunks_mut(4).enumerate() {
                let angle = i as f32 / num_segments as f32 * 2.0 * std::f32::consts::PI;
                let cos = angle.cos();
                let sin = angle.sin();
                for (v, p) in pair.iter_mut().zip(&[
                    (-pixel_size, 0),
                    (0.0, (255.0 * thickness) as u8),
                    (pixel_size, 0),
                ]) {
                    v.set_pos([
                        center[0] + cos * (radius + p.0),
                        center[1] + sin * (radius + p.0),
                    ]);
                    v.set_alpha(p.1);
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

    // Coordinates are assumed to be pixel.
    pub fn add_polyline<const ANTIALIAS: bool>(
        &mut self,
        points: &[Vec2],
        color: [u8; 4],
        closed: bool,
        thickness: f32,
    ) {
        // based on AddPoyline from Dear ImGui by Omar Cornut (MIT)
        if points.len() < 2 {
            return;
        }
        let count = match closed {
            true => points.len(),
            false => points.len() - 1,
        };
        let thick_line = thickness > self.pixel_size;

        let gradient_size = self.pixel_size;
        let color_transparent = [color[0], color[1], color[2], 0];
        let index_count = if thick_line { count * 18 } else { count * 12 };
        let vertex_count = if thick_line {
            points.len() * 4
        } else {
            points.len() * 3
        };
        let (vs, is, first) = self.allocate(vertex_count, index_count, Vertex::default());
        let mut temp_normals = Vec::new();
        let mut temp_points = Vec::new();
        temp_normals.resize(points.len(), vec2(0., 0.));
        temp_points.resize(points.len() * if thick_line { 4 } else { 2 }, vec2(0., 0.));
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
        if !closed {
            temp_normals[points.len() - 1] = temp_normals[points.len() - 2];
        }
        if !thick_line {
            if !closed {
                temp_points[0] = points[0] + temp_normals[0] * gradient_size;
                temp_points[1] = points[1] - temp_normals[1] * gradient_size;

                temp_points[(points.len() - 1) * 2 + 0] =
                    points[points.len() - 1] + temp_normals[points.len() - 1] * gradient_size;
                temp_points[(points.len() - 1) * 2 + 1] =
                    points[points.len() - 1] - temp_normals[points.len() - 1] * gradient_size;
            }

            let mut idx1 = first;
            for i1 in 0..count {
                let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
                let idx2 = if (i1 + 1) == points.len() {
                    first
                } else {
                    idx1 + 3
                };

                let mut dm = (temp_normals[i1] + temp_normals[i2]) * 0.5;
                // average normals
                let mut dm_len2 = dm.dot(dm);
                if dm_len2 < 0.5 {
                    dm_len2 = 0.5;
                }
                let inv_len2 = gradient_size / dm_len2;
                dm *= inv_len2;

                // compute points
                temp_points[i2 * 2 + 0] = points[i2] + dm;
                temp_points[i2 * 2 + 1] = points[i2] - dm;

                // indices
                is[i1 * 12..(i1 + 1) * 12].copy_from_slice(&[
                    idx2 + 0,
                    idx1 + 0,
                    idx1 + 2,
                    idx1 + 2,
                    idx2 + 2,
                    idx2 + 0,
                    idx2 + 1,
                    idx1 + 1,
                    idx1 + 0,
                    idx1 + 0,
                    idx2 + 0,
                    idx2 + 1,
                ]);

                idx1 = idx2;
            }

            for i in 0..points.len() {
                vs[i * 3 + 0].set_pos(points[i].into());
                vs[i * 3 + 0].set_color(color);
                vs[i * 3 + 1].set_pos(temp_points[i * 2 + 0].into());
                vs[i * 3 + 1].set_color(color_transparent);
                vs[i * 3 + 2].set_pos(temp_points[i * 2 + 1].into());
                vs[i * 3 + 2].set_color(color_transparent);
            }
        } else {
            let half_inner_thickness = (thickness - gradient_size) * 0.5;
            if !closed {
                temp_points[0] =
                    points[0] + temp_normals[0] * (half_inner_thickness + gradient_size);
                temp_points[1] = points[0] + temp_normals[0] * half_inner_thickness;
                temp_points[2] = points[0] - temp_normals[0] * half_inner_thickness;
                temp_points[3] =
                    points[0] - temp_normals[0] * (half_inner_thickness + gradient_size);

                temp_points[(points.len() - 1) * 4 + 0] = points[points.len() - 1]
                    + temp_normals[points.len() - 1] * (half_inner_thickness + gradient_size);
                temp_points[(points.len() - 1) * 4 + 1] = points[points.len() - 1]
                    + temp_normals[points.len() - 1] * (half_inner_thickness);
                temp_points[(points.len() - 1) * 4 + 2] = points[points.len() - 1]
                    - temp_normals[points.len() - 1] * (half_inner_thickness);
                temp_points[(points.len() - 1) * 4 + 3] = points[points.len() - 1]
                    - temp_normals[points.len() - 1] * (half_inner_thickness + gradient_size);
            }

            let mut idx1 = first;
            for i1 in 0..count {
                let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
                let idx2 = if (i1 + 1) == points.len() {
                    first
                } else {
                    idx1 + 4
                };

                let mut dm = temp_normals[i1] + temp_normals[i2] * 0.5;

                // direction of first edge
                let v0 = vec2(-temp_normals[i1].y, temp_normals[i1].x);

                // project direction of first edge on second edge normal
                if closed || i2 != count {
                    let dot = v0.dot(temp_normals[i2]);
                    // Negative direction of 2nd edge
                    let v1 = vec2(temp_normals[i2].y, -temp_normals[i2].x);
                    // Scale
                    dm = (v0 + v1) / dot;
                } else {
                    let mut dm_len2 = dm.dot(dm);
                    if dm_len2 < 0.5 {
                        dm_len2 = 0.5;
                    }
                    let inv_len2 = 1.0 / dm_len2;
                    dm *= inv_len2;
                }

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
                vs[i * 4 + 0].set_pos(temp_points[i * 4 + 0].into());
                vs[i * 4 + 0].set_color(color_transparent);

                vs[i * 4 + 1].set_pos(temp_points[i * 4 + 1].into());
                vs[i * 4 + 1].set_color(color);

                vs[i * 4 + 2].set_pos(temp_points[i * 4 + 2].into());
                vs[i * 4 + 2].set_color(color);

                vs[i * 4 + 3].set_pos(temp_points[i * 4 + 3].into());
                vs[i * 4 + 3].set_color(color_transparent);
            }
        }
    }

    pub fn add_polyline_miter<const ANTIALIAS: bool>(
        &mut self,
        points: &[Vec2],
        color: [u8; 4],
        closed: bool,
        thickness: f32,
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
        let col_trans = [color[0], color[1], color[2], 0];

        let mut v = Vertex::default();
        v.set_color(color);

        if ANTIALIAS && thickness <= pixel_size {
            // Anti-aliased stroke approximation
            let idx_count = count * 12;
            let vtx_count = count * 6;
            let (vs, is, first_index) = self.allocate(vtx_count, idx_count, v);
            let mut col_faded = color;
            col_faded[3] = (color[3] as f32 * thickness) as u8;

            for i1 in 0..count {
                let i2 = if (i1 + 1) == points_count { 0 } else { i1 + 1 };
                let p1 = &points[i1];
                let p2 = &points[i2];

                let mut dx = p2.x - p1.x;
                let mut dy = p2.y - p1.y;
                let d_len = (dx * dx + dy * dy).sqrt();
                if d_len > 0.0 {
                    let inv = pixel_size / d_len;
                    dx *= inv;
                    dy *= inv;
                }

                let current_vertex = i1 * 6;
                vs[current_vertex + 0].set_pos([p1.x + dy, p1.y - dx]);
                vs[current_vertex + 0].set_color(col_trans);
                vs[current_vertex + 1].set_pos([p1.x, p1.y]);
                vs[current_vertex + 1].set_color(col_faded);
                vs[current_vertex + 2].set_pos([p1.x - dy, p1.y + dx]);
                vs[current_vertex + 2].set_color(col_trans);
                vs[current_vertex + 3].set_pos([p2.x + dy, p2.y - dx]);
                vs[current_vertex + 3].set_color(col_trans);
                vs[current_vertex + 4].set_pos([p2.x, p2.y]);
                vs[current_vertex + 4].set_color(col_faded);
                vs[current_vertex + 5].set_pos([p2.x - dy, p2.y + dx]);
                vs[current_vertex + 5].set_color(col_trans);

                let current_vertex = first_index + current_vertex as u16;
                let current_index = i1 * 12;
                is[current_index + 0] = current_vertex;
                is[current_index + 1] = current_vertex + 1;
                is[current_index + 2] = current_vertex + 4;
                is[current_index + 3] = current_vertex;
                is[current_index + 4] = current_vertex + 4;
                is[current_index + 5] = current_vertex + 3;
                is[current_index + 6] = current_vertex + 1;
                is[current_index + 7] = current_vertex + 2;
                is[current_index + 8] = current_vertex + 5;
                is[current_index + 9] = current_vertex + 1;
                is[current_index + 10] = current_vertex + 5;
                is[current_index + 11] = current_vertex + 4;
            }
        } else {
            // Precise line with bevels on acute angles
            let max_n_vtx = if ANTIALIAS { 6 } else { 3 };
            let max_n_idx = 3 * if ANTIALIAS { 9 } else { 3 };
            let vtx_count = points_count * max_n_vtx;
            let idx_count = count * max_n_idx;
            let (mut vs, mut is, first) = self.allocate(vtx_count, idx_count, v);

            let half_thickness = if ANTIALIAS {
                thickness - pixel_size
            } else {
                thickness
            } * 0.5;
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
                    let inv = pixel_size / d_len;
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
                    let inv = pixel_size / d_len;
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
                let ((mlx, mly, mrx, mry), (mlax, mlay, mrax, mray)) = if miter_l_recip.abs() > 1e-3
                {
                    let mut miter_l = half_thickness / miter_l_recip;
                    // Limit (inner) miter so it doesn't shoot away when miter is longer than adjacent line segments on acute angles
                    if bevel {
                        // This is too aggressive (not exactly precise)
                        let min_sqlen = if sqlen1 > sqlen2 { sqlen2 } else { sqlen1 };
                        let miter_sqlen = ((dx1 + dx2) * (dx1 + dx2) + (dy1 + dy2) * (dy1 + dy2))
                            * miter_l
                            * miter_l;
                        if miter_sqlen > min_sqlen {
                            miter_l *= (min_sqlen / miter_sqlen).sqrt();
                        }
                    }
                    (
                        (
                            p1.x - (dx1 + dx2) * miter_l,
                            p1.y - (dy1 + dy2) * miter_l,
                            p1.x + (dx1 + dx2) * miter_l,
                            p1.y + (dy1 + dy2) * miter_l,
                        ),
                        if ANTIALIAS {
                            let miter_al = half_thickness_aa / miter_l_recip;
                            (
                                p1.x - (dx1 + dx2) * miter_al,
                                p1.y - (dy1 + dy2) * miter_al,
                                p1.x + (dx1 + dx2) * miter_al,
                                p1.y + (dy1 + dy2) * miter_al,
                            )
                        } else {
                            (0.0, 0.0, 0.0, 0.0)
                        },
                    )
                } else {
                    // Avoid degeneracy for (nearly) straight lines
                    (
                        (
                            p1.x + dy1 * half_thickness,
                            p1.y - dx1 * half_thickness,
                            p1.x - dy1 * half_thickness,
                            p1.y + dx1 * half_thickness,
                        ),
                        if ANTIALIAS {
                            (
                                p1.x + dy1 * half_thickness_aa,
                                p1.y - dx1 * half_thickness_aa,
                                p1.x - dy1 * half_thickness_aa,
                                p1.y + dx1 * half_thickness_aa,
                            )
                        } else {
                            (0.0, 0.0, 0.0, 0.0)
                        },
                    )
                };
                // The two bevel vertices if the angle is right or obtuse
                // miter_sign == 1, iff the outer (maybe bevelled) edge is on the right, -1 iff it is on the left
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
                let (b1ax, b1ay, b2ax, b2ay) = if bevel && ANTIALIAS {
                    (
                        p1.x + (dx1 - dy1 * miter_sign) * half_thickness_aa,
                        p1.y + (dy1 + dx1 * miter_sign) * half_thickness_aa,
                        p1.x + (dx2 + dy2 * miter_sign) * half_thickness_aa,
                        p1.y + (dy2 - dx2 * miter_sign) * half_thickness_aa,
                    )
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                // Set the previous line direction so it doesn't need to be recomputed
                dx1 = -dx2;
                dy1 = -dy2;
                sqlen1 = sqlen2;

                // Now that we have all the point coordinates, put them into buffers

                // Vertices for each point are ordered in vertex buffer like this (looking in the direction of the polyline):
                // - left vertex*
                // - right vertex*
                // - left vertex AA fringe*  (if antialias)
                // - right vertex AA fringe* (if antialias)
                // - the remaining vertex (if bevel)
                // - the remaining vertex AA fringe (if bevel and antialias)
                // (*) if there is bevel, these vertices are the ones on the incoming edge.
                // Having all the vertices of the incoming edge in predictable positions is important - we reference them
                // even if we don't know relevant line properties yet

                let vertex_count = if ANTIALIAS {
                    if bevel {
                        6
                    } else {
                        4
                    }
                } else {
                    if bevel {
                        3
                    } else {
                        2
                    }
                };
                let bi = if ANTIALIAS { 4 } else { 2 }; // Outgoing edge bevel vertex index
                let bevel_l = bevel && miter_sign < 0.0;
                let bevel_r = bevel && miter_sign > 0.0;

                vs[0].set_pos([
                    if bevel_l { b1x } else { mlx },
                    if bevel_l { b1y } else { mly },
                ]);
                vs[1].set_pos([
                    if bevel_r { b1x } else { mrx },
                    if bevel_r { b1y } else { mry },
                ]);

                if bevel {
                    vs[bi].set_pos([b2x, b2y]);
                }

                if ANTIALIAS {
                    vs[2].set_pos([
                        if bevel_l { b1ax } else { mlax },
                        if bevel_l { b1ay } else { mlay },
                    ]);
                    vs[2].set_color(col_trans);
                    vs[3].set_pos([
                        if bevel_r { b1ax } else { mrax },
                        if bevel_r { b1ay } else { mray },
                    ]);
                    vs[3].set_color(col_trans);
                    if bevel {
                        vs[5].set_pos([b2ax, b2ay]);
                        vs[5].set_color(col_trans);
                    }
                }
                unused_vertices += max_n_vtx - vertex_count;

                vs = &mut vs[vertex_count..];

                if i1 < count {
                    let vtx_next_id = if i1 < points_count - 1 {
                        vi + vertex_count
                    } else {
                        first_vtx_ptr as usize
                    };
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

                    if ANTIALIAS {
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
                }
                vi += vertex_count;
            }
            self.reclaim_allocation(unused_vertices, unused_indices);
        }
    }

    pub fn add_polyline_variable<const ANTIALIAS: bool>(
        &mut self,
        points: &[Vec2],
        radius: &[f32],
        def: Vertex,
    ) {
        if points.len() < 2 {
            return;
        }
        assert!(points.len() == radius.len());
        let count = points.len() - 1;

        let gradient_size = self.pixel_size;
        let half_gradient = gradient_size * 0.5;
        let alpha_transparent = 0;
        let alpha_opaque = def.alpha();
        let index_count = count * 18;
        let vertex_count = points.len() * 4;
        let (vs, is, first) = self.allocate(vertex_count, index_count, def);
        let mut temp_normals = Vec::new();
        let mut temp_points = Vec::new();
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
            vs[i * 4 + 0].set_pos(temp_points[i * 4 + 0].into());
            vs[i * 4 + 0].set_alpha(alpha_transparent);

            vs[i * 4 + 1].set_pos(temp_points[i * 4 + 1].into());
            vs[i * 4 + 1].set_alpha(alpha_opaque);

            vs[i * 4 + 2].set_pos(temp_points[i * 4 + 2].into());
            vs[i * 4 + 2].set_alpha(alpha_opaque);

            vs[i * 4 + 3].set_pos(temp_points[i * 4 + 3].into());
            vs[i * 4 + 3].set_alpha(alpha_transparent);
        }
    }

    pub fn add_capsule_chain<const ANTIALIAS: bool>(
        &mut self,
        points: &[Vec2],
        radius: &[f32],
        def: Vertex,
    ) {
        // TODO: optimal non-overlapping implementation
        self.add_polyline_variable::<ANTIALIAS>(points, radius, def.clone());
        for (&point, &r) in points.iter().zip(radius.iter()) {
            self.add_circle::<ANTIALIAS>(point, r, r.ceil() as usize * 3, def.clone());
        }
    }
}

impl<Vertex: Copy + VertexPos2> GeometryBatch<Vertex> {
    pub fn add_position_indices(
        &mut self,
        positions: &[[f32; 2]],
        indices: &[IndexType],
        def: Vertex,
    ) {
        let (vs, is, first) = self.allocate(positions.len(), indices.len(), def);
        for (dest, pos) in vs.iter_mut().zip(positions) {
            dest.set_pos(*pos);
        }
        for (dest, index) in is.iter_mut().zip(indices) {
            *dest = index + first;
        }
    }
    pub fn add_rect(&mut self, start: Vec2, end: Vec2, def: Vertex) {
        let (vs, is, first) = self.allocate(4, 6, def);
        vs[0].set_pos([start.x, start.y]);
        vs[1].set_pos([end.x, start.y]);
        vs[2].set_pos([end.x, end.y]);
        vs[3].set_pos([start.x, end.y]);

        is[0] = first + 0;
        is[1] = first + 1;
        is[2] = first + 2;
        is[3] = first + 0;
        is[4] = first + 2;
        is[5] = first + 3;
    }

    pub fn add_rect_outline(&mut self, start: Vec2, end: Vec2, thickness: f32, def: Vertex) {
        let (vs, indices, first) = self.allocate(8, 24, def);

        let ht = thickness * 0.5;
        let ht = vec2(ht, ht);
        let os = start - ht;
        let oe = end + ht;
        let is = start + ht;
        let ie = end - ht;

        vs[0].set_pos([os.x, os.y]);
        vs[1].set_pos([oe.x, os.y]);
        vs[2].set_pos([oe.x, oe.y]);
        vs[3].set_pos([os.x, oe.y]);
        vs[4].set_pos([is.x, is.y]);
        vs[5].set_pos([ie.x, is.y]);
        vs[6].set_pos([ie.x, ie.y]);
        vs[7].set_pos([is.x, ie.y]);

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
}

impl<Vertex: Copy + VertexPos3> GeometryBatch<Vertex> {
    pub fn add_box(&mut self, center: [f32; 3], size: [f32; 3], def: Vertex) {
        let (vs, is, first) = self.allocate(8, 36, def);
        for (v, i) in vs.iter_mut().zip(0..8) {
            v.set_pos3([
                center[0] + size[0] * ((i & 1) as f32 - 1.0f32),
                center[1] + size[1] * (((i & 2) >> 1) as f32 - 1.0f32),
                center[2] + size[2] * (((i & 4) >> 2) as f32 - 1.0f32),
            ]);
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
        def: Vertex,
    ) {
        let (vs, is, first) = self.allocate(positions.len(), indices.len(), def);
        for (dest, pos) in vs.iter_mut().zip(positions) {
            dest.set_pos3(*pos);
        }
        for (dest, index) in is.iter_mut().zip(indices) {
            *dest = index + first;
        }
    }
}

impl<Vertex: Copy + VertexPos2 + VertexUV> GeometryBatch<Vertex> {
    pub fn add_rect_uv(&mut self, rect: [f32; 4], uv: [f32; 4], def: Vertex) -> &mut [Vertex] {
        let (vs, is, first) = self.allocate(4, 6, def);

        vs[0].set_pos([rect[0], rect[1]]);
        vs[1].set_pos([rect[2], rect[1]]);
        vs[2].set_pos([rect[2], rect[3]]);
        vs[3].set_pos([rect[0], rect[3]]);

        vs[0].set_uv([uv[0], uv[1]]);
        vs[1].set_uv([uv[2], uv[1]]);
        vs[2].set_uv([uv[2], uv[3]]);
        vs[3].set_uv([uv[0], uv[3]]);

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
        def: Vertex,
        uv_rect: [f32; 4],
    ) {
        let num_points = (divisions[0] * divisions[1]) as usize;
        let (vs, is, first) = self.allocate(num_points, 6 * num_points, def);

        for j in 0..=divisions[1] {
            for i in 0..=divisions[0] {
                let v = &mut vs[(j * (divisions[0] + 1) + i) as usize];
                let x_f = i as f32 / divisions[0] as f32;
                let y_f = j as f32 / divisions[1] as f32;
                v.set_pos([
                    (x_f - 0.5f32) * extents[0] + center[0],
                    (y_f - 0.5f32) * extents[1] + center[1],
                ]);
                v.set_uv([
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

/// A trait to be implemented for your vertex format in to use colored or
/// alpha-blended primitives.
pub trait VertexColor {
    fn set_color(&mut self, color: [u8; 4]);
    fn set_alpha(&mut self, alpha: u8);
    fn alpha(&self) -> u8;
}

/// A trait that needs to be implemented for your vertex format in order
/// to use 2D-primitives.
pub trait VertexPos2 {
    fn set_pos(&mut self, pos: [f32; 2]);
}

/// A trait that needs to be implemented for your vertex format in order
/// to use 3D-primitives.
pub trait VertexPos3 {
    fn set_pos3(&mut self, pos: [f32; 3]);
}

/// A trait to be implemented for your vertex format in order
/// to use primitives with UV coordinates.
pub trait VertexUV {
    fn set_uv(&mut self, uv: [f32; 2]);
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
