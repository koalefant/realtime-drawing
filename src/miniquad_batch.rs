#[cfg(feature = "miniquad")]
use miniquad;
type IndexType = u16;
use crate::{GeometryBatch, GeometryCommand};
use core::marker::Copy;
use miniquad::{Buffer, BufferType, Context};
use std::mem::size_of;

const NUM_FRAMES: usize = 2;
const DEFAULT_BATCH_CAPACITY: usize = 65535;

#[cfg(feature = "miniquad")]
struct Draw {
    image: miniquad::Texture,
    first_command: usize,
    clip: Option<[i32; 4]>,
}

/// [`miniquad`]-backend for [`GeometryBatch`]
///
/// # Functionality
/// - Vertex/index buffer creation and update for GeometryBatch through Miniquad backend. See [`MiniquadBatch::draw`]
/// - Tracking of a texture binding (see [`MiniquadBatch::set_image`])
/// - Cycling of pooled vertex/index buffers to avoid writing into buffer that is being rendered. See [`MiniquadBatch::begin_frame`]
///
/// # Example
///
/// ```
/// impl Example {
///     fn new()->Self{
///         Self {
///             texture: miniquad::Texture::new(/* ... */),
///             pipeline: miniquad::Pipeline::new(/* ... */),
///             batch: MiniquadBatch::new(),
///         }
///     }
/// }
/// impl miniquad::EventHandler for Example {
///     fn draw(&mut self, context: &mut miniquad::Context) {
///         context.begin_default_pass(Default::default());
///         self.batch.begin_frame();
///         self.batch.clear();
///         self.batch.set_image(self.texture);
///         self.batch.geometry
///             .add_circle_outline_aa(vec2(256.0, 256.0), 128.0, 1.0, 64, [255, 0, 0, 255]);
///
///         context.apply_pipeline(&self.pipeline);
///         self.batch.draw(context);
///
///         context.end_render_pass();
///         context.commit_frame();
///     }
/// }
/// ```
///
/// [`miniquad`]: https://docs.rs/miniquad/
#[cfg(feature = "miniquad")]
pub struct MiniquadBatch<Vertex: Copy> {
    draws: Vec<Draw>,
    pub geometry: GeometryBatch<Vertex>,
    pub images: Vec<miniquad::Texture>,

    vertex_pool: [Vec<miniquad::Buffer>; NUM_FRAMES],
    index_pool: [Vec<miniquad::Buffer>; NUM_FRAMES],
    empty_vb: Option<miniquad::Buffer>,
    empty_ib: Option<miniquad::Buffer>,
    frame: usize,
    temp_bindings: Option<miniquad::Bindings>,
}

#[cfg(feature = "miniquad")]
impl<Vertex: Copy> MiniquadBatch<Vertex> {
    /// Creates new MiniquadBatch, equivalent of `MiniquadBatch::with_capacity(65535, 65535)`.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BATCH_CAPACITY, DEFAULT_BATCH_CAPACITY)
    }
    /// Creates new MiniquadBatch with given capacity.
    ///
    /// Capacity limits number of vertices/indices that can be used in a single primitive.
    ///
    /// When capacity of individual vertex/index buffer is exhausted another buffer is created with the same capacity.
    pub fn with_capacity(buffer_vertices: usize, buffer_indices: usize) -> Self {
        Self {
            draws: Vec::new(),
            geometry: GeometryBatch::with_capacity(buffer_vertices, buffer_indices),
            temp_bindings: None,
            vertex_pool: [Vec::new(), Vec::new()],
            index_pool: [Vec::new(), Vec::new()],
            images: Vec::new(),
            empty_vb: None,
            empty_ib: None,
            frame: 0,
        }
    }

    /// Marks beginning of the frame by performing rotation of buffers.
    ///
    /// This is done to prevent writing into buffers that are being used for rendering current or past frame.
    pub fn begin_frame(&mut self) {
        self.frame = (self.frame + 1) % NUM_FRAMES;
        // move unused buffers to the next frame
        let mut unused_vertices = std::mem::replace(&mut self.vertex_pool[self.frame], Vec::new());
        self.vertex_pool[(self.frame + 1) % NUM_FRAMES].extend(unused_vertices.drain(..));
        let mut unused_indices = std::mem::replace(&mut self.index_pool[self.frame], Vec::new());
        self.index_pool[(self.frame + 1) % NUM_FRAMES].extend(unused_indices.drain(..));
        self.vertex_pool[self.frame] = unused_vertices;
        self.index_pool[self.frame] = unused_indices;
    }

    /// Performs actual rendering.
    ///
    /// Does not clear geometry buffers, can be called multiple times to render exactly the same geometry.
    pub fn draw(&mut self, c: &mut Context) {
        self.geometry.finish_commands();

        let bindings = {
            let empty_ib = self.empty_ib.get_or_insert_with(|| {
                let empty_indices: &[i16] = &[];
                Buffer::immutable(c, BufferType::IndexBuffer, empty_indices)
            });
            let empty_vb = self.empty_vb.get_or_insert_with(|| {
                let empty_vertices: &[Vertex] = &[];
                Buffer::immutable(c, BufferType::VertexBuffer, empty_vertices)
            });

            let bindings = self
                .temp_bindings
                .get_or_insert_with(|| miniquad::Bindings {
                    vertex_buffers: vec![empty_vb.clone()],
                    index_buffer: empty_ib.clone(),
                    images: Vec::new(),
                });
            bindings.images = std::mem::take(&mut self.images);
            bindings
        };

        let mut index_pos = 0;
        let mut vertex_pos = 0;
        let mut draw_pos: usize = 0;

        let next_start_iter = self
            .draws
            .iter()
            .skip(1)
            .map(|x| x.first_command)
            .chain(std::iter::once(self.geometry.commands.len()));
        let screen_size = c.screen_size();
        let screen_h = screen_size.1 as i32;
        let default_clip = [0, 0, screen_size.0 as i32, screen_size.1 as i32];
        for (
            Draw {
                image,
                first_command,
                clip,
            },
            last_command,
        ) in self.draws.iter().zip(next_start_iter)
        {
            if bindings.images.is_empty() {
                bindings.images.push(*image);
            } else {
                bindings.images[0] = *image;
            }
            let clip = clip.unwrap_or(default_clip);
            c.apply_scissor_rect(
                clip[0],
                screen_h - clip[3],
                clip[2] - clip[0],
                clip[3] - clip[1],
            );
            let mut dirty_bindings = true;
            for command_index in *first_command..last_command {
                let command = &self.geometry.commands[command_index];
                match *command {
                    GeometryCommand::Indices { num_indices } => {
                        assert!(num_indices <= self.geometry.max_buffer_indices);
                        let capacity = self.geometry.max_buffer_indices * size_of::<IndexType>();
                        let buf = self.index_pool[(self.frame + 1) % NUM_FRAMES]
                            .pop()
                            .unwrap_or_else(|| {
                                miniquad::Buffer::stream(
                                    c,
                                    miniquad::BufferType::IndexBuffer,
                                    capacity,
                                )
                            });
                        assert!(capacity == buf.size());
                        let indices = &self.geometry.indices[index_pos..index_pos + num_indices];
                        buf.update(c, indices);
                        self.index_pool[self.frame].push(buf.clone());
                        index_pos += num_indices;
                        draw_pos = 0;
                        bindings.index_buffer = buf;
                        dirty_bindings = true;
                    }
                    GeometryCommand::Vertices { num_vertices } => {
                        assert!(num_vertices <= self.geometry.max_buffer_vertices);
                        let capacity = self.geometry.max_buffer_vertices * size_of::<Vertex>();
                        let buf = self.vertex_pool[(self.frame + 1) % NUM_FRAMES]
                            .pop()
                            .unwrap_or_else(|| {
                                miniquad::Buffer::stream(
                                    c,
                                    miniquad::BufferType::VertexBuffer,
                                    capacity,
                                )
                            });
                        assert!(capacity == buf.size());
                        buf.update(
                            c,
                            &self.geometry.vertices[vertex_pos..vertex_pos + num_vertices],
                        );
                        self.vertex_pool[self.frame].push(buf.clone());
                        vertex_pos += num_vertices;
                        bindings.vertex_buffers[0] = buf;
                        dirty_bindings = true;
                    }
                    GeometryCommand::DrawCall { num_indices } => {
                        if dirty_bindings {
                            c.apply_bindings(&bindings);
                            dirty_bindings = false;
                        }
                        if num_indices != 0 {
                            c.draw(draw_pos as _, num_indices as _, 1);
                            draw_pos += num_indices;
                        }
                    }
                }
            }
        }

        self.images = std::mem::take(&mut bindings.images);
    }

    pub fn clear(&mut self) {
        let last_image_clip = self.draws.last().map(|d| (d.image, d.clip));
        self.draws.clear();
        if let Some((last_image, last_clip)) = last_image_clip {
            self.draws.push(Draw {
                image: last_image,
                first_command: 0,
                clip: last_clip,
            });
        }
        self.geometry.clear();
    }

    /// Performs `draw()` followed by `clear()`
    pub fn flush(&mut self, c: &mut miniquad::Context) {
        self.draw(c);
        self.clear();
    }

    pub fn set_image(&mut self, image: miniquad::Texture) {
        self.geometry.finish_commands();
        let first_command = if self.draws.is_empty() {
            0
        } else {
            self.geometry.commands.len()
        };
        let last = self.draws.last();
        let clip = last.map(|l| l.clip).flatten();
        let last_image = last
            .map(|l| l.image)
            .unwrap_or_else(|| miniquad::Texture::empty());
        if last_image != image || self.draws.is_empty() {
            self.draws.push(Draw {
                image,
                first_command,
                clip,
            });
        }
    }

    pub fn set_clip(&mut self, clip: Option<[i32; 4]>) {
        self.geometry.finish_commands();
        let first_command = if self.draws.is_empty() {
            0
        } else {
            self.geometry.commands.len()
        };
        let last = self.draws.last();
        let last_clip = last.map(|d| d.clip).flatten();
        let image = last
            .map(|l| l.image)
            .unwrap_or_else(|| miniquad::Texture::empty());
        if last_clip != clip || self.draws.is_empty() {
            self.draws.push(Draw {
                image,
                first_command,
                clip,
            });
        }
    }
}

#[cfg(feature = "miniquad")]
impl<Vertex: Copy> Drop for MiniquadBatch<Vertex> {
    fn drop(&mut self) {
        for pool in &mut self.vertex_pool {
            for b in pool {
                b.delete();
            }
        }
        for pool in &mut self.index_pool {
            for b in pool {
                b.delete();
            }
        }
    }
}
