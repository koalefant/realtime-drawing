use crate::{VertexColor, VertexPos2, VertexPos3, VertexUV};
use core::default::Default;

#[repr(C)]
#[derive(Copy, Clone)]
/// Example of struct defining traits for custom vertex format.
///
/// See `Vertex*` trait implementations below.
pub struct VertexPos3UvColor {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

impl VertexPos3UvColor {
    pub fn of_color(color: [u8; 4]) -> Self {
        Self {
            pos: [0f32, 0f32, 0f32],
            uv: [0f32, 0f32],
            color,
        }
    }
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for VertexPos3UvColor {
    fn default() -> Self {
        Self {
            pos: [0f32, 0f32, 0f32],
            uv: [0f32, 0f32],
            color: [0, 0, 0, 0],
        }
    }
}
impl VertexPos2 for VertexPos3UvColor {
    fn set_pos(&mut self, pos: [f32; 2]) {
        self.pos = [pos[0], pos[1], 0.0];
    }
}
impl VertexPos3 for VertexPos3UvColor {
    fn set_pos3(&mut self, pos: [f32; 3]) {
        self.pos = pos;
    }
}
impl VertexUV for VertexPos3UvColor {
    fn set_uv(&mut self, uv: [f32; 2]) {
        self.uv = uv;
    }
}
impl VertexColor for VertexPos3UvColor {
    fn set_color(&mut self, color: [u8; 4]) {
        self.color = color;
    }
    fn set_alpha(&mut self, alpha: u8) {
        self.color[3] = alpha;
    }
    fn alpha(&self) -> u8 {
        self.color[3]
    }
}
