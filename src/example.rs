use crate::{FromPos2Color, FromPos3Color, FromPos2ColorUV};
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
impl FromPos2Color for VertexPos3UvColor {
    fn from_pos2_color(pos: [f32; 2], color: [u8; 4])->Self {
        Self{
            pos: [pos[0], pos[1], 0.0],
            color,
            uv: [0.0, 0.0]
        }
    }
}

impl FromPos3Color for VertexPos3UvColor {
    fn from_pos3_color(pos: [f32; 3], color: [u8; 4])->Self {
        Self{
            pos,
            color,
            uv: [0.0, 0.0]
        }
    }
}

impl FromPos2ColorUV for VertexPos3UvColor {
    fn from_pos2_color_uv(pos: [f32; 2], color: [u8; 4], uv: [f32; 2])->Self {
        Self{
            pos: [pos[0], pos[1], 0.0],
            color,
            uv
        }
    }
}

