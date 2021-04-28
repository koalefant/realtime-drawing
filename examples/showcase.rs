use realtime_drawing::*;
use realtime_drawing::example::*;
use glam::vec2;

fn main() {
    let mut geometry = GeometryBuilder::<VertexPos3UvColor>::new(1024, 1024);
    geometry.add_circle::<true>(vec2(0.0, 0.0), 10.0, 64, Default::default());
}
