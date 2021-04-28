use core::default::Default;
use glam::vec2;
use miniquad::{
    conf, date, Bindings, Buffer, BufferLayout, BufferType, Context, EventHandler, Pipeline,
    Shader, ShaderMeta, Texture, UniformBlockLayout, UniformDesc, UniformType, UserData,
    VertexAttribute, VertexFormat,
};
use realtime_drawing::example::*;
use realtime_drawing::*;

struct Stage {
    geometry: GeometryBuilder<VertexPos3UvColor>,
    pipeline: Pipeline,
    bindings: Bindings,
    window_size: [f32; 2],
}

pub struct ShaderUniforms {
    pub offset: [f32; 2],
    pub screen_size: [f32; 2],
}

impl Stage {
    pub fn new(ctx: &mut Context) -> Stage {
        #[rustfmt::skip]
        let vertices = [
            VertexPos3UvColor { pos: [-5.0, -5.0, 0.0], uv: [0.0, 0.0], color: [255, 255, 255, 255] },
            VertexPos3UvColor { pos: [ 5.0, -5.0, 0.0], uv: [1.0, 0.0], color: [255, 255, 255, 255] },
            VertexPos3UvColor { pos: [ 5.0,  5.0, 0.0], uv: [1.0, 1.0], color: [255, 255, 255, 255] },
            VertexPos3UvColor { pos: [-5.0,  5.0, 0.0], uv: [0.0, 1.0], color: [255, 255, 255, 255] },
        ];
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices: [i16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let pixels: [u8; 4 * 4 * 4] = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00,
            0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        let texture = Texture::from_rgba8(ctx, 4, 4, &pixels);

        let pipeline = Stage::create_pipeline(ctx);
        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer: index_buffer,
            images: vec![texture],
        };

        let geometry = GeometryBuilder::new(1024, 1024);

        Stage {
            geometry,
            pipeline,
            bindings,
            window_size: [800.0, 600.0],
        }
    }

    fn create_pipeline(ctx: &mut Context) -> Pipeline {
        let vertex_shader = r#"#version 100
            attribute vec2 pos;
            attribute vec2 uv;
            attribute vec4 color;
            uniform vec2 offset;
            uniform vec2 screen_size;
            varying lowp vec2 v_uv;
            varying lowp vec4 v_color;
            void main() {
                gl_Position = vec4((pos + offset) / screen_size, 0, 1);
                v_uv = uv;
                v_color = color / 255.0;
            }"#;
        let fragment_shader = r#"#version 100
            varying lowp vec2 v_uv;
            varying lowp vec4 v_color;
            uniform sampler2D tex;
            void main() {
                gl_FragColor = v_color * texture2D(tex, v_uv);
            }"#;
        let shader = Shader::new(
            ctx,
            vertex_shader,
            fragment_shader,
            ShaderMeta {
                images: vec!["tex".to_owned()],
                uniforms: UniformBlockLayout {
                    // describes struct ShaderUniforms
                    uniforms: vec![
                        UniformDesc::new("offset", UniformType::Float2),
                        UniformDesc::new("screen_size", UniformType::Float2),
                    ],
                },
            },
        )
        .unwrap();

        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float3),
                VertexAttribute::new("uv", VertexFormat::Float2),
                VertexAttribute::new("color", VertexFormat::Byte4),
            ],
            shader,
        );
        pipeline
    }
}

impl EventHandler for Stage {
    fn update(&mut self, _ctx: &mut Context) {}

    fn resize_event(&mut self, _ctx: &mut Context, width: f32, height: f32) {
        self.window_size = [width, height];
    }

    fn draw(&mut self, ctx: &mut Context) {
        let t = date::now();

        ctx.begin_default_pass(Default::default());

        self.geometry.add_circle_outline::<true>(
            vec2(self.window_size[0] * 0.5, self.window_size[1] * 0.5),
            64.0,
            2.0,
            64,
            VertexPos3UvColor {
                pos: [0.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                color: [255, 0, 0, 255],
            },
        );

        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        for i in 0..10 {
            let t = t + i as f64 * 0.3;

            ctx.apply_uniforms(&ShaderUniforms {
                offset: [t.sin() as f32 * 64.0, (t * 3.).cos() as f32 * 64.0],
                screen_size: self.window_size,
            });
            ctx.draw(0, 6, 1);
        }
        ctx.end_render_pass();

        ctx.commit_frame();
    }
}

fn main() {
    miniquad::start(conf::Conf::default(), |mut ctx| {
        UserData::owning(Stage::new(&mut ctx), ctx)
    });
}
