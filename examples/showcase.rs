use core::default::Default;
use glam::vec2;
use miniquad::{
    conf, date, Bindings, BlendFactor, BlendState, BlendValue, Buffer, BufferLayout, BufferType,
    Context, Equation, EventHandler, Pipeline, PipelineParams, Shader, ShaderMeta, Texture,
    UniformBlockLayout, UniformDesc, UniformType, UserData, VertexAttribute, VertexFormat,
};
use realtime_drawing::{MiniquadBatch, VertexPos3UvColor};

struct Example {
    batch: MiniquadBatch<VertexPos3UvColor>,
    pipeline: Pipeline,
    white_texture: Texture,
    window_size: [f32; 2],
}

pub struct ShaderUniforms {
    pub offset: [f32; 2],
    pub screen_size: [f32; 2],
}

impl Example {
    pub fn new(context: &mut Context) -> Example {
        let batch = MiniquadBatch::new(4096, 4096);

        let white_texture = Texture::from_rgba8(
            context,
            4,
            4,
            &[
                // white RGBA-image 4x4 pixels
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            ],
        );
        let pipeline = Example::create_pipeline(context);

        Example {
            batch,
            pipeline,
            white_texture,
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
                gl_Position = vec4((pos + offset) / screen_size * 2.0 - 1.0, 0, 1);
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

        let pipeline = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float3),
                VertexAttribute::new("uv", VertexFormat::Float2),
                VertexAttribute::new("color", VertexFormat::Byte4),
            ],
            shader,
            PipelineParams {
                alpha_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );
        pipeline
    }
}

impl EventHandler for Example {
    fn update(&mut self, _context: &mut Context) {}

    fn draw(&mut self, context: &mut Context) {
        context.begin_default_pass(Default::default());

        self.batch.clear();
        self.batch.set_image(self.white_texture);

        self.batch.geometry.add_circle_outline::<true>(
            vec2(self.window_size[0] * 0.5, self.window_size[1] * 0.5),
            64.0,
            1.0,
            64,
            [255, 0, 0, 255]
        );

        context.apply_pipeline(&self.pipeline);
        context.apply_uniforms(&ShaderUniforms {
            offset: [0.0, 0.0],
            screen_size: self.window_size,
        });
        self.batch.draw(context);

        context.end_render_pass();

        context.commit_frame();
    }

    fn resize_event(&mut self, _context: &mut Context, width: f32, height: f32) {
        self.window_size = [width, height];
    }
}

fn main() {
    miniquad::start(conf::Conf::default(), |mut context| {
        UserData::owning(Example::new(&mut context), context)
    });
}
