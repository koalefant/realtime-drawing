use core::default::Default;
use glam::vec2;
use miniquad::{conf, BlendFactor, BlendState, BlendValue, BufferLayout, Context, Equation, EventHandler, Pipeline, PipelineParams, Shader, ShaderMeta, Texture, UniformBlockLayout, UniformDesc, UniformType, UserData, VertexAttribute, VertexFormat, PassAction};
use realtime_drawing::{MiniquadBatch, VertexPos3UvColor};
use std::f32::consts::PI;

#[path="../rabbit.rs"]
mod rabbit;
use rabbit::{Rabbit, RabbitMap, linearstep};

struct Example {
    start_time: f64,
    last_time: f32,
    batch: MiniquadBatch<VertexPos3UvColor>,
    pipeline: Pipeline,
    white_texture: Texture,
    window_size: [f32; 2],

    rabbit: Rabbit,
    rabbit_map: RabbitMap,
}

pub struct ShaderUniforms {
    pub screen_size: [f32; 2],
}


impl Example {
    pub fn new(context: &mut Context) -> Example {
        let batch = MiniquadBatch::new();

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
            start_time: miniquad::date::now(),
            last_time: 0.0,
            batch,
            pipeline,
            white_texture,
            window_size: [1280.0, 720.0],
            rabbit: Rabbit::new(),
            rabbit_map: RabbitMap::new(),
        }
    }

    fn create_pipeline(ctx: &mut Context) -> Pipeline {
        let vertex_shader = r#"#version 100
            attribute vec2 pos;
            attribute vec2 uv;
            attribute vec4 color;
            uniform vec2 ;
            uniform vec2 screen_size;
            varying lowp vec2 v_uv;
            varying lowp vec4 v_color;
            void main() {
                gl_Position = vec4((pos / screen_size * 2.0 - 1.0) * vec2(1.0, -1.0), 0, 1);
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
    fn update(&mut self, _context: &mut Context) {
        let time = (miniquad::date::now() - self.start_time) as f32;
        let dt = time - self.last_time;

        // update position and velocity of jumping rabbit
        let rabbit_map = &self.rabbit_map;
        self.rabbit.update(time, dt, &|p| rabbit_map.distance(p), &|p| rabbit_map.normal(p));

        self.last_time = time;
    }

    fn draw(&mut self, context: &mut Context) {
        let time = (miniquad::date::now() - self.start_time) as f32;
        context.begin_default_pass(PassAction::Clear {
            color: Some((0.2, 0.2, 0.2, 1.0)),
            depth: None,
            stencil: None
        });

        self.batch.begin_frame();
        self.batch.clear();
        self.batch.set_image(self.white_texture);

        let [w, h] = self.window_size;
        let h = 1280.0 * h / w;
        let w = 1280.0;
        let view_scale = self.window_size[0] / w;
        self.batch.geometry.pixel_size = 1.0 / view_scale;

        // pulsing circles
        for (index, &(radius, thickness)) in [
            (48.0, 0.25),
            (64.0, 0.5),
            (80.0, 1.0),
            (96.0, 4.0),
        ].iter().rev().enumerate() {
            let r = radius * (1.0 + 0.05 * ((time * 0.5 + 0.25 * index as f32).cos()));
            let center = vec2(w * 0.25, h * 0.3);
            let num_segments = ((64.0 * view_scale) as usize).max(32);
            // fill
            self.batch.geometry.add_circle_aa(
                center,
                r,
                num_segments,
                [0, 32, 0, 64],
            );

            // dark outline
            self.batch.geometry.add_circle_outline_aa(
                center,
                r,
                thickness + 2.0,
                num_segments,
                [0, 32, 0, 255]
            );

            // circle outline with circular gradient
            self.batch.geometry.add_circle_outline_aa_with(
                center,
                r,
                thickness,
                num_segments,
                |pos, alpha, u| {
                    let t = u + (time * 0.1).fract();
                    VertexPos3UvColor {
                        pos: [pos.x, pos.y, 0.0],
                        // circular gradient calculation
                        color: [64, (160.0 + 32.0 * (t * PI * 6.0).cos()) as u8, 64, (255.0 * alpha) as u8],
                        uv: [0.0, 0.0],
                    }
                }
            );
        }

        // clock-like lines
        for (index, &thickness) in [0.5, 1.0, 2.0, 4.0].iter().enumerate() {
            let mut points = [vec2(w * 0.75, h * 0.3); 4];
            for i in 1..points.len() {
                let t = (time + index as f32 * 0.5 * PI) * 1.41_f32.powf(i as f32);
                let x = t.cos() as f32;
                let y = t.sin() as f32;
                points[i] = points[i - 1] + vec2(x, y) * 64.0 / (i as f32);
            }
            self.batch.geometry.add_polyline_miter_aa(&points, [0, 0, 64, 255], false, thickness as f32 + 2.0);
            self.batch.geometry.add_polyline_miter_aa(&points, [64, 128, 255, 255], false, thickness as f32);
        }

        // jumping rabbit
        let camera_offset = vec2(w * 0.5, h * 0.7);
        let rabbit_map = &self.rabbit_map;
        rabbit_map.draw(&mut self.batch.geometry, camera_offset);
        self.rabbit.draw(
            &mut self.batch.geometry,
            camera_offset,
            linearstep(0.45, 0.35, (1.0 - (time / 12.0).fract() * 2.0).abs()),
            &|p| rabbit_map.distance(p),
            |p| rabbit_map.normal(p)
            );

        context.apply_pipeline(&self.pipeline);
        context.apply_uniforms(&ShaderUniforms {
            screen_size: [w, h],
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
    miniquad::start(conf::Conf{
        sample_count: 1,
        window_width: 1280,
        window_height: 720,
        ..Default::default()
    }, |mut context| {
        UserData::owning(Example::new(&mut context), context)
    });
}
