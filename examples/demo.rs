use core::default::Default;
use glam::{vec2, Vec2};
use miniquad::{
    conf, BlendFactor, BlendState, BlendValue, BufferLayout, Context, Equation, EventHandler,
    PassAction, Pipeline, PipelineParams, Shader, ShaderMeta, Texture, UniformBlockLayout,
    UniformDesc, UniformType, UserData, VertexAttribute, VertexFormat,
};
use realtime_drawing::{MiniquadBatch, VertexPos3UvColor};

#[path = "../rabbit.rs"]
mod rabbit;
use rabbit::{linearstep, Rabbit, RabbitMap};

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

impl EventHandler for Example {
    fn draw(&mut self, context: &mut Context) {
        let time = (miniquad::date::now() - self.start_time) as f32;
        context.begin_default_pass(PassAction::Clear {
            color: Some((0.2, 0.2, 0.2, 1.0)),
            depth: None,
            stencil: None,
        });

        self.batch.begin_frame();
        self.batch.clear();
        self.batch.set_image(self.white_texture);

        let [w, h] = self.window_size;
        let h = 1280.0 * h / w;
        let w = 1280.0;
        let view_scale = self.window_size[0] / w;
        self.batch.geometry.pixel_size = 1.0 / view_scale;
       
        // circles
        {
            let center = vec2(w * 0.15, h * 0.5).floor();
            let center_aa = vec2(w * 0.15, h * 0.2).floor();
            let num_segments = ((64.0 * view_scale) as usize).max(32);

            // fill
            self.batch
                .geometry
                .fill_circle(center, 32.0, num_segments, [255, 255, 255, 255]);

            self.batch
                .geometry
                .fill_circle_aa(center_aa, 32.0, num_segments, [255, 255, 255, 255]);

            // multiple outlines
            for &(r, thickness) in [(48.0, 2.0), (64.0, 1.0), (80.0, 0.5), (96.0, 0.25)].iter().rev() {
                self.batch.geometry.stroke_circle(
                    center,
                    r,
                    thickness,
                    num_segments,
                    [255, 255, 255, 255]
                );

                self.batch.geometry.stroke_circle_aa(
                    center_aa,
                    r,
                    thickness,
                    num_segments,
                    [255, 255, 255, 255]
                );
            }

        }

        // lines
        {
            let thickness_list = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0];
            for (i, &thickness) in thickness_list.iter().enumerate() {
                let offset = vec2(w * 0.3, h * 0.2 + (i as f32 - 5.0) * 15.0).floor();
                self.batch.geometry.stroke_line_aa(
                    offset + vec2(-50.0, 10.0),
                    offset + vec2(50.0, -10.0),
                    thickness,
                    [255, 255, 255, 255]);

                let offset = vec2(w * 0.3, h * 0.5 + (i as f32 - 5.0) * 15.0).floor();
                self.batch.geometry.stroke_line(
                    offset + vec2(-50.0, 10.0),
                    offset + vec2(50.0, -10.0),
                    thickness,
                    [255, 255, 255, 255]);
            }
        }

        // round rectangles
        {
            // with antialiasing
            let mut offset = vec2(w * 0.45, h * 0.1).round();
            self.batch.geometry.stroke_round_rect_aa(offset + vec2(-50.5, -25.5), offset + vec2(50.5, 25.5), 4.0, 6, 1.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect_aa(offset + vec2(-50.5, -25.5), offset + vec2(50.5, 25.5), 8.0, 6, 1.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect_aa(offset + vec2(-50.5, -25.5), offset + vec2(50.5, 25.5), 16.0, 6, 1.0, [255, 255, 255, 255]);

            let mut offset = vec2(w * 0.45, h * 0.2).round();
            self.batch.geometry.stroke_round_rect_aa(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 4.0, 4, 4.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect_aa(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 8.0, 8, 4.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect_aa(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 16.0, 16, 4.0, [255, 255, 255, 255]);

            let mut offset = vec2(w * 0.45, h * 0.3).round();
            self.batch.geometry.fill_round_rect_aa(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 4.0, 4, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.fill_round_rect_aa(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 8.0, 8, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.fill_round_rect_aa(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 16.0, 16, [255, 255, 255, 255]);

            // no antialiasing

            let mut offset = vec2(w * 0.45, h * 0.4).round();
            self.batch.geometry.stroke_round_rect(offset + vec2(-50.5, -25.5), offset + vec2(50.5, 25.5), 4.0, 6, 1.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect(offset + vec2(-50.5, -25.5), offset + vec2(50.5, 25.5), 8.0, 6, 1.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect(offset + vec2(-50.5, -25.5), offset + vec2(50.5, 25.5), 16.0, 6, 1.0, [255, 255, 255, 255]);

            let mut offset = vec2(w * 0.45, h * 0.5).round();
            self.batch.geometry.stroke_round_rect(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 4.0, 6, 4.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 8.0, 6, 4.0, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.stroke_round_rect(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 16.0, 6, 4.0, [255, 255, 255, 255]);

            let mut offset = vec2(w * 0.45, h * 0.6).round();
            self.batch.geometry.fill_round_rect(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 4.0, 4, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.fill_round_rect(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 8.0, 8, [255, 255, 255, 255]);
            offset.x += 110.0;
            self.batch.geometry.fill_round_rect(offset + vec2(-50.0, -25.0), offset + vec2(50.0, 25.0), 16.0, 16, [255, 255, 255, 255]);
        }

        // polylines
        {
            let points = [
                vec2(0.0, 0.0), vec2(48.0, 0.0), vec2(48.0, 4.0), vec2(24.176, 9.916),
                vec2(20.0, 3.029), vec2(15.824, 9.916), vec2(8.0, 8.0), vec2(9.916, 15.824),
                vec2(3.029, 20.0), vec2(9.916, 24.176), vec2(8.0, 32.0), vec2(12.0, 40.0),
                vec2(36.0, 40.0), vec2(20.0, 24.0), vec2(24.0, 20.0), vec2(48.0, 44.0),
                vec2(48.0, 48.0), vec2(8.0, 48.0), vec2(0.0, 40.0), 
            ];
            let thickness_list = [1.0, 2.0, 4.0];
            for (i, &thickness) in thickness_list.iter().enumerate() {
                let offset = vec2(w * 0.75 + (i as f32 - 0.5) * 112.0, h * 0.2 - 48.0).floor() + Vec2::splat((thickness * 0.5f32).fract());
                self.batch.geometry.stroke_polyline_aa(
                    &points
                    .iter()
                    .map(|p| *p * 2.0 + offset)
                    .collect::<Vec<_>>(),
                    true,
                    thickness,
                    [255, 255, 255, 255]);

                let offset = vec2(w * 0.75 + (i as f32 - 0.5) * 112.0, h * 0.5 - 48.0).floor() + Vec2::splat((thickness * 0.5f32).fract());
                self.batch.geometry.stroke_polyline(
                    &points
                    .iter()
                    .map(|p| *p * 2.0 + offset)
                    .collect::<Vec<_>>(),
                    true,
                    thickness,
                    [255, 255, 255, 255]);
            }
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
            |p| rabbit_map.normal(p),
        );

        context.apply_pipeline(&self.pipeline);
        context.apply_uniforms(&ShaderUniforms {
            screen_size: [w, h],
        });
        self.batch.draw(context);

        context.end_render_pass();

        context.commit_frame();
    }

    fn update(&mut self, _context: &mut Context) {
        let time = (miniquad::date::now() - self.start_time) as f32;
        let dt = time - self.last_time;

        // update position and velocity of jumping rabbit
        let rabbit_map = &self.rabbit_map;
        self.rabbit
            .update(time, dt, &|p| rabbit_map.distance(p), &|p| {
                rabbit_map.normal(p)
            });

        self.last_time = time;
    }


    fn resize_event(&mut self, _context: &mut Context, width: f32, height: f32) {
        self.window_size = [width, height];
    }
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
                    uniforms: vec![UniformDesc::new("screen_size", UniformType::Float2)],
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

pub struct ShaderUniforms {
    pub screen_size: [f32; 2],
}


fn main() {
    miniquad::start(
        conf::Conf {
            sample_count: 0,
            window_width: 1280,
            window_height: 720,
            ..Default::default()
        },
        |mut context| UserData::owning(Example::new(&mut context), context),
    );
}
