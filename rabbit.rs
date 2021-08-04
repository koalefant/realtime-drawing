//!
//! An example of simple animated character drawn using `realtime_drawing` crate.
//!
use glam::{vec2, Vec2};
use realtime_drawing::{GeometryBatch, VertexPos3UvColor};
use std::f32::consts::PI;

/// Jumping rabbit. An example for real-time drawing.
pub struct Rabbit {
    pub pos: Vec2,
    pub velocity: Vec2,
    pub last_contact: (Vec2, Vec2),
    pub next_contact: Vec2,
}

pub struct RabbitMap {
    boxes: [(Vec2, Vec2); 4],
}

const RABBIT_RADIUS: f32 = 12.0;
const RABBIT_JUMP_SPEED: f32 = 400.0;
const GRAVITY_ACCELERATION: f32 = 1200.0;
const MAX_VELOCITY: f32 = 4000.0;
const DEBUG_VELOCITY_DRAW_SCALE: f32 = 0.2;

impl Rabbit {
    pub fn new() -> Self {
        Self {
            pos: vec2(0.0, 0.0),
            velocity: vec2(400.0, 0.0),
            last_contact: (Vec2::ZERO, Vec2::ZERO),
            next_contact: Vec2::ZERO,
        }
    }

    pub fn update<D, N>(
        &mut self,
        current_time: f32,
        dt: f32,
        distance_func_dynamic: &D,
        normal_func: &N,
    ) where
        D: Fn(Vec2) -> f32,
        N: Fn(Vec2) -> Vec2,
    {
        let mut velocity = self.velocity + vec2(0.0, GRAVITY_ACCELERATION * dt);
        let velocity_len = velocity.length();
        if velocity_len > MAX_VELOCITY {
            velocity = velocity * MAX_VELOCITY / velocity_len;
        }
        let new_pos = self.pos + velocity * dt;
        let (mut new_pos, hit_obstacle) =
            march_circle(self.pos, new_pos, RABBIT_RADIUS, &distance_func_dynamic);
        if new_pos == self.pos && hit_obstacle {
            // if we are blocked - try to slide out
            new_pos = march_circle_sliding(
                self.pos,
                velocity / velocity_len,
                velocity_len * dt,
                RABBIT_RADIUS,
                2.0,
                distance_func_dynamic,
                normal_func,
            );
        }
        if hit_obstacle {
            let normal = normal_func(new_pos);
            let contact_pos = new_pos - normal * RABBIT_RADIUS;
            self.last_contact = (contact_pos, velocity);
            let normal = normal_func(new_pos);
            let is_obstacle_normal = |n: Vec2| n.y > 0.0;
            if is_obstacle_normal(normal) {
                // we hit something above us, stop and let it fall
                velocity = vec2(0.0, 0.0);
            } else {
                // change direction on high slopes
                let turnaround_slope_limit = 4.0;
                if normal.x.abs() > normal.y.abs() * turnaround_slope_limit {
                    // bounce horizontally
                    velocity = (velocity * vec2(-1.0, 1.0))
                        .try_normalize()
                        .unwrap_or(vec2(-1.0, 0.0))
                        * RABBIT_JUMP_SPEED;
                } else {
                    // are we going to land soon?
                    let mut land_velocity = None;
                    let mut escape_velocity = None;
                    let default_direction = (normal.perp() * 1.0f32.copysign(velocity.x)
                        + vec2(0.0, -1.0))
                    .try_normalize()
                    .unwrap_or(vec2(-1.0, 0.0));
                    for &dir in [default_direction].iter() {
                        for &v in [
                            RABBIT_JUMP_SPEED,
                            RABBIT_JUMP_SPEED * 3.0 / 4.0,
                            RABBIT_JUMP_SPEED / 2.0,
                            RABBIT_JUMP_SPEED / 3.0,
                        ]
                        .iter()
                        {
                            let test_velocity = dir * v;
                            let (land_point, hit) = march_parabola(
                                self.pos,
                                RABBIT_RADIUS,
                                test_velocity,
                                GRAVITY_ACCELERATION,
                                150.0,
                                1.0 / 60.0,
                                &distance_func_dynamic,
                            );
                            if hit && land_point != self.pos {
                                let normal = normal_func(land_point);
                                if !is_obstacle_normal(normal) {
                                    land_velocity = Some(test_velocity);
                                    break;
                                }
                            } else {
                                if escape_velocity.is_none() {
                                    escape_velocity = Some(test_velocity);
                                }
                            }
                        }
                    }

                    velocity = if let Some(land_velocity) = land_velocity {
                        land_velocity
                    } else {
                        // there is no place to land, but there is a window for a jump
                        // where we don't hit the ceiling
                        if let Some(escape_velocity) = escape_velocity {
                            escape_velocity
                        } else {
                            // we are stuck, try randomizing direction in hope that one of
                            // the bounces gets us out
                            let random_angle = (float_hash(vec2(current_time, 1.0)) - 0.5) * PI;
                            let random_direction = random_angle.cos() * default_direction
                                + random_angle.sin() * default_direction.perp();
                            random_direction
                        }
                    };
                }
            }
        }
        self.velocity = velocity;
        self.pos = new_pos;
        self.next_contact = march_parabola(
            new_pos,
            RABBIT_RADIUS,
            self.velocity,
            GRAVITY_ACCELERATION,
            200.0,
            1.0 / 60.0,
            &distance_func_dynamic,
        )
        .0;
    }

    pub fn draw<D, N>(
        &mut self,
        geometry: &mut GeometryBatch<VertexPos3UvColor>,
        offset: Vec2,
        debug_alpha: f32,
        distance_func: &D,
        normal_func: N,
    ) where
        D: Fn(Vec2) -> f32,
        N: Fn(Vec2) -> Vec2,
    {
        let push_against = |points: &[Vec2], r: f32| -> Vec2 {
            for pair in points.windows(2) {
                let (pos, hit) = march_circle(pair[0] - offset, pair[1] - offset, r, distance_func);
                if hit {
                    return pos + offset;
                }
            }
            return *points.last().unwrap();
        };

        let pos = self.pos + offset;

        let next_contact =
            (self.next_contact - normal_func(self.next_contact) * RABBIT_RADIUS) + offset;

        let (last_contact, _last_velocity) = self.last_contact;
        let last_contact = last_contact + offset;

        let contact_t = (pos - last_contact).length()
            / ((pos - last_contact).length() + (pos - next_contact).length());

        let front_contact = last_contact.lerp(next_contact, smootherstep(0.15, 0.4, contact_t));
        let back_contact = last_contact.lerp(next_contact, smootherstep(0.8, 0.9, contact_t));
        let normal = vec2(0.0, -1.0);
        let vel = self.velocity;

        let dir = vel.try_normalize().unwrap_or(vec2(0.0, -1.0));
        let color = [184, 184, 184, 255];
        let def = [150, 150, 150, 255];
        let ldef = color;
        let odef = [80, 80, 80, 255];
        let dir_sign = 1.0f32.copysign(dir.x);

        // body
        let compression = linearstep(100.0, 0.0, (front_contact - pos).length());
        let body_dir = (-dir).lerp(-normal.perp() * dir_sign, compression);
        let body_front =
            pos + vec2(0.0, 0.0).lerp(vec2(7.5 * dir_sign, 0.0), compression) + normal * 6.0;
        let body_back = push_against(
            &[
                body_front,
                body_front + body_dir * 30.0.lerp(15.0, compression),
            ],
            10.0,
        );
        let v = -body_dir.perp() * dir_sign;
        let body_s1 =
            body_back.lerp(body_front, 0.2) - v * (0.0 + 0.0 * smoothstep(0.0, 1.0, compression));
        let body_s2 =
            body_back.lerp(body_front, 0.8) + v * (2.0 + 0.0 * smoothstep(0.0, 1.0, compression));
        let back_r = 10.0;
        let front_r = 6.0;
        let mid_r1 = back_r.lerp(front_r, 0.2) * 0.8.lerp(1.0, compression);
        let mid_r2 = back_r.lerp(front_r, 0.8) * 0.9.lerp(1.0, compression);

        let white_back = body_back + v * back_r * 0.5;
        let white_s1 = body_s1 + v * mid_r1 * 0.5;
        let white_s2 = body_s2 + v * mid_r2 * 0.5;
        let white_front = body_front + v * front_r * 0.5;

        let tail_dir = (body_dir - v)
            .lerp(body_dir + v, smootherstep(0.0, 1.0, compression))
            .normalize();
        let tail_dir_mid = body_dir.lerp(tail_dir, 0.5).normalize();
        let tail_len = 8.0;
        let tail_base = body_back + body_dir * back_r;
        let tail_mid = tail_base + tail_dir_mid * tail_len * 0.5;
        let tail_tip = tail_mid + tail_dir * tail_len * 0.5;
        let tail_white_points = [
            white_back.into(),
            (tail_base + v * 3.0).into(),
            (tail_mid + v * 1.5).into(),
            (tail_tip + v - tail_dir).into(),
        ];
        let tail_points = [
            (body_back - v * 6.0).into(),
            (tail_base - v * 1.0).into(),
            (tail_mid - v * 1.0).into(),
            tail_tip.into(),
        ];

        // head
        let head_back = push_against(&[body_front, body_front + vec2(6.0 * dir_sign, -6.0)], 7.0);
        let head_front = head_back + vec2(6.5 * dir_sign, 7.5);

        let ear_base = head_back + vec2(-3.5 * dir_sign, 0.0);
        let ear_points_l = [
            ear_base.into(),
            (push_against(
                &[
                    ear_base + vec2(5.0 * dir_sign, -6.0),
                    ear_base + vec2(5.0 * dir_sign, -12.0),
                ],
                2.0,
            ))
            .into(),
            (push_against(
                &[
                    ear_base + vec2(12.0 * dir_sign, -9.0),
                    ear_base + vec2(12.0 * dir_sign, -18.0),
                ],
                2.0,
            ))
            .into(),
        ];
        let ear_points_r = [
            ear_base.into(),
            (push_against(
                &[
                    ear_base + vec2(3.0 * dir_sign, -6.5),
                    ear_base + vec2(3.0 * dir_sign, -13.0),
                ],
                2.0,
            ))
            .into(),
            (push_against(
                &[
                    ear_base + vec2(6.0 * dir_sign, -10.0),
                    ear_base + vec2(6.0 * dir_sign, -20.0),
                ],
                2.0,
            ))
            .into(),
        ];

        let vmask = [32, 32, 32, 255];

        let mut ellipse: [Vec2; 15] = Default::default();
        let r = 3.0;
        let c = head_back + vec2(6.5 * dir_sign, -2.0);
        for i in 0..ellipse.len() {
            let a = i as f32 * 2.0 * std::f32::consts::PI / (ellipse.len() as f32 + 1.0);
            ellipse[i] = vec2(c.x + a.cos() * r * 0.6, c.y + a.sin() * r);
        }

        let r = 3.5;
        let c = head_back + vec2(9.0 * dir_sign, 9.0);
        let rot_x = vec2((dir_sign * PI / 10.0).cos(), (dir_sign * PI / 10.0).sin());
        let rot_y = rot_x.perp();
        let mut ellipse2: [Vec2; 15] = Default::default();
        for i in 0..ellipse2.len() {
            let a = i as f32 * 2.0 * PI / (ellipse2.len() as f32 + 1.0);
            let p = vec2(a.cos(), a.sin()) * r;
            ellipse2[i] = (c + p.x * rot_x * 0.6 + p.y * rot_y).into();
        }

        let mut ellipse3: [Vec2; 15] = Default::default();
        for i in 0..ellipse3.len() {
            ellipse3[i][0] = (ellipse2[i][0] - c.x) * 0.5 + c.x;
            ellipse3[i][1] = (ellipse2[i][1] - c.y) * 0.5 + c.y;
        }

        // front leg
        let shoulder = body_front + vec2(-2.0 * dir_sign, 3.0);
        let shoulder_to_contact = front_contact - shoulder;
        let shoulder_to_contact_dir = shoulder_to_contact
            .try_normalize()
            .unwrap_or(vec2(dir_sign, 0.0));
        let front_paw =
            shoulder + shoulder_to_contact_dir * shoulder_to_contact.length().min(22.75);
        let front_paw_tip = push_against(
            &[
                front_paw + shoulder_to_contact_dir.perp() * -dir_sign * 5.0,
                front_paw + shoulder_to_contact_dir * 5.0,
            ],
            2.0,
        );
        let elbow = ankle_position(shoulder, front_paw, -dir_sign, 15.0, 8.0);

        // back leg
        let thigh = body_back - dir_sign * body_dir.perp() * 4.0;
        let thigh_to_contact = back_contact - thigh;
        let thigh_to_contact_dir = thigh_to_contact
            .try_normalize()
            .unwrap_or(vec2(dir_sign, 0.0));
        let back_paw_base = thigh + thigh_to_contact_dir * thigh_to_contact.length().min(30.0);
        let knee = ankle_position(thigh, back_paw_base, dir_sign, 12.0, 13.0);
        let back_paw_start = ankle_position(knee, back_paw_base, -dir_sign, 9.0, 14.0);
        let paw_tip_offset = (back_paw_base - back_paw_start) * 0.4;

        let back_paw_tip = push_against(
            &[
                back_paw_base + paw_tip_offset.perp() * -dir_sign,
                back_paw_base + paw_tip_offset,
            ],
            2.0,
        );

        // actual drawing
        // outline
        geometry.fill_circle_aa(body_back, back_r + 1.0, 24, odef);
        geometry.stroke_capsule_chain_aa(
            &[body_s1.into(), body_s2.into(), body_front.into()],
            &[mid_r1 + 1.0, mid_r2 + 1.0, front_r + 1.0],
            odef,
        );
        geometry.stroke_capsule_chain_aa(
            &tail_white_points,
            &[1.0, 2.0, 4.0, 3.0],
            [100, 100, 100, 255],
        );
        geometry.stroke_capsule_chain_aa(&tail_points, &[6.0, 4.0, 2.75, 2.0], odef);
        geometry.stroke_capsule_chain_aa(&ear_points_l, &[3.0, 4.0, 1.5], odef);
        geometry.stroke_capsule_chain_aa(&ear_points_r, &[3.0, 4.0, 1.5], odef);
        geometry.fill_circle_aa(shoulder, 5.5, 16, odef);
        geometry.stroke_capsule_chain_aa(
            &[
                shoulder.into(),
                elbow.into(),
                front_paw.into(),
                front_paw_tip.into(),
            ],
            &[5.5, 3.0, 2.5, 1.5],
            odef,
        );
        geometry.stroke_capsule_chain_aa(
            &[
                thigh.into(),
                knee.into(),
                knee.lerp(back_paw_start, 0.9).into(),
                back_paw_start.into(),
                back_paw_base.into(),
                back_paw_tip.into(),
            ],
            &[6.5, 4.5, 3.5, 3.5, 3.5, 2.5],
            odef,
        );
        geometry.stroke_capsule_chain_aa(&[head_back.into(), head_front.into()], &[8.0, 4.5], odef);

        // color
        geometry.fill_circle_aa(body_back, back_r, 24, def);
        geometry.stroke_capsule_chain_aa(
            &[body_s1.into(), body_s2.into(), body_front.into()],
            &[mid_r1, mid_r2, front_r],
            def,
        );
        geometry.stroke_capsule_chain_aa(
            &[
                white_back.into(),
                white_s1.into(),
                white_s2.into(),
                white_front.into(),
            ],
            &[back_r * 0.5, mid_r1 * 0.5, mid_r2 * 0.5, front_r * 0.5],
            ldef,
        );

        geometry.stroke_capsule_chain_aa(
            &tail_white_points,
            &[0.0, 1.0, 3.0, 2.0],
            [220, 220, 220, 255],
        );
        geometry.stroke_capsule_chain_aa(&tail_points, &[5.0, 3.0, 1.75, 1.0], def);

        // body spot
        geometry.fill_circle_aa(
            body_back + body_dir.perp() * dir_sign * 5.0 + body_dir * 2.0,
            2.0,
            8,
            ldef,
        );
        // ears
        geometry.stroke_capsule_chain_aa(&ear_points_l, &[2.0, 3.0, 0.5], def);
        geometry.stroke_capsule_chain_aa(&ear_points_r, &[2.0, 3.0, 0.5], ldef);

        geometry.stroke_capsule_chain_aa(&[head_back.into(), head_front.into()], &[7.0, 3.5], def);
        //  mask strap
        geometry.stroke_capsule_chain_aa(
            &[
                (head_back + vec2(-7.0 * dir_sign, -1.0)).into(),
                (head_back + vec2(2.5 * dir_sign, -2.0)).into(),
            ],
            &[1.0, 1.0],
            vmask,
        );

        // mask
        geometry.stroke_capsule_chain_aa(
            &[
                (head_back + vec2(2.5 * dir_sign, -2.0)).into(),
                (head_back + vec2(6.5 * dir_sign, -2.0)).into(),
            ],
            &[3.5, 3.5],
            vmask,
        );
        geometry.stroke_capsule_chain_aa(
            &[
                (head_back + vec2(6.0 * dir_sign, 7.0)).into(),
                (head_back + vec2(8.0 * dir_sign, 8.0)).into(),
            ],
            &[4.5, 4.5],
            vmask,
        );
        geometry.stroke_polyline_aa(&ellipse, true, 1.0, [0, 255, 0, 255]);
        geometry.stroke_polyline_aa(&ellipse2, true, 1.0, [89, 89, 89, 255]);
        geometry.stroke_polyline_aa(&ellipse3, true, 1.0, [64, 64, 64, 255]);
        geometry.fill_circle_aa(shoulder, 4.5, 16, def);
        geometry.stroke_capsule_chain_aa(
            &[
                shoulder.into(),
                elbow.into(),
                front_paw.into(),
                front_paw_tip.into(),
            ],
            &[4.5, 2.0, 1.5, 0.5],
            def,
        );
        geometry.stroke_capsule_chain_aa(
            &[
                thigh.into(),
                knee.into(),
                knee.lerp(back_paw_start, 0.9).into(),
                back_paw_start.into(),
                back_paw_base.into(),
                back_paw_tip.into(),
            ],
            &[5.5, 3.5, 2.5, 2.5, 2.5, 1.5],
            def,
        );

        if debug_alpha > 0.0 {
            let alpha = (255.0 * debug_alpha) as u8;
            geometry.stroke_circle_aa(
                self.pos + offset,
                RABBIT_RADIUS as _,
                1.0,
                32,
                [255, 255, 255, alpha],
            );
            geometry.fill_circle_aa(front_contact, 4.5, 16, [255, 0, 0, alpha]);
            geometry.fill_circle_aa(back_contact, 4.5, 16, [0, 0, 200, alpha]);
            geometry.stroke_polyline_aa(
                &[body_front, body_front + -normal.perp() * dir_sign * 10.0],
                false,
                2.0,
                [255, 0, 0, alpha],
            );
            geometry.stroke_polyline_aa(
                &[body_front, body_front + body_dir * 10.0],
                false,
                2.0,
                [0, 200, 0, alpha],
            );
            geometry.stroke_polyline_aa(
                &[
                    body_front,
                    body_front + self.velocity * DEBUG_VELOCITY_DRAW_SCALE,
                ],
                false,
                2.0,
                [255, 0, 255, alpha],
            );
        }
    }
}

pub fn march_circle<F>(start: Vec2, end: Vec2, radius: f32, distance_func: F) -> (Vec2, bool)
where
    F: Fn(Vec2) -> f32,
{
    let delta = end - start;
    let max_length = delta.length();
    let start_d = distance_func(start);
    if start_d < 0.0 {
        return (start, true);
    }
    if max_length == 0.0 {
        if start_d < radius {
            return (start, true);
        } else {
            return (end, false);
        }
    }
    let dir = delta / max_length;

    let mut l = 0.0;
    let mut hit = false;
    let forward_step = 1.0;
    let backward_step = 0.125;
    // current_r is used to escape penetration
    let mut current_r = radius.min(start_d);
    loop {
        let point = start + dir * l;
        let d = distance_func(point);
        if d >= current_r {
            // to make sure that we can continue marching in d == 0 areas we overstep
            l = (l + (d - current_r).max(forward_step)).min(max_length);
        }
        if d < current_r {
            hit = true;
            break;
        }
        if l == max_length {
            // it is possible that we are hitting here, due to overstepping
            break;
        }
        current_r = current_r.max(d.min(radius));
    }

    let mut b = l;
    while b > 0.0 {
        let point = start + dir * b;
        let sample = distance_func(point);
        let d = sample - current_r;
        if d >= 0.0 {
            if b == max_length {
                return (end, hit);
            } else {
                let result = start + dir * b;
                assert!(hit || result == end);
                return (result, hit);
            }
        } else {
            hit = true;
        }
        b -= d.max(backward_step);
    }
    assert!(hit || start == end);
    return (start, hit);
}

pub fn march_circle_sliding<F, N>(
    start: Vec2,
    start_dir: Vec2,
    max_length: f32,
    radius: f32,
    slide_cost: f32,
    distance_func: F,
    normal_func: N,
) -> Vec2
where
    F: Fn(Vec2) -> f32,
    N: Fn(Vec2) -> Vec2,
{
    let start_d = distance_func(start);
    if start_d < 0.0 {
        return start;
    }
    if max_length == 0.0 {
        return start;
    }
    assert!(max_length > 0.0);

    let forward_step = 0.25;

    // current_r is used to escape penetration
    let mut current_r = radius.min(start_d);
    let mut point = start;
    let mut d = start_d;
    let max_iterations = ((max_length * 8.0 / forward_step).ceil() as i32).max(64);
    let mut remaining_distance = max_length;
    for _ in 0..max_iterations {
        let step_distance = (d - current_r).max(forward_step).min(remaining_distance);
        //info!(" {} step_distance {}", i, step_distance);
        let mut new_point = point + start_dir * step_distance;
        let new_d = distance_func(new_point);
        let passed_distance = if new_d < current_r {
            let normal = normal_func(new_point);
            let slid_point = new_point + normal * (current_r - new_d);
            let eps = 0.1;
            let slid_sample = distance_func(slid_point);
            if slid_sample < current_r - eps {
                break;
            }
            new_point = slid_point;
            (new_point - point).length() * slide_cost
        } else {
            step_distance
        };
        d = new_d;
        point = new_point;
        remaining_distance = (remaining_distance - passed_distance).max(0.0);
        if remaining_distance == 0.0 {
            break;
        }
        current_r = current_r.max(d.min(radius));
    }
    point
}

fn march_parabola<F>(
    start: Vec2,
    radius: f32,
    vel: Vec2,
    y_accel: f32,
    max_distance: f32,
    step_t: f32,
    distance_func: &F,
) -> (Vec2, bool)
where
    F: Fn(Vec2) -> f32,
{
    let mut distance_traveled = 0.0;
    let mut pos = start;
    let max_velocity = MAX_VELOCITY;
    let max_velocity_sq = max_velocity * max_velocity;
    let mut vel = vel;
    for _ in 0..1024 {
        vel = vel + vec2(0.0, y_accel * step_t);
        let vel_len = vel.length_squared();
        if vel_len > max_velocity_sq {
            vel = vel * max_velocity / vel_len;
        }
        let new_pos = pos + vel * step_t;
        let (new_pos, hit) = march_circle(pos, new_pos, radius, distance_func);
        if hit {
            return (new_pos, true);
        }
        distance_traveled += (new_pos - pos).length();
        if distance_traveled >= max_distance {
            return (new_pos, false);
        }
        pos = new_pos;
    }
    return (pos, false);
}

pub fn linearstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0)
}
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let x = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    x * x * (3.0 - 2.0 * x)
}
fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    // by Ken Perlin
    let x = ((x - edge0) / (edge1 - edge0)).max(0.0).min(1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

fn float_hash(p: Vec2) -> f32 {
    ((p.dot(vec2(12.9898, 78.233))).sin() * 43758.5453).fract()
}

fn two_segment_ik(a: f32, b: f32, d: f32) -> Vec2 {
    assert!(d >= (a - b).abs());
    vec2(
        0.5 * (a * a - b * b + d * d) / d,
        -0.5 * (-(a - b - d) * (a - b + d) * (a + b - d) * (a + b + d)).sqrt() / d,
    )
}

fn ankle_position(start: Vec2, end: Vec2, up_sign: f32, upper_len: f32, lower_len: f32) -> Vec2 {
    let total_len = lower_len + upper_len;
    let len_diff = (upper_len - lower_len).abs();
    let delta = end - start;
    let delta_len = delta.length();
    if delta_len <= len_diff {
        // compress longer segment to reach this pose
        let (dir, point, len) = if upper_len > lower_len {
            (delta, end, lower_len)
        } else {
            (-delta, start, upper_len)
        };
        point + dir.normalize_or_zero() * len
    } else if delta_len < total_len {
        let delta_dir = delta / delta_len;
        let ik = two_segment_ik(upper_len, lower_len, delta_len);
        let delta_perp = vec2(-delta_dir.y, delta_dir.x);
        start + ik.x * delta_dir + up_sign * ik.y * delta_perp
    } else {
        // extend
        let fraction = upper_len / total_len;
        start + delta * fraction
    }
}

trait LocalLerp {
    fn lerp(self, b: Self, f: f32) -> Self;
}
impl LocalLerp for f32 {
    fn lerp(self, b: Self, f: f32) -> Self {
        self * (1.0 - f) + b * f
    }
}

impl RabbitMap {
    pub fn new() -> Self {
        Self {
            boxes: [
                (vec2(0.0, 90.0), vec2(500.0, 10.0)),
                (vec2(-500.0, 50.0), vec2(10.0, 50.0)),
                (vec2(500.0, 50.0), vec2(10.0, 50.0)),
                (vec2(40.0, 80.0), vec2(15.0, 15.0)),
            ],
        }
    }

    pub fn distance(&self, p: Vec2) -> f32 {
        let mut d = f32::MAX;
        for &(center, half_extents) in &self.boxes {
            d = d.min(sd_box(p - center, half_extents));
        }
        d
    }

    pub fn normal(&self, p: Vec2) -> Vec2 {
        let c = self.distance(p);
        let grad = vec2(
            self.distance(p + vec2(1.0, 0.0)) - c,
            self.distance(p + vec2(0.0, 1.0)) - c,
        );
        grad.try_normalize().unwrap_or(vec2(0.0, -1.0))
    }

    pub fn draw(&self, geometry: &mut GeometryBatch<VertexPos3UvColor>, offset: Vec2) {
        for &(pos, half_extents) in &self.boxes {
            geometry.fill_rect(
                pos - half_extents + offset,
                pos + half_extents + offset,
                [32, 32, 32, 255],
            );
        }
    }
}

fn sd_box(p: Vec2, b: Vec2) -> f32 {
    let q = p.abs() - b;
    q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0)
}
