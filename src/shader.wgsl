// Vertex shader

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) clip: vec2<f32>,

};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos: vec2<f32>,
};

const M = 1000u;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.pos = model.position;
    out.clip_position = vec4(model.clip, 0.0, 1.0);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let m = mandel(in.pos);
    return vec4(m,m,m,1.0);
}

fn mandel(p: vec2<f32>) -> f32 {
    var iter: u32 = 0u;
    var z = vec2<f32>(0.0);

    while iter < M {
        z = mul(z,z) + p;
        if z.x*z.x+z.y*z.y > 4.0 {
            break;
        }
     	iter = iter + 1u;
    }

    let res = f32(iter)/f32(M);

    return res*res;
}

fn mul(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return vec2(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x);
}
