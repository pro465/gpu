// Vertex shader

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) diff: vec2<f32>,
    @location(2) actual: vec2<f32>,

};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) delta: vec2<f32>,
};

const M = 1000u;
var<private> zn: array<vec2<f32>, M>;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.delta = model.diff;
    out.clip_position = vec4(model.actual, 0.0, 1.0);

    var i = 0u;
    var z = vec2(0.0);
    var c = model.position;

    while i < M {
        let p = &zn;
        (*p)[i] = z;
	i += 1u;
	z = mul(z, z)+c;
    }

    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = &zn;
    let m = mandel(in.delta,p);
    return vec4(m,m,m,1.0);
}

fn mandel(d: vec2<f32>, zn: ptr<private, array<vec2<f32>, M>>) -> f32 {
    var iter: u32 = 0u;
    var e = vec2<f32>(0.0);

    while iter < M - 1u {
        let i = (*zn)[iter];

        //z = mul(z,z) + d;

        e = mul(e, e) + mul(2.0*e, i) + d;

	let z = e + (*zn)[iter + 1u];

	if z.x*z.x + z.y*z.y > 4.0 {
	    break;
	}

	iter = iter + 1u;
    }

    let res = f32(iter+1u)/f32(M);

    return res*res;
}

fn mul(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return vec2(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x);
}
