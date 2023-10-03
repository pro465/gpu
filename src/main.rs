fn main() {
    pollster::block_on(mandelbrot_gpu::run());
}
