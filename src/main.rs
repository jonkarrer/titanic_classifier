mod data;
mod infer;
mod model;
mod training;

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};

pub type MyDevice = Wgpu<f32, i32>;
pub type MyBackend = Autodiff<MyDevice>;

pub fn get_device() -> WgpuDevice {
    WgpuDevice::default()
}

const MODEL_PATH: &str = "model/bce-adam";

fn main() {
    let device = get_device();
    // training::train::<MyBackend>(MODEL_PATH, device);
    infer::infer::<MyBackend>(MODEL_PATH, &device);
}
