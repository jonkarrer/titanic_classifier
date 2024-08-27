mod data;
mod model;
mod training;

use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Autodiff, Wgpu,
};
use training::train;

pub type MyDevice = Wgpu<AutoGraphicsApi, f32, i32>;
pub type MyBackend = Autodiff<MyDevice>;

pub fn get_device() -> WgpuDevice {
    WgpuDevice::default()
}

fn main() {
    let device = get_device();
    train::<MyBackend>(device);
}
