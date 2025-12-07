// System information command

use anyhow::Result;
use colored::Colorize;
use sysinfo::{System, SystemExt, CpuExt};

use crate::DeviceType;

pub async fn show_system_info(device: &DeviceType) -> Result<()> {
    println!("{}", "System Information".bold().cyan());
    println!("{}", "═".repeat(60).cyan());
    println!();

    // System info
    let mut sys = System::new_all();
    sys.refresh_all();

    println!("{}", "  CPU:".bold());
    if let Some(cpu) = sys.cpus().first() {
        println!("    Model: {}", cpu.brand());
        println!("    Cores: {} (physical)", sys.cpus().len());
    }

    println!();
    println!("{}", "  Memory:".bold());
    println!("    Total: {:.2} GB", sys.total_memory() as f64 / 1024_f64.powi(3));
    println!("    Available: {:.2} GB", sys.available_memory() as f64 / 1024_f64.powi(3));

    println!();
    println!("{}", "  GPU:".bold());

    // Check GPU availability
    #[cfg(feature = "gpu")]
    {
        match check_gpu_info().await {
            Ok(info) => {
                println!("    Status: {}", "Available ✓".green());
                println!("    Name: {}", info.name);
                println!("    Compute Capability: {}", info.compute_cap);
                println!("    Memory: {:.2} GB", info.memory_gb);
                println!("    CUDA Version: {}", info.cuda_version);
            }
            Err(e) => {
                println!("    Status: {}", "Not available".yellow());
                println!("    Reason: {}", e);
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("    Status: {}", "Disabled (compiled without GPU support)".yellow());
        println!("    Rebuild with: cargo build --features gpu");
    }

    println!();
    println!("{}", "  Runtime:".bold());
    println!("    Device mode: {:?}", device);
    println!("    Tokio threads: {}", tokio::runtime::Handle::current().metrics().num_workers());

    println!();
    println!("{}", "  Build Info:".bold());
    println!("    Version: {}", env!("CARGO_PKG_VERSION"));
    println!("    Features: {}", get_enabled_features());
    println!("    Target: {}", std::env::consts::ARCH);

    Ok(())
}

#[cfg(feature = "gpu")]
async fn check_gpu_info() -> Result<GpuInfo> {
    use cudarc::driver::CudaDevice;

    let device = CudaDevice::new(0)?;
    let props = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;

    Ok(GpuInfo {
        name: device.name()?,
        compute_cap: format!("{}.{}", props, 0),
        memory_gb: device.total_memory()? as f64 / 1024_f64.powi(3),
        cuda_version: "11.7".to_string(), // TODO: Get actual version
    })
}

#[cfg(feature = "gpu")]
struct GpuInfo {
    name: String,
    compute_cap: String,
    memory_gb: f64,
    cuda_version: String,
}

fn get_enabled_features() -> String {
    let mut features = Vec::new();

    #[cfg(feature = "gpu")]
    features.push("gpu");

    #[cfg(feature = "cpu-only")]
    features.push("cpu-only");

    if features.is_empty() {
        "none".to_string()
    } else {
        features.join(", ")
    }
}
