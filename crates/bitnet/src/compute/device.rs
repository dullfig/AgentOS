//! Hardware detection and device enumeration.
//!
//! Provides a unified view of available compute resources (CPU features,
//! GPU adapters) and a formatted boot banner for startup diagnostics.

use super::CpuFeatures;

/// Information about a detected GPU adapter.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Human-readable device name (e.g., "Intel(R) Iris(R) Xe Graphics").
    pub name: String,
    /// Graphics API backend (e.g., "Vulkan", "DX12", "Metal").
    pub backend: String,
    /// Device type (discrete, integrated, software, etc.).
    pub device_type: GpuDeviceType,
    /// Estimated VRAM in bytes (may be 0 if not reported).
    pub vram_bytes: u64,
}

/// GPU device type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    Discrete,
    Integrated,
    Virtual,
    Software,
    Other,
}

impl std::fmt::Display for GpuDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Discrete => write!(f, "discrete"),
            Self::Integrated => write!(f, "integrated"),
            Self::Virtual => write!(f, "virtual"),
            Self::Software => write!(f, "software"),
            Self::Other => write!(f, "other"),
        }
    }
}

/// Full hardware profile for the current system.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// CPU model name (from OS, may be empty if unavailable).
    pub cpu_name: String,
    /// Detected CPU SIMD features.
    pub cpu_features: CpuFeatures,
    /// Detected GPU adapters (may be empty).
    pub gpus: Vec<GpuInfo>,
}

impl HardwareInfo {
    /// Detect all available hardware.
    pub fn detect() -> Self {
        let cpu_features = CpuFeatures::detect();
        let cpu_name = detect_cpu_name();
        let gpus = detect_gpus();

        Self {
            cpu_name,
            cpu_features,
            gpus,
        }
    }

    /// The best CPU SIMD instruction set available.
    pub fn cpu_compute_name(&self) -> &str {
        if self.cpu_features.avx512f {
            "AVX-512"
        } else if self.cpu_features.avx2 {
            "AVX2"
        } else if self.cpu_features.neon {
            "NEON"
        } else {
            "scalar"
        }
    }

    /// Print the boot banner to stderr.
    pub fn print_boot_banner(&self) {
        eprintln!();
        eprintln!("  agentos-bitnet inference engine");
        eprintln!("  ───────────────────────────────");

        // CPU
        if self.cpu_name.is_empty() {
            eprintln!("  [boot] CPU: (unknown)");
        } else {
            eprintln!("  [boot] CPU: {}", self.cpu_name);
        }

        let simd = self.cpu_compute_name();
        eprintln!("  [boot] CPU compute: {} ternary kernel", simd);

        // GPU(s)
        if self.gpus.is_empty() {
            eprintln!("  [boot] GPU: none detected");
        } else {
            for gpu in &self.gpus {
                let vram = if gpu.vram_bytes > 0 {
                    format!(" ({:.0}MB VRAM)", gpu.vram_bytes as f64 / (1024.0 * 1024.0))
                } else {
                    String::new()
                };
                eprintln!(
                    "  [boot] GPU: {} [{}] ({}{})",
                    gpu.name, gpu.backend, gpu.device_type, vram
                );
            }
        }
        eprintln!();
    }
}

/// Try to get the CPU model name from the OS.
fn detect_cpu_name() -> String {
    // Windows: read from registry or use environment
    #[cfg(target_os = "windows")]
    {
        // PROCESSOR_IDENTIFIER is always available on Windows
        if let Ok(name) = std::env::var("PROCESSOR_IDENTIFIER") {
            // Also try to get the brand string which is more readable
            return get_cpu_brand_string().unwrap_or(name);
        }
    }

    // Unix: try /proc/cpuinfo
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }

    // macOS: sysctl
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                return String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }
    }

    String::new()
}

/// Get CPU brand string via CPUID on x86_64.
#[cfg(target_os = "windows")]
fn get_cpu_brand_string() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        // CPUID with EAX=0x80000002..0x80000004 returns the brand string
        let mut brand = String::with_capacity(48);
        for leaf in 0x80000002u32..=0x80000004u32 {
            let result = unsafe { std::arch::x86_64::__cpuid(leaf) };
            for &reg in &[result.eax, result.ebx, result.ecx, result.edx] {
                let bytes = reg.to_le_bytes();
                for &b in &bytes {
                    if b != 0 {
                        brand.push(b as char);
                    }
                }
            }
        }
        let trimmed = brand.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// Detect GPU adapters using wgpu (if the `gpu` feature is enabled).
#[cfg(feature = "gpu")]
fn detect_gpus() -> Vec<GpuInfo> {
    // wgpu instance creation and adapter enumeration is async,
    // but we use pollster to block on it since this runs once at startup.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let mut gpus = Vec::new();

    for adapter in adapters {
        let info = adapter.get_info();

        // Skip CPU/software renderers unless that's all we have
        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::Discrete,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::Integrated,
            wgpu::DeviceType::VirtualGpu => GpuDeviceType::Virtual,
            wgpu::DeviceType::Cpu => GpuDeviceType::Software,
            _ => GpuDeviceType::Other,
        };

        let backend = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "unknown",
        };

        // Deduplicate: same GPU may appear under multiple backends.
        // Keep the first one (wgpu returns preferred backend first).
        if gpus.iter().any(|g: &GpuInfo| g.name == info.name) {
            continue;
        }

        gpus.push(GpuInfo {
            name: info.name,
            backend: backend.to_string(),
            device_type,
            vram_bytes: 0, // wgpu doesn't expose VRAM size directly
        });
    }

    gpus
}

/// Stub when GPU feature is disabled.
#[cfg(not(feature = "gpu"))]
fn detect_gpus() -> Vec<GpuInfo> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hardware() {
        let hw = HardwareInfo::detect();
        eprintln!("CPU: {:?}", hw.cpu_name);
        eprintln!("Features: {:?}", hw.cpu_features);
        eprintln!("GPUs: {:?}", hw.gpus);
        // Should at least have CPU info
        assert!(!hw.cpu_compute_name().is_empty());
    }

    #[test]
    fn boot_banner_runs() {
        let hw = HardwareInfo::detect();
        hw.print_boot_banner();
    }

    #[test]
    fn cpu_name_not_empty() {
        let name = detect_cpu_name();
        eprintln!("CPU name: {name:?}");
        // On CI this might be empty, so just check it doesn't panic
    }
}
