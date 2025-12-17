# Deploying CLIP in C++ Frameworks

This document analyzes options for running CLIP models in C++ applications.

---

## 1. ONNX Runtime

**Overview**: Export PyTorch model to ONNX format, run inference with Microsoft's ONNX Runtime.

### Pros
- Cross-platform (Windows, Linux, macOS, mobile, web)
- Hardware-agnostic (CPU, CUDA, DirectML, CoreML, etc.)
- Well-maintained by Microsoft with regular updates
- Easy export from HuggingFace via `optimum` library
- Supports quantization (INT8, FP16) out of the box
- Large community and extensive documentation
- Can run in browsers via ONNX.js/ort-web

### Cons
- Export can be tricky for complex models with dynamic shapes
- Some PyTorch ops may not have ONNX equivalents
- Additional dependency (ONNX Runtime library ~50-200MB)
- Tokenizer must be handled separately (not included in ONNX)
- Performance may be slightly lower than native frameworks on specific hardware

### Best For
- **Cross-platform deployment**
- **Cloud/server inference**
- **When hardware flexibility is important**

---

## 2. LibTorch (TorchScript)

**Overview**: Trace or script PyTorch model, load in C++ using LibTorch.

### Pros
- Native PyTorch support - most ops work out of the box
- Maintains exact PyTorch behavior and numerical precision
- Can use `torch.jit.trace()` or `torch.jit.script()`
- Full CUDA support with identical GPU kernels
- Easier debugging (same framework as training)
- Supports custom C++ extensions

### Cons
- Large library size (~1-2GB with CUDA)
- Slower startup time (library initialization)
- TorchScript has limitations with dynamic control flow
- Tracing requires representative input shapes
- Heavier memory footprint than alternatives
- Limited mobile support compared to ONNX

### Best For
- **Research/prototyping in C++**
- **When exact PyTorch parity is required**
- **Projects already using LibTorch**

---

## 3. OpenVINO (Intel)

**Overview**: Intel's toolkit for optimizing and deploying models on Intel hardware.

### Pros
- Excellent CPU performance on Intel processors
- Supports Intel GPUs, VPUs (Movidius), and FPGAs
- Aggressive optimizations (layer fusion, quantization)
- Lower memory usage than LibTorch
- INT8 quantization with accuracy-aware tuning
- Good documentation and Intel support

### Cons
- **Intel-only** - poor/no support for AMD or ARM
- Requires model conversion (ONNX → OpenVINO IR)
- Some ops may not be supported
- Less flexible than ONNX Runtime
- Larger learning curve for optimization features
- Not suitable for NVIDIA GPU deployment

### Best For
- **Intel CPU deployment (servers, edge)**
- **Intel integrated/discrete GPUs**
- **Maximum CPU inference performance**

---

## 4. TensorRT (NVIDIA)

**Overview**: NVIDIA's high-performance inference optimizer and runtime.

### Pros
- **Fastest inference on NVIDIA GPUs** (often 2-5x faster)
- Automatic kernel tuning for specific GPU
- FP16/INT8 quantization with minimal accuracy loss
- Layer fusion and memory optimization
- Supports dynamic batching
- Well-integrated with NVIDIA ecosystem (Triton, DeepStream)

### Cons
- **NVIDIA GPUs only** - no CPU or other vendor support
- Complex setup and conversion process
- Engine files are GPU-architecture specific (not portable)
- Long optimization/build times (minutes to hours)
- Some ops require plugins or custom implementation
- Frequent breaking changes between versions

### Best For
- **Production NVIDIA GPU deployment**
- **Real-time inference with strict latency requirements**
- **Maximum throughput on NVIDIA hardware**

---

## 5. clip.cpp (GGML)

**Overview**: Lightweight C++ implementation using GGML (same backend as llama.cpp).

### Pros
- **Minimal dependencies** - pure C/C++
- Very small binary size (~few MB)
- Runs on CPU without heavy frameworks
- Supports quantization (4-bit, 8-bit)
- Easy to build and integrate
- Good for edge/embedded devices
- Active community (llama.cpp ecosystem)

### Cons
- CPU-only (no GPU acceleration currently)
- Limited model support (not all CLIP variants)
- Less mature than other options
- May have slight accuracy differences from original
- Smaller community than ONNX/TensorRT
- Manual updates needed for new models

### Best For
- **Edge/embedded deployment**
- **Minimal dependency requirements**
- **CPU-only environments**
- **Quick prototyping**

---

## Comparison Summary

| Feature | ONNX Runtime | LibTorch | OpenVINO | TensorRT | clip.cpp |
|---------|--------------|----------|----------|----------|----------|
| **Platforms** | All | All | Intel only | NVIDIA only | All (CPU) |
| **GPU Support** | Multi-vendor | NVIDIA | Intel | NVIDIA | No |
| **Binary Size** | Medium | Large | Medium | Medium | Small |
| **Performance** | Good | Good | Best (Intel CPU) | Best (NVIDIA GPU) | Good (CPU) |
| **Setup Complexity** | Low | Medium | Medium | High | Low |
| **Quantization** | Yes | Limited | Yes | Yes | Yes |
| **Production Ready** | Yes | Yes | Yes | Yes | Maturing |

---

## Recommendations

| Scenario | Recommended Option |
|----------|-------------------|
| Cross-platform deployment | **ONNX Runtime** |
| NVIDIA GPU production | **TensorRT** |
| Intel CPU servers | **OpenVINO** |
| Edge/embedded devices | **clip.cpp** or **ONNX Runtime** |
| Research/prototyping | **LibTorch** |
| Browser/web deployment | **ONNX Runtime** (via ort-web) |
| Mobile (iOS/Android) | **ONNX Runtime** or **TensorFlow Lite** |

---

## Export Examples

### Python → ONNX
```python
from optimum.onnxruntime import ORTModelForImageClassification
model = ORTModelForImageClassification.from_pretrained(
    "openai/clip-vit-base-patch32", export=True
)
model.save_pretrained("clip_onnx/")
```

### Python → TorchScript
```python
import torch
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
traced = torch.jit.trace(model, example_inputs)
traced.save("clip_traced.pt")
```

### ONNX → TensorRT
```bash
trtexec --onnx=clip.onnx --saveEngine=clip.trt --fp16
```

### ONNX → OpenVINO
```bash
mo --input_model clip.onnx --output_dir clip_openvino/
```
