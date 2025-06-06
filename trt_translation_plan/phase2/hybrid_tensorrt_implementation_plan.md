# Hybrid TensorRT Implementation Plan: Phase 2
## ControlNet + UNet TensorRT Optimization with Modular Flexibility

---

## **Summary**

This plan outlines the implementation of a hybrid TensorRT architecture where both ControlNet and UNet models are compiled as separate TensorRT engines while maintaining full runtime flexibility for dynamic ControlNet management.

### **Current Architecture:**
```
ControlNet (PyTorch) → outputs → UNet (TensorRT)
```

### **Target Architecture:**
```
ControlNet (TensorRT) → outputs → UNet (TensorRT)
```

---

## **Motivation**

### **Performance Goals:**
- **ControlNet acceleration**: 2-5x faster inference (currently ~10-20ms → target ~2-5ms)
- **Total pipeline speedup**: 1.5-3x overall improvement
- **Memory efficiency**: 20-40% reduction in GPU memory usage
- **Latency consistency**: More predictable frame times

### **Flexibility Requirements:**
- ✅ Dynamic ControlNet addition/removal at runtime
- ✅ Multiple ControlNet types (Canny, Depth, Pose, OpenPose, etc.)
- ✅ Adjustable conditioning scales per ControlNet
- ✅ Backward compatibility with PyTorch fallback

### **Strategic Benefits:**
- **Competitive advantage**: Best-in-class real-time performance
- **Resource efficiency**: Lower hardware requirements
- **User experience**: Smoother real-time applications
- **Scalability**: Support for more complex multi-ControlNet workflows

---

## 📊 **Current Architecture Analysis**

### **Key Components in Current Setup:**

#### **Core Files:**
```
StreamDiffusion/src/streamdiffusion/
├── acceleration/tensorrt/
│   ├── engine.py                    # UNet TensorRT engine
│   ├── models.py                    # UNet model definitions with ControlNet support
│   ├── builder.py                   # TensorRT compilation utilities
│   ├── utilities.py                 # Engine management utilities
│   └── controlnet_wrapper.py       # ONNX export wrapper
├── controlnet/
│   ├── base_controlnet_pipeline.py # ControlNet PyTorch execution
│   ├── controlnet_pipeline.py      # SD1.5 ControlNet implementation
│   └── controlnet_sdxlturbo_pipeline.py # SDXL ControlNet implementation
└── pipeline.py                     # Core StreamDiffusion pipeline
```

#### **Current Execution Flow:**
1. **ControlNet inference** (PyTorch): `base_controlnet_pipeline.py:_get_controlnet_conditioning()`
2. **Output formatting**: Convert ControlNet outputs to UNet inputs
3. **UNet inference** (TensorRT): `engine.py:UNet2DConditionModelEngine.__call__()`

#### **Current Strengths:**
- ✅ Clean separation between ControlNet and UNet
- ✅ Modular ControlNet management
- ✅ Established TensorRT pipeline for UNet
- ✅ Comprehensive profiling already in place

#### **Current Bottlenecks:**
- ❌ ControlNet PyTorch inference (~40-60% of total time)
- ❌ GPU↔CPU transfers for ControlNet processing
- ❌ Memory fragmentation from mixed PyTorch/TensorRT execution

---

## 🏗 **Proposed Implementation Architecture**

### **Phase 2.1: ControlNet TensorRT Engine Infrastructure**

#### **New Components:**
```
StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/
├── controlnet_engine.py            # NEW: ControlNet TensorRT engine wrapper
├── controlnet_models.py            # NEW: ControlNet model definitions for TensorRT
├── controlnet_builder.py           # NEW: ControlNet compilation utilities
└── engine_pool.py                  # NEW: Multi-engine management
```

#### **Modified Components:**
```
StreamDiffusion/src/streamdiffusion/
├── acceleration/tensorrt/
│   ├── builder.py                   # MODIFY: Add ControlNet compilation
│   ├── utilities.py                 # MODIFY: Enhanced engine management
│   └── __init__.py                  # MODIFY: ControlNet acceleration integration
└── controlnet/
    └── base_controlnet_pipeline.py # MODIFY: TensorRT/PyTorch hybrid loading
```

### **Component Specifications:**

#### **1. ControlNet TensorRT Engine (`controlnet_engine.py`)**
```python
class ControlNetModelEngine:
    """TensorRT-accelerated ControlNet inference engine"""
    
    def __init__(self, engine_path: str, stream: cuda.Stream):
        self.engine = Engine(engine_path)
        self.stream = stream
        
    def __call__(self, 
                 sample: torch.Tensor,
                 timestep: torch.Tensor, 
                 encoder_hidden_states: torch.Tensor,
                 controlnet_cond: torch.Tensor,
                 conditioning_scale: float = 1.0) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through TensorRT ControlNet engine"""
        # Implementation details in Phase 2.2
```

#### **2. ControlNet Model Definitions (`controlnet_models.py`)**
```python
class ControlNetTRT(BaseModel):
    """TensorRT model definition for ControlNet compilation"""
    
    def __init__(self, controlnet_type: str = "canny", **kwargs):
        super().__init__(**kwargs)
        self.controlnet_type = controlnet_type
        self.name = f"ControlNet-{controlnet_type}"
        
    def get_input_names(self) -> List[str]:
        return ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"]
        
    def get_output_names(self) -> List[str]:
        # 12 down block outputs + 1 middle block output
        down_names = [f"down_block_{i}" for i in range(12)]
        return down_names + ["mid_block"]
        
    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "B", 2: "H", 3: "W"},
            "controlnet_cond": {0: "B", 2: "H_ctrl", 3: "W_ctrl"},
            # ... additional dynamic axes
        }
```

#### **3. Engine Pool Management (`engine_pool.py`)**
```python
class ControlNetEnginePool:
    """Manages multiple ControlNet TensorRT engines"""
    
    def __init__(self):
        self.engines: Dict[str, ControlNetModelEngine] = {}
        self.streams: Dict[str, cuda.Stream] = {}
        self.compiled_models: Set[str] = set()
        
    def get_or_load_engine(self, model_id: str, 
                          engine_dir: str) -> Union[ControlNetModelEngine, ControlNetModel]:
        """Load TensorRT engine if available, fallback to PyTorch"""
        # Implementation details in Phase 2.3
        
    def compile_controlnet_on_demand(self, model_id: str, 
                                   sample_inputs: Dict[str, torch.Tensor]) -> bool:
        """Compile ControlNet to TensorRT if not already compiled"""
        # Implementation details in Phase 2.4
```

---

##  **Implementation Phases**

### **Phase 2.1: Infrastructure Setup (Week 1-2)**

#### **Goals:**
- Create TensorRT ControlNet engine wrapper
- Establish compilation pipeline
- Implement basic engine loading

#### **Key Tasks:**
1. **Create `controlnet_engine.py`**
   - TensorRT engine wrapper class
   - Input/output handling
   - Error handling and fallbacks

2. **Create `controlnet_models.py`**
   - Model definitions for different ControlNet types
   - Dynamic axes configuration
   - Input/output specifications

3. **Extend `builder.py`**
   - Add `compile_controlnet()` function
   - Handle ControlNet-specific optimizations
   - Support multiple ControlNet architectures

#### **Success Criteria:**
- ✅ Can compile a single ControlNet to TensorRT
- ✅ Basic inference through TensorRT engine
- ✅ Fallback to PyTorch if compilation fails

#### **Risk Mitigation:**
- Start with simplest ControlNet (Canny)
- Maintain PyTorch fallback throughout
- Incremental testing at each step

### **Phase 2.2: Core Engine Implementation (Week 2-3)**

#### **Goals:**
- Implement full ControlNet TensorRT inference
- Handle dynamic input shapes
- Optimize memory management

#### **Key Tasks:**
1. **Complete `ControlNetModelEngine.__call__()`**
   ```python
   def __call__(self, sample, timestep, encoder_hidden_states, 
                controlnet_cond, conditioning_scale=1.0):
       # Prepare inputs with proper shapes
       input_dict = {
           "sample": sample,
           "timestep": timestep, 
           "encoder_hidden_states": encoder_hidden_states,
           "controlnet_cond": controlnet_cond
       }
       
       # Execute TensorRT inference
       outputs = self.engine.infer(input_dict, self.stream)
       
       # Extract and format outputs
       down_blocks = [outputs[f"down_block_{i}"] for i in range(12)]
       mid_block = outputs["mid_block"]
       
       # Apply conditioning scale
       if conditioning_scale != 1.0:
           down_blocks = [d * conditioning_scale for d in down_blocks]
           mid_block = mid_block * conditioning_scale
           
       return down_blocks, mid_block
   ```

2. **Handle Dynamic Shapes**
   - Variable control image resolutions
   - Batch size variations
   - Memory optimization for different input sizes

3. **Implement Engine Caching**
   - Avoid redundant allocations
   - Efficient memory reuse
   - Context switching optimization

#### **Success Criteria:**
- ✅ Full ControlNet inference through TensorRT
- ✅ Handles variable input shapes correctly
- ✅ Performance improvement vs PyTorch baseline

### **Phase 2.3: Integration & Pool Management (Week 3-4)**

#### **Goals:**
- Integrate TensorRT ControlNets into existing pipeline
- Implement engine pool management
- Support multiple simultaneous ControlNets

#### **Key Tasks:**
1. **Modify `base_controlnet_pipeline.py`**
   ```python
   def _load_controlnet_model(self, model_id: str):
       """Enhanced loading with TensorRT support"""
       # Check for existing TensorRT engine
       engine_path = os.path.join(self.engine_dir, f"controlnet_{model_id}.engine")
       
       if os.path.exists(engine_path):
           # Load TensorRT engine
           return self.engine_pool.get_or_load_engine(model_id, engine_path)
       else:
           # Attempt compilation or fallback to PyTorch
           return self._load_with_compilation_fallback(model_id)
   ```

2. **Implement `engine_pool.py`**
   - Multi-engine lifecycle management
   - CUDA stream coordination
   - Memory pool optimization

3. **Update `_get_controlnet_conditioning()`**
   - Support both TensorRT and PyTorch ControlNets
   - Maintain profiling capabilities
   - Optimize execution flow

#### **Success Criteria:**
- ✅ Seamless TensorRT/PyTorch hybrid operation
- ✅ Multiple ControlNets working simultaneously
- ✅ Proper resource cleanup and management

### **Phase 2.4: Optimization & Advanced Features (Week 4-5)**

#### **Goals:**
- Performance optimization
- Advanced compilation strategies
- Production-ready features

#### **Key Tasks:**
1. **On-Demand Compilation**
   ```python
   def compile_controlnet_on_demand(self, model_id: str, sample_inputs: Dict):
       """Compile ControlNet in background thread"""
       if model_id not in self.compiled_models:
           # Background compilation with user notification
           threading.Thread(target=self._background_compile, 
                           args=(model_id, sample_inputs)).start()
           return False  # Use PyTorch for now
       return True  # TensorRT available
   ```

2. **Performance Optimizations**
   - FP16 optimization
   - Kernel fusion opportunities
   - Memory layout optimization
   - CUDA graph support

3. **Advanced Engine Management**
   - Engine warming strategies
   - Memory pressure handling
   - Automatic fallback mechanisms

4. **Monitoring & Diagnostics**
   - Enhanced profiling with engine breakdown
   - Performance regression detection
   - Memory usage tracking

#### **Success Criteria:**
- ✅ 2-5x ControlNet performance improvement
- ✅ Robust error handling and recovery
- ✅ Production-ready stability


## 🛠 **Technical Implementation Details**

### **Compilation Pipeline Integration**

#### **Modified `accelerate_with_tensorrt()` in `__init__.py`:**
```python
def accelerate_with_tensorrt(stream: StreamDiffusion, engine_dir: str, **kwargs):
    # Existing UNet compilation...
    
    # NEW: ControlNet compilation detection and compilation
    if use_controlnet:
        controlnet_engine_dir = os.path.join(engine_dir, "controlnet")
        os.makedirs(controlnet_engine_dir, exist_ok=True)
        
        # Initialize ControlNet engine pool
        stream.controlnet_engine_pool = ControlNetEnginePool(controlnet_engine_dir)
        
        # Pre-compile common ControlNet types if requested
        if kwargs.get('precompile_controlnets', False):
            common_types = ['canny', 'depth', 'openpose']
            for cn_type in common_types:
                stream.controlnet_engine_pool.precompile_controlnet(cn_type)
    
    return stream
```

### **Dynamic Shape Handling**

#### **ControlNet Input Profile Generation:**
```python
def get_controlnet_input_profile(self, batch_size, image_height, image_width, 
                                static_batch, static_shape):
    """Generate TensorRT input profiles for ControlNet"""
    min_batch = batch_size if static_batch else self.min_batch
    max_batch = batch_size if static_batch else self.max_batch
    
    # Control image can be different resolution than latent
    min_ctrl_h = 256 if not static_shape else image_height
    max_ctrl_h = 1024 if not static_shape else image_height
    min_ctrl_w = 256 if not static_shape else image_width  
    max_ctrl_w = 1024 if not static_shape else image_width
    
    return {
        "sample": [
            (min_batch, 4, image_height//8, image_width//8),
            (batch_size, 4, image_height//8, image_width//8),
            (max_batch, 4, image_height//8, image_width//8),
        ],
        "controlnet_cond": [
            (min_batch, 3, min_ctrl_h, min_ctrl_w),
            (batch_size, 3, image_height, image_width),
            (max_batch, 3, max_ctrl_h, max_ctrl_w),
        ],
        # ... other inputs
    }
```

### **Error Handling & Fallback Strategy**

#### **Graceful Degradation:**
```python
class HybridControlNet:
    """Wrapper that handles TensorRT/PyTorch fallback"""
    
    def __init__(self, model_id: str, engine_pool: ControlNetEnginePool):
        self.model_id = model_id
        self.engine_pool = engine_pool
        self.trt_engine = None
        self.pytorch_model = None
        self.use_tensorrt = False
        
    def __call__(self, *args, **kwargs):
        try:
            if self.trt_engine is not None:
                return self.trt_engine(*args, **kwargs)
        except Exception as e:
            logging.warning(f"TensorRT ControlNet failed: {e}, falling back to PyTorch")
            self.use_tensorrt = False
            
        # Fallback to PyTorch
        if self.pytorch_model is None:
            self.pytorch_model = self._load_pytorch_model()
        return self.pytorch_model(*args, **kwargs)
```

---



## 🔄 **Future Roadmap**

### **Phase 3: Advanced Optimizations**
- **CUDA Graph Integration**: Further latency reduction
- **Multi-GPU Support**: Scale across multiple GPUs
- **Model Quantization**: INT8 optimization for edge devices
- **Batch Processing**: Optimize for batch inference scenarios

### **Phase 4: Ecosystem Expansion**
- **ONNX Runtime Integration**: Alternative to TensorRT
- **DirectML Support**: Windows native acceleration
- **Mobile Optimization**: ARM/Android deployment
- **Cloud Integration**: Scalable cloud deployment

---


## 📈 **Phase 2.1 Implementation Status - COMPLETED**

**Date Completed**: January 2025  
**Status**: ✅ **SUCCESS** - All success criteria met

### **Components Implemented:**
- ✅ `controlnet_models.py` - ControlNet TensorRT model definitions
- ✅ `controlnet_engine.py` - ControlNet TensorRT engine wrapper 
- ✅ `engine_pool.py` - Engine pool management infrastructure
- ✅ `builder.py` - Extended with `compile_controlnet()` function
- ✅ `test_controlnet_tensorrt.py` - Comprehensive test script

### **Success Criteria Results:**
- ✅ **Can compile a single ControlNet to TensorRT**: PASS
- ✅ **Basic inference through TensorRT engine**: PASS  
- ✅ **Fallback to PyTorch if compilation fails**: PASS

### **Performance Results:**
- **PyTorch ControlNet**: 14.01ms
- **TensorRT ControlNet**: 5.55ms
- **Speedup Achieved**: **2.52x** (meets 2-5x target)
- **Engine Size**: 706.2 MB
- **Compilation Time**: 188.47s

### **Next Phase Ready:**
Phase 2.1 infrastructure provides solid foundation for Phase 2.2: Core Engine Implementation and optimization. 