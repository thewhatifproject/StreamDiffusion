# Multi-Stage Image Processing System Redesign Plan

This is what it might look like to continue reimplementing all of the capabilities we have out of the box with comfystream already. 

## Current System Analysis

### Existing Architecture
The current system has a focused ControlNet preprocessor architecture:

- **Base Class**: `BasePreprocessor` with standardized interface
- **Registry System**: Dynamic preprocessor registration and lookup
- **Specialized Processors**: Canny, Depth, OpenPose, LineArt, MediaPipe Pose/Segmentation
- **Integration**: Tight coupling with ControlNet pipeline for conditioning
- **Performance**: GPU-optimized tensor processing with CPU fallbacks

### Current Limitations
1. **Single Purpose**: Preprocessors only serve ControlNet conditioning
2. **No Post-processing**: Missing output enhancement capabilities  
3. **Limited Reusability**: Components can't be easily composed or reused
4. **Rigid Pipeline**: Linear flow from input → preprocessor → ControlNet
5. **Missing Stages**: No intermediate processing, filtering, or enhancement steps

## Proposed Multi-Stage Architecture

### Core Concepts

#### 1. Processing Stages
```
Input → Pre-processing → Generation → Post-processing → Output
    ↓         ↓            ↓           ↓            ↓
  Filter   Transform    Condition   Enhance    Finalize
```

#### 2. Component Types
- **Filters**: Image quality, similarity, content filtering
- **Preprocessors**: Input preparation and conditioning
- **Generators**: Diffusion model inference  
- **Postprocessors**: Output enhancement and refinement
- **Finalizers**: Output formatting and delivery

### Detailed Design

#### Stage 1: Input Processing
**Purpose**: Filter, validate, and prepare incoming images

**Components**:
- `InputValidator` - Format validation, size checks
- `QualityFilter` - Blur detection, quality assessment
- `SimilarityFilter` - Duplicate frame detection (existing)
- `ContentFilter` - Safety, content moderation
- `NormalizationProcessor` - Color space, resolution standardization

#### Stage 2: Pre-processing
**Purpose**: Transform input for optimal generation

**Components** (Enhanced existing):
- All current preprocessors (Canny, Depth, etc.)
- `BackgroundSegmentation` - MediaPipe segmentation for background removal
- `ObjectSegmentation` - Object isolation and masking
- `StyleTransfer` - Artistic style preprocessing
- `GeometricTransform` - Perspective, rotation, scaling

#### Stage 3: Generation
**Purpose**: Run diffusion model with multi-modal conditioning

**Components**:
- Enhanced ControlNet pipeline (existing)
- `MultiModalConditioner` - Combine multiple conditioning inputs
- `AdaptiveScheduler` - Dynamic step scheduling
- `RegionController` - Spatial conditioning control

#### Stage 4: Post-processing  
**Purpose**: Enhance and refine generated outputs

**Components**:
- `QualityEnhancer` - Upscaling, sharpening, detail enhancement
- `ColorCorrector` - Color grading, tone mapping
- `StyleHarmonizer` - Style consistency enforcement
- `ArtifactRemover` - Noise reduction, artifact cleanup
- `CompositeBlender` - Multi-image blending and composition

#### Stage 5: Output Finalization
**Purpose**: Format and deliver final results

**Components**:
- `OutputFormatter` - Format conversion, compression
- `MetadataEmbedder` - Generation parameters, timestamps
- `QualityValidator` - Final quality checks
- `DeliveryHandler` - File saving, streaming, callbacks

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Base Processing Framework
- Create `BaseProcessor` interface
- Implement `ProcessingStage` enum
- Design `ProcessingContext` for state management
- Create `ProcessingPipeline` orchestrator

#### 1.2 Stage Management System
- `StageRegistry` for component discovery
- `PipelineBuilder` for dynamic pipeline construction
- `ProcessingGraph` for non-linear processing flows
- Configuration system for stage parameters

#### 1.3 Enhanced Base Classes
```python
class BaseProcessor(ABC):
    """Universal base class for all processing components"""
    stage: ProcessingStage
    priority: int
    supports_batch: bool
    supports_gpu: bool
    
    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext
    
    def can_process(self, context: ProcessingContext) -> bool
    def get_requirements(self) -> List[str]
    def get_outputs(self) -> List[str]
```

### Phase 2: Stage Implementation (Week 3-6)

#### 2.1 Input Processing Stage
- Migrate `SimilarImageFilter` to new architecture
- Implement quality assessment tools
- Create content filtering framework
- Add format validation and normalization

#### 2.2 Enhanced Pre-processing
- Refactor existing preprocessors to new base class
- Implement segmentation-based processors
- Add geometric transformation tools
- Create style transfer components

#### 2.3 Post-processing Framework
- Design enhancement pipeline architecture
- Implement quality improvement tools
- Create color correction and style harmonization
- Add artifact removal capabilities

#### 2.4 Output Processing
- Create flexible output formatting system
- Implement metadata embedding
- Add quality validation pipeline
- Design delivery mechanism framework

### Phase 3: Integration & Optimization (Week 7-8)

#### 3.1 Pipeline Integration
- Integrate with existing ControlNet system
- Maintain backward compatibility
- Update examples and documentation
- Performance optimization and caching

#### 3.2 Advanced Features
- Conditional processing flows
- Dynamic stage selection
- Performance monitoring and profiling
- Memory optimization for long pipelines

## Technical Architecture

### Core Classes

```python
class ProcessingContext:
    """Carries state through processing pipeline"""
    input_image: Union[Image.Image, torch.Tensor]
    intermediate_results: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_history: List[str]
    device: torch.device
    dtype: torch.dtype

class ProcessingPipeline:
    """Orchestrates multi-stage processing"""
    stages: List[ProcessingStage]
    processors: Dict[ProcessingStage, List[BaseProcessor]]
    
    def process(self, context: ProcessingContext) -> ProcessingContext
    def add_processor(self, processor: BaseProcessor, stage: ProcessingStage)
    def remove_processor(self, processor_id: str)

class StageRegistry:
    """Central registry for all processors"""
    def register_processor(self, processor_class: Type[BaseProcessor])
    def get_processors(self, stage: ProcessingStage) -> List[BaseProcessor]
    def build_pipeline(self, config: PipelineConfig) -> ProcessingPipeline
```

### Configuration System

```yaml
# example_pipeline.yaml
pipeline:
  input_processing:
    - processor: quality_filter
      params:
        min_quality: 0.7
    - processor: similarity_filter
      params:
        threshold: 0.98
  
  preprocessing:
    - processor: mediapipe_segmentation
      params:
        threshold: 0.5
        output_mode: background
    - processor: canny
      params:
        low_threshold: 50
        high_threshold: 100
  
  postprocessing:
    - processor: quality_enhancer
      params:
        upscale_factor: 2
    - processor: color_corrector
      params:
        saturation: 1.2
```

## Use Cases & Benefits

### Enhanced Segmentation Workflow
```python
# Before: Single-purpose segmentation
segmentation_preprocessor = MediaPipeSegmentationPreprocessor(threshold=0.5)
control_image = segmentation_preprocessor.process(input_image)

# After: Multi-stage segmentation
pipeline = PipelineBuilder()
    .add_input_filter(QualityFilter(min_quality=0.7))
    .add_preprocessor(MediaPipeSegmentation(threshold=0.5, output_mode="mask"))
    .add_generator(ControlNetGenerator(model="segmentation"))
    .add_postprocessor(EdgeSmoother(radius=2))
    .add_finalizer(OutputFormatter(format="png"))
    .build()

result = pipeline.process(ProcessingContext(input_image))
```

### Background Replacement Pipeline
```python
pipeline = PipelineBuilder()
    .add_preprocessor(BackgroundSegmentation(threshold=0.7))
    .add_preprocessor(BackgroundRemoval())
    .add_generator(InpaintingGenerator(prompt="studio background"))
    .add_postprocessor(EdgeBlender(feather=5))
    .build()
```

### Style Transfer + Enhancement
```python
pipeline = PipelineBuilder()
    .add_preprocessor(StyleAnalyzer())
    .add_generator(StyleTransferGenerator())
    .add_postprocessor(QualityEnhancer(upscale=2))
    .add_postprocessor(StyleHarmonizer())
    .build()
```

## Migration Strategy

### Backward Compatibility
- Keep existing preprocessor interfaces functional
- Provide automatic migration utilities
- Maintain current configuration formats
- Gradual deprecation of old patterns

### Testing Strategy
- Unit tests for each processing component
- Integration tests for full pipelines
- Performance benchmarks vs current system
- Memory usage and GPU utilization monitoring

### Documentation
- Architecture documentation with diagrams
- Component development guides
- Pipeline configuration examples
- Migration guides for existing code

## Performance Considerations

### Optimization Strategies
- Lazy loading of heavy processors
- Intelligent caching between stages
- GPU memory pool management
- Batch processing optimization

### Memory Management
- Streaming processing for large images
- Configurable memory limits per stage
- Automatic cleanup of intermediate results
- Optional disk-based intermediate storage

### Scalability
- Parallel processing of independent stages
- Dynamic processor scaling based on load
- Cloud-native deployment support
- Distributed processing capabilities

## Success Metrics

### Technical Metrics
- Processing latency vs current system
- Memory usage efficiency
- GPU utilization optimization
- Pipeline flexibility (number of configurable flows)

### Developer Experience
- Time to implement new processors
- Configuration complexity reduction
- Code reusability improvement
- Documentation completeness

### Feature Completeness
- Number of supported processing stages
- Integration with existing workflows
- Backward compatibility maintenance
- Advanced feature adoption rate

## Next Steps

1. **Review & Feedback**: Gather team feedback on architecture design
2. **Prototype Development**: Build core infrastructure prototype
3. **Component Migration**: Port existing preprocessors to new system
4. **Testing & Validation**: Comprehensive testing across use cases
5. **Documentation & Examples**: Complete developer resources
6. **Production Deployment**: Gradual rollout with monitoring

This plan transforms the current focused preprocessor system into a comprehensive, reusable multi-stage image processing framework that supports the full lifecycle from input to output while maintaining the performance and flexibility that makes StreamDiffusion effective. 