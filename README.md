# Mobile Sculpture MCP Server

**Kinetic sculpture visual vocabulary based on Calder mobiles and balance physics**

## Overview

This MCP server provides a systematic vocabulary for mobile sculpture composition, combining deterministic physics calculations with aesthetic taxonomy. It follows the three-layer olog architecture for optimal cost efficiency.

## Three-Layer Architecture

### Layer 1: Pure Taxonomy (0 tokens)
- **Balance Types**: Asymmetric counterbalance, symmetric balance, cantilever dynamics, hierarchical suspension
- **Movement Vocabularies**: Rotational, pendular, oscillatory, cascading, bobbing
- **Material Surfaces**: Industrial metal, biomorphic organic, wire armature, reflective kinetic
- **Spatial Relationships**: Suspension geometry, inter-element clearance, depth layering, visual weight distribution

### Layer 2: Deterministic Operations (0 tokens)
- **Balance Physics**: Torque equilibrium calculations (Σ(weight × distance) = 0)
- **Keyword Extraction**: Pattern-based parameter detection from prompts
- **Geometric Specifications**: Explicit coordinates, angles, scales for image generation
- **Visual Vocabulary**: Translation from taxonomy to image-generation-ready keywords

### Layer 3: Creative Synthesis Interface (~150-200 tokens)
- **Natural Language Prompt Assembly**: Preserves geometric precision in readable form
- **Compositional Description**: Neutral description of parameters, not prescriptive outcomes

## Cost Savings

**60-85% reduction** vs pure LLM approach through:
- Zero-cost taxonomy lookup (Layer 1)
- Zero-cost physics calculations (Layer 2)
- Single LLM synthesis call (Layer 3 only)

## Key Features

### 1. Physics-Based Balance Calculations
```python
calculate_balance_physics(
    element_weights='[0.45, 0.30, 0.25]',
    arm_length=1.0,
    balance_type='asymmetric_counterbalance'
)
```

Returns:
- Suspension point location (torque equilibrium)
- Element positions along arm
- Angles from vertical
- Individual torque values
- Equilibrium verification

### 2. Explicit Geometric Specifications
Following Dal's preference for precision over suggestion:
```
"Element 0: disc at frame position [0.65, 0.72] (foreground layer),
scale factor 0.90, suspended at 35° from vertical,
wire length 0.36 units, rotated 0°, tilted 17.5°"
```

### 3. Movement Vocabularies
Describes kinetic **potential**, not frozen action:
- "Rotational capacity around vertical axis with 360° range"
- "Pendular arc-based swing potential with gravity-driven return"
- "Oscillatory motion across multiple axes with bounded displacement"

## Usage Workflow

### Basic Usage
```python
# 1. Map prompt to parameters (Layer 2)
params = map_prompt_to_mobile_parameters(
    "asymmetric mobile with spinning elements"
)

# 2. Generate geometric specs (Layer 2)
geo_specs = generate_mobile_geometric_specs(
    balance_type="asymmetric_counterbalance",
    element_count=5,
    movement_types='["rotational", "pendular"]',
    material_surface="industrial_metal"
)

# 3. Extract visual vocabulary (Layer 2)
vocab = extract_mobile_visual_vocabulary(geo_specs)

# 4. Prepare synthesis context (Layer 3 Interface)
context = prepare_mobile_synthesis_context(
    prompt="asymmetric mobile with spinning elements"
)
# Then: Claude synthesizes natural language prompt from context
```

### Complete Pipeline
```python
# One-step synthesis preparation
synthesis_context = prepare_mobile_synthesis_context(
    prompt="Create a Calder-style mobile with 4 elements",
    explicit_balance_type="asymmetric_counterbalance",
    explicit_element_count=4,
    explicit_movements='["rotational"]',
    explicit_material="industrial_metal"
)

# synthesis_context contains:
# - Geometric specifications (coordinates, angles, scales)
# - Visual vocabulary (keywords, directives)
# - Synthesis instructions for Claude
# - Cost profile showing 0 tokens used so far
```

## Balance Physics Details

### Torque Equilibrium
For a mobile to balance, the sum of torques around the suspension point must equal zero:

```
Σ(w_i × d_i) = 0

where:
  w_i = visual weight of element i
  d_i = distance from suspension point
```

### Asymmetric Counterbalance Example
```
Element A: weight = 0.7, distance = 0.3 → torque = 0.21
Element B: weight = 0.3, distance = 0.7 → torque = 0.21

Suspension point at 0.3 from left edge balances these unequal weights
```

## Visual Vocabulary Translation

### From Taxonomy to Image Keywords

**Balance Type** → **Compositional Keywords**
- `asymmetric_counterbalance` → "dynamic tension", "calculated imbalance", "unequal distribution achieving equilibrium"
- `hierarchical_suspension` → "tiered levels", "cascading elements", "parent-child relationships"

**Movement Type** → **Kinetic Behavior**
- `rotational` → "spinning around vertical axis", "reveals all perspectives sequentially", "360° potential range"
- `pendular` → "arc-based swinging", "gravity-driven return", "periodic oscillation"

**Material** → **Surface Characteristics**
- `industrial_metal` → "bold solid colors", "hard-edged geometric forms", "matte painted finish", "primary palette"
- `wire_armature` → "visible structural logic", "line-based composition", "minimal mass, maximum gesture"

## Geometric Precision Examples

### Suspension Point Specification
```
Suspension point at frame coordinates [0.5, 0.85] (center-top),
visible swivel_joint with wire armature
```

### Element Position Specification
```
Element 2: triangle at frame position [0.623, 0.628] (background layer),
scale factor 0.60, suspended at -15° from vertical,
wire length 0.48 units, rotated 60°, tilted -7.5°
```

### Spatial Envelope Specification
```
Spatial envelope: 100% frame width, 70% frame height,
50% depth layering, 60% negative space
```

## Available Tools

### Layer 1: Taxonomy
- `list_balance_types()` - All balance type specifications
- `list_movement_vocabularies()` - All movement types
- `list_material_surfaces()` - All material/surface options
- `get_balance_specifications(balance_id)` - Detailed specs for one type

### Layer 2: Deterministic Operations
- `calculate_balance_physics(weights, arm_length, balance_type)` - Torque equilibrium
- `map_prompt_to_mobile_parameters(prompt)` - Keyword extraction
- `generate_mobile_geometric_specs(...)` - Complete geometric specifications
- `extract_mobile_visual_vocabulary(geo_specs)` - Image-ready vocabulary

### Layer 3: Synthesis Interface
- `prepare_mobile_synthesis_context(prompt, ...)` - Complete context for Claude

### Utility
- `get_server_info()` - Server capabilities and architecture

## Integration with Lushy

This MCP server can be integrated into Lushy workflows for:

1. **ComfyUI Workflow Enhancement**: Inject mobile sculpture parameters into composition nodes
2. **Multi-Domain Composition**: Combine with other aesthetic MCPs (catastrophe morph, diatom morph, etc.)
3. **Systematic Validation**: Test framework confidence through empirical generation
4. **Academic Research**: Reproducible kinetic sculpture parameter spaces

## Cost Profile Comparison

### Pure LLM Approach
```
Prompt → LLM extracts parameters (200 tokens) →
LLM calculates balance (150 tokens) →
LLM generates specs (300 tokens) →
LLM synthesizes prompt (200 tokens)
Total: ~850 tokens
```

### Three-Layer Approach
```
Prompt → Deterministic extraction (0 tokens) →
Physics calculation (0 tokens) →
Geometric specs (0 tokens) →
LLM synthesizes prompt (150 tokens)
Total: ~150 tokens (82% savings)
```

## Design Philosophy

Following Dal's categorical composition principles:

1. **Taxonomy as Foundation**: Systematic knowledge representation before computation
2. **Deterministic Where Possible**: Physics, geometry, and keyword matching require no LLM
3. **Creative Synthesis Last**: LLM only for natural language assembly
4. **Geometric Precision**: Explicit coordinates over suggestive descriptions
5. **Neutral Description**: Compositional parameters described factually, not prescriptively

## Deployment

### FastMCP Cloud
```bash
# Entry point: main.py:mcp
# or: mobile_sculpture_mcp.py:mcp
```

### Local Development
```python
python mobile_sculpture_mcp.py
```

## Future Extensions

Potential additions maintaining the three-layer pattern:

- **Wind Dynamics**: Airflow simulation for realistic movement prediction
- **Material Physics**: Mass, inertia, and damping calculations
- **Multi-Tier Hierarchies**: Recursive balance calculations for complex mobiles
- **Temporal Sequencing**: Movement phase relationships over time
- **Light Interaction**: Shadow and reflection pattern mapping

## References

- Alexander Calder's mobile sculptures (asymmetric counterbalance aesthetic)
- Classical mechanics: torque equilibrium and center of mass
- Kinetic sculpture movement vocabularies
- Three-layer olog architecture (Dal Marsters / Lushy)

## License

MIT

## Author

Dal Marsters (Lushy.app)
