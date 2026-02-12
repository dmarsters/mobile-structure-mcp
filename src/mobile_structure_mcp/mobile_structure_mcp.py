"""
Mobile Sculpture MCP Server

Three-layer architecture for mobile sculpture visual vocabulary:
- Layer 1: Pure taxonomy (balance types, movement, materials)
- Layer 2: Deterministic operations (balance physics, spatial mapping)
- Layer 3: Creative synthesis interface

Based on kinetic sculpture principles, particularly Calder's mobiles.
"""

from fastmcp import FastMCP
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import math

mcp = FastMCP("Mobile Sculpture Aesthetics")

# ============================================================================
# LAYER 1: PURE TAXONOMY (0 tokens)
# ============================================================================

BALANCE_TYPES = {
    "asymmetric_counterbalance": {
        "name": "Asymmetric Counterbalance",
        "description": "Calder's signature - unequal weights at calculated distances",
        "characteristics": [
            "Dynamic tension through unequal distribution",
            "Requires precise suspension point calculation",
            "Visual interest from imbalance that achieves equilibrium",
            "Torque equation: W1 × D1 = W2 × D2"
        ],
        "visual_weight_ratios": [0.7, 0.3],  # typical ratios
        "typical_angles": [-15, 35],  # degrees from vertical
        "complexity": "high"
    },
    "symmetric_balance": {
        "name": "Symmetric Balance",
        "description": "Equal distribution around central axis",
        "characteristics": [
            "Static stability",
            "Formal composition",
            "Equal visual weights",
            "Predictable movement patterns"
        ],
        "visual_weight_ratios": [0.5, 0.5],
        "typical_angles": [-30, 30],
        "complexity": "low"
    },
    "cantilever_dynamic": {
        "name": "Cantilever Dynamics",
        "description": "Extended arms with distributed loads",
        "characteristics": [
            "Horizontal arm extends from vertical support",
            "Multiple elements along cantilever length",
            "Progressive weight distribution",
            "Tension between extension and stability"
        ],
        "visual_weight_ratios": [0.6, 0.25, 0.15],  # distributed
        "typical_angles": [0, 15, 30],  # relative to horizontal
        "complexity": "medium"
    },
    "hierarchical_suspension": {
        "name": "Hierarchical Suspension",
        "description": "Tiers of dependent elements",
        "characteristics": [
            "Cascading levels of suspension",
            "Parent-child element relationships",
            "Compound movement (each tier affects subordinates)",
            "Complex spatial depth"
        ],
        "visual_weight_ratios": [0.5, 0.3, 0.2],
        "typical_angles": [-20, 10, 25],
        "complexity": "very_high"
    }
}

MOVEMENT_VOCABULARIES = {
    "rotational": {
        "name": "Rotational Movement",
        "plane": "horizontal",
        "characteristics": [
            "Spinning around vertical axis",
            "360° potential range",
            "Centrifugal force at speed",
            "Reveals all perspectives sequentially"
        ],
        "typical_velocity": "slow_continuous",
        "visual_effect": "sequential_revelation"
    },
    "pendular": {
        "name": "Pendular Movement",
        "plane": "vertical",
        "characteristics": [
            "Arc-based swinging",
            "Periodic oscillation",
            "Gravity-driven return",
            "Predictable rhythm"
        ],
        "typical_velocity": "rhythmic_oscillation",
        "visual_effect": "temporal_breathing"
    },
    "oscillatory": {
        "name": "Oscillatory Movement",
        "plane": "variable",
        "characteristics": [
            "Back-and-forth within bounds",
            "Multiple axes simultaneously",
            "Bounded displacement",
            "Complex interference patterns"
        ],
        "typical_velocity": "variable_tempo",
        "visual_effect": "spatial_negotiation"
    },
    "cascading": {
        "name": "Cascading Movement",
        "plane": "hierarchical",
        "characteristics": [
            "Sequential activation through tiers",
            "Time-delayed response",
            "Parent motion amplified in children",
            "Wave-like propagation"
        ],
        "typical_velocity": "progressive_acceleration",
        "visual_effect": "temporal_unfolding"
    },
    "bobbing": {
        "name": "Bobbing Movement",
        "plane": "vertical",
        "characteristics": [
            "Vertical displacement only",
            "Spring-like behavior",
            "Minimal horizontal drift",
            "Buoyant quality"
        ],
        "typical_velocity": "gentle_pulse",
        "visual_effect": "weightless_suspension"
    }
}

MATERIAL_SURFACES = {
    "industrial_metal": {
        "name": "Industrial Metal (Calder)",
        "materials": ["painted steel", "aluminum", "brass"],
        "surface_finish": "matte_painted",
        "characteristics": [
            "Bold solid colors (primary palette)",
            "Hard-edged geometric forms",
            "Visible construction methods",
            "Industrial fabrication aesthetic"
        ],
        "reflectivity": "low",
        "color_palette": ["primary_red", "cobalt_blue", "cadmium_yellow", "matte_black"]
    },
    "biomorphic_organic": {
        "name": "Biomorphic Forms",
        "materials": ["curved metal", "resin", "wood"],
        "surface_finish": "smooth_organic",
        "characteristics": [
            "Flowing curved surfaces",
            "Natural form references",
            "Soft edges and transitions",
            "Growth-like asymmetry"
        ],
        "reflectivity": "medium",
        "color_palette": ["earth_tones", "natural_wood", "oxidized_patina"]
    },
    "wire_armature": {
        "name": "Wire Armature",
        "materials": ["steel wire", "copper wire", "aluminum rod"],
        "surface_finish": "linear_skeletal",
        "characteristics": [
            "Visible structural logic",
            "Line-based composition",
            "Transparency and overlap",
            "Minimal mass, maximum gesture"
        ],
        "reflectivity": "high_specular",
        "color_palette": ["metallic_silver", "oxidized_copper", "raw_steel"]
    },
    "reflective_kinetic": {
        "name": "Reflective Surfaces",
        "materials": ["polished metal", "mirror", "chrome"],
        "surface_finish": "specular_reflective",
        "characteristics": [
            "Environmental reflection",
            "Light multiplication",
            "Dematerialization through reflection",
            "Context-dependent appearance"
        ],
        "reflectivity": "very_high",
        "color_palette": ["chrome_silver", "mirror_finish", "polished_brass"]
    }
}

SPATIAL_RELATIONSHIPS = {
    "suspension_geometry": {
        "name": "Suspension Point Geometry",
        "parameters": [
            "attachment_angle (degrees from vertical)",
            "suspension_height (relative to frame)",
            "wire_visibility (exposed vs hidden)",
            "connection_method (direct, swivel, spring)"
        ],
        "typical_values": {
            "attachment_angle": 90,  # perpendicular to vertical
            "suspension_height": 0.85,  # 85% of frame height
            "wire_visibility": "visible",
            "connection_method": "swivel"
        }
    },
    "inter_element_clearance": {
        "name": "Inter-Element Clearance",
        "parameters": [
            "minimum_separation (prevents collision)",
            "visual_breathing_room (aesthetic spacing)",
            "movement_envelope (maximum displacement)",
            "overlap_zones (intentional intersections)"
        ],
        "typical_values": {
            "minimum_separation": 0.05,  # 5% of element size
            "visual_breathing_room": 0.15,  # 15% spacing
            "movement_envelope": 0.20,  # 20% swing radius
            "overlap_zones": 0.02  # 2% intentional overlap
        }
    },
    "depth_layering": {
        "name": "Depth Layering",
        "parameters": [
            "foreground_elements (front 33%)",
            "middle_ground (middle 33%)",
            "background_elements (rear 33%)",
            "depth_separation (z-axis spacing)"
        ],
        "typical_values": {
            "foreground_elements": 2,
            "middle_ground": 3,
            "background_elements": 2,
            "depth_separation": 0.25  # 25% of frame depth
        }
    },
    "visual_weight_distribution": {
        "name": "Visual Weight Distribution",
        "parameters": [
            "primary_anchor (dominant element)",
            "secondary_elements (supporting)",
            "terminal_points (endpoints)",
            "negative_space_ratio"
        ],
        "typical_values": {
            "primary_anchor": 0.45,  # 45% visual weight
            "secondary_elements": 0.35,  # 35% combined
            "terminal_points": 0.20,  # 20% total
            "negative_space_ratio": 0.60  # 60% empty space
        }
    }
}


# ============================================================================
# LAYER 2: DETERMINISTIC OPERATIONS (0 tokens)
# ============================================================================

@mcp.tool()
def list_balance_types() -> str:
    """
    List all available balance types for mobile sculptures.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        JSON string with balance type specifications
    """
    return json.dumps({
        "balance_types": BALANCE_TYPES,
        "count": len(BALANCE_TYPES),
        "usage": "Use get_balance_specifications(balance_id) for detailed specs"
    }, indent=2)


@mcp.tool()
def list_movement_vocabularies() -> str:
    """
    List all movement vocabularies for kinetic behavior.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        JSON string with movement vocabulary specifications
    """
    return json.dumps({
        "movement_vocabularies": MOVEMENT_VOCABULARIES,
        "count": len(MOVEMENT_VOCABULARIES),
        "usage": "Combine multiple movement types for complex kinetic behavior"
    }, indent=2)


@mcp.tool()
def list_material_surfaces() -> str:
    """
    List all material and surface finish options.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Returns:
        JSON string with material/surface specifications
    """
    return json.dumps({
        "material_surfaces": MATERIAL_SURFACES,
        "count": len(MATERIAL_SURFACES),
        "usage": "Material choice significantly affects movement and light interaction"
    }, indent=2)


@mcp.tool()
def get_balance_specifications(balance_id: str) -> str:
    """
    Get complete specifications for a specific balance type.
    
    Layer 1: Pure taxonomy lookup (0 tokens)
    
    Args:
        balance_id: One of the balance type IDs
        
    Returns:
        Complete balance specifications
    """
    if balance_id not in BALANCE_TYPES:
        return json.dumps({
            "error": f"Unknown balance type: {balance_id}",
            "available": list(BALANCE_TYPES.keys())
        }, indent=2)
    
    return json.dumps({
        "balance_id": balance_id,
        "specifications": BALANCE_TYPES[balance_id]
    }, indent=2)


@mcp.tool()
def calculate_balance_physics(
    element_weights: str,  # JSON list of visual weights [0.0-1.0]
    arm_length: float = 1.0,  # relative length of balance arm
    balance_type: str = "asymmetric_counterbalance"
) -> str:
    """
    Calculate suspension point and element positions for equilibrium.
    
    Layer 2: Deterministic physics calculation (0 tokens)
    
    Uses torque equilibrium: Σ(weight × distance) = 0 around suspension point
    
    Args:
        element_weights: JSON list of relative visual weights (must sum to 1.0)
        arm_length: Total length of balance arm (default 1.0)
        balance_type: Which balance type to use
        
    Returns:
        JSON with suspension point, element positions, and angles
    """
    weights = json.loads(element_weights)
    
    # Validate weights sum to 1.0
    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        return json.dumps({
            "error": f"Weights must sum to 1.0, got {total}",
            "weights": weights
        }, indent=2)
    
    balance_spec = BALANCE_TYPES.get(balance_type, BALANCE_TYPES["asymmetric_counterbalance"])
    typical_angles = balance_spec["typical_angles"]
    
    # Calculate suspension point using torque equilibrium
    # For n elements, find point where Σ(w_i × d_i) = 0
    
    n = len(weights)
    positions = []
    
    if balance_type == "symmetric_balance":
        # Equal spacing, symmetric around center
        for i in range(n):
            pos = (i - (n-1)/2) / n * arm_length
            positions.append(pos)
        suspension_point = 0.0
        
    elif balance_type == "cantilever_dynamic":
        # Elements distributed along horizontal cantilever
        # Suspension at one end, elements progressively along length
        positions = [i / (n-1) * arm_length for i in range(n)]
        # Calculate center of mass for suspension point
        center_of_mass = sum(w * p for w, p in zip(weights, positions)) / sum(weights)
        suspension_point = center_of_mass
        
    else:  # asymmetric_counterbalance and hierarchical_suspension
        # Use typical angles to distribute elements
        angles_rad = [math.radians(typical_angles[i % len(typical_angles)]) for i in range(n)]
        # Project onto horizontal for balance calculation
        positions = [math.sin(angle) * arm_length * (i+1) / n for i, angle in enumerate(angles_rad)]
        # Find suspension point where torques balance
        center_of_mass = sum(w * p for w, p in zip(weights, positions)) / sum(weights)
        suspension_point = center_of_mass
    
    # Calculate angles from vertical for each element
    element_specs = []
    for i, (weight, pos) in enumerate(zip(weights, positions)):
        # Distance from suspension point
        distance = abs(pos - suspension_point)
        
        # Angle from vertical (use typical angles as base)
        base_angle = typical_angles[i % len(typical_angles)]
        
        # Calculate actual angle based on position relative to suspension
        if pos > suspension_point:
            angle = abs(base_angle)
        else:
            angle = -abs(base_angle)
        
        element_specs.append({
            "element_index": i,
            "visual_weight": weight,
            "position_along_arm": pos,
            "distance_from_suspension": distance,
            "angle_from_vertical": angle,
            "torque": weight * distance
        })
    
    # Verify equilibrium
    total_torque = sum(spec["torque"] * (1 if spec["position_along_arm"] > suspension_point else -1) 
                       for spec in element_specs)
    
    return json.dumps({
        "balance_type": balance_type,
        "arm_length": arm_length,
        "suspension_point": suspension_point,
        "suspension_point_normalized": suspension_point / arm_length,  # 0.0-1.0
        "element_specifications": element_specs,
        "total_torque": total_torque,
        "equilibrium_achieved": abs(total_torque) < 0.01,
        "methodology": "torque_equilibrium"
    }, indent=2)


@mcp.tool()
def map_prompt_to_mobile_parameters(prompt: str) -> str:
    """
    Extract mobile sculpture parameters from natural language prompt.
    
    Layer 2: Deterministic keyword extraction (0 tokens)
    
    Analyzes prompt for:
    - Balance type indicators
    - Movement suggestions
    - Material references
    - Compositional terms
    
    Args:
        prompt: Natural language description
        
    Returns:
        JSON with detected parameters
    """
    prompt_lower = prompt.lower()
    
    # Detect balance type
    detected_balance = None
    balance_keywords = {
        "asymmetric_counterbalance": ["asymmetric", "unequal", "counterbalance", "tension", "dynamic"],
        "symmetric_balance": ["symmetric", "equal", "balanced", "formal", "centered"],
        "cantilever_dynamic": ["cantilever", "extended", "horizontal", "distributed"],
        "hierarchical_suspension": ["hierarchical", "tiered", "cascading", "levels", "parent"]
    }
    
    for balance_id, keywords in balance_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            detected_balance = balance_id
            break
    
    # Detect movement types
    detected_movements = []
    movement_keywords = {
        "rotational": ["spin", "rotate", "revolve", "circular"],
        "pendular": ["swing", "pendulum", "arc", "sway"],
        "oscillatory": ["oscillate", "vibrate", "flutter", "shimmer"],
        "cascading": ["cascade", "wave", "ripple", "sequential"],
        "bobbing": ["bob", "bounce", "float", "buoyant"]
    }
    
    for movement_id, keywords in movement_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            detected_movements.append(movement_id)
    
    # Detect materials
    detected_material = None
    material_keywords = {
        "industrial_metal": ["metal", "steel", "painted", "industrial", "calder"],
        "biomorphic_organic": ["organic", "biomorphic", "curved", "flowing", "natural"],
        "wire_armature": ["wire", "linear", "skeletal", "transparent"],
        "reflective_kinetic": ["reflective", "mirror", "chrome", "polished", "shiny"]
    }
    
    for material_id, keywords in material_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            detected_material = material_id
            break
    
    # Count element references
    element_indicators = ["element", "piece", "component", "shape", "form", "disc", "sphere", "rectangle"]
    element_count = sum(prompt_lower.count(ind) for ind in element_indicators)
    element_count = max(3, min(7, element_count))  # clamp to 3-7
    
    return json.dumps({
        "detected_balance_type": detected_balance or "asymmetric_counterbalance",
        "detected_movements": detected_movements or ["rotational", "pendular"],
        "detected_material": detected_material or "industrial_metal",
        "estimated_element_count": element_count,
        "confidence": "deterministic_keyword_matching",
        "prompt_analyzed": prompt
    }, indent=2)


@mcp.tool()
def generate_mobile_geometric_specs(
    balance_type: str,
    element_count: int = 5,
    movement_types: str = '["rotational", "pendular"]',  # JSON list
    material_surface: str = "industrial_metal"
) -> str:
    """
    Generate explicit geometric specifications for image generation.
    
    Layer 2: Deterministic composition (0 tokens)
    
    Creates precise spatial coordinates, angles, and visual weight distributions
    following Dal's preference for explicit geometric specifications.
    
    Args:
        balance_type: Which balance type to use
        element_count: Number of elements (3-7 recommended)
        movement_types: JSON list of movement types to combine
        material_surface: Material/surface finish
        
    Returns:
        JSON with explicit geometric specifications ready for image prompt
    """
    movements = json.loads(movement_types)
    
    # Generate visual weights (sum to 1.0)
    # Use fibonacci-like distribution for natural appearance
    weights = []
    remaining = 1.0
    for i in range(element_count):
        if i < element_count - 1:
            weight = remaining / (2 ** (i + 1))
            weights.append(weight)
            remaining -= weight
        else:
            weights.append(remaining)
    
    # Calculate balance physics
    balance_result = json.loads(calculate_balance_physics(
        element_weights=json.dumps(weights),
        arm_length=1.0,
        balance_type=balance_type
    ))
    
    if "error" in balance_result:
        return json.dumps(balance_result, indent=2)
    
    # Get material specs
    material_spec = MATERIAL_SURFACES[material_surface]
    
    # Get movement specs
    movement_specs = [MOVEMENT_VOCABULARIES[m] for m in movements if m in MOVEMENT_VOCABULARIES]
    
    # Construct explicit geometric specifications
    geometric_specs = {
        "composition_type": "mobile_sculpture",
        "balance_configuration": {
            "type": balance_type,
            "suspension_point": {
                "position": "frame_center_top",
                "normalized_coordinates": [0.5, 0.85],  # x, y in frame
                "wire_visibility": "visible",
                "connection_type": "swivel_joint"
            },
            "equilibrium_achieved": balance_result["equilibrium_achieved"]
        },
        "element_specifications": [],
        "movement_characteristics": movement_specs,
        "material_specifications": material_spec,
        "spatial_envelope": {
            "width": 1.0,  # normalized frame width
            "height": 0.7,  # 70% of frame height
            "depth": 0.5,   # depth layering
            "negative_space_ratio": 0.60
        }
    }
    
    # Add detailed element specifications
    for elem in balance_result["element_specifications"]:
        # Calculate frame position
        # Suspension point is at [0.5, 0.85]
        # Elements hang below, distributed by angle and distance
        
        angle_rad = math.radians(elem["angle_from_vertical"])
        distance = elem["distance_from_suspension"]
        
        # Project position in frame
        x_offset = math.sin(angle_rad) * distance * 0.3  # scale to frame
        y_offset = -math.cos(angle_rad) * distance * 0.4  # negative = down
        
        frame_x = 0.5 + x_offset
        frame_y = 0.85 + y_offset
        
        # Determine depth layer (alternate for visual interest)
        depth_layer = "foreground" if elem["element_index"] % 3 == 0 else \
                     "middle_ground" if elem["element_index"] % 3 == 1 else \
                     "background"
        
        element_spec = {
            "element_id": elem["element_index"],
            "visual_weight": elem["visual_weight"],
            "frame_position": {
                "x": round(frame_x, 3),
                "y": round(frame_y, 3),
                "depth_layer": depth_layer
            },
            "suspension_geometry": {
                "angle_from_vertical": elem["angle_from_vertical"],
                "distance_from_suspension": elem["distance_from_suspension"],
                "wire_length": elem["distance_from_suspension"] * 1.2  # wire is longer than straight-line distance
            },
            "geometric_form": "disc" if elem["element_index"] % 2 == 0 else "triangle",
            "scale_factor": elem["visual_weight"] * 2.0,  # larger weights = larger elements
            "orientation": {
                "rotation": elem["element_index"] * 30,  # degrees, staggered
                "tilt": elem["angle_from_vertical"] / 2  # slight tilt echoing suspension
            }
        }
        
        geometric_specs["element_specifications"].append(element_spec)
    
    return json.dumps({
        "geometric_specifications": geometric_specs,
        "usage": "These specs provide explicit coordinates, angles, and scales for image generation",
        "prompt_translation_needed": "Convert to natural language with preserved precision"
    }, indent=2)


@mcp.tool()
def extract_mobile_visual_vocabulary(geometric_specs: str) -> str:
    """
    Extract image-generation-ready vocabulary from geometric specs.
    
    Layer 2: Deterministic vocabulary mapping (0 tokens)
    
    Translates geometric specifications into visual keywords and explicit
    compositional directives suitable for text-to-image models.
    
    Args:
        geometric_specs: JSON output from generate_mobile_geometric_specs
        
    Returns:
        Image generation vocabulary with preserved geometric precision
    """
    specs = json.loads(geometric_specs)
    geo = specs["geometric_specifications"]
    
    # Extract core vocabulary
    balance_type = geo["balance_configuration"]["type"]
    material = geo["material_specifications"]
    movements = geo["movement_characteristics"]
    
    # Build vocabulary lists
    compositional_keywords = []
    geometric_directives = []
    material_keywords = material["characteristics"]
    movement_keywords = []
    
    # Compositional keywords from balance type
    balance_vocab = {
        "asymmetric_counterbalance": ["dynamic tension", "calculated imbalance", "unequal distribution achieving equilibrium"],
        "symmetric_balance": ["formal symmetry", "centered composition", "equal distribution"],
        "cantilever_dynamic": ["horizontal extension", "distributed load", "progressive arrangement"],
        "hierarchical_suspension": ["tiered levels", "cascading elements", "parent-child relationships"]
    }
    compositional_keywords.extend(balance_vocab.get(balance_type, []))
    
    # Geometric directives from element specs
    suspension = geo["balance_configuration"]["suspension_point"]
    geometric_directives.append(
        f"Suspension point at frame coordinates [{suspension['normalized_coordinates'][0]}, {suspension['normalized_coordinates'][1]}] (center-top)"
    )
    geometric_directives.append(
        f"Visible {suspension['connection_type']} with wire armature"
    )
    
    # Element-specific geometric directives
    for elem in geo["element_specifications"]:
        pos = elem["frame_position"]
        susp = elem["suspension_geometry"]
        
        geometric_directives.append(
            f"Element {elem['element_id']}: {elem['geometric_form']} at frame position "
            f"[{pos['x']}, {pos['y']}] ({pos['depth_layer']} layer), "
            f"scale factor {elem['scale_factor']:.2f}, "
            f"suspended at {susp['angle_from_vertical']}° from vertical, "
            f"wire length {susp['wire_length']:.2f} units, "
            f"rotated {elem['orientation']['rotation']}°, "
            f"tilted {elem['orientation']['tilt']:.1f}°"
        )
    
    # Movement vocabulary
    for movement in movements:
        movement_keywords.extend(movement["characteristics"])
    
    # Spatial envelope
    envelope = geo["spatial_envelope"]
    geometric_directives.append(
        f"Spatial envelope: {envelope['width']*100}% frame width, "
        f"{envelope['height']*100}% frame height, "
        f"{envelope['depth']*100}% depth layering, "
        f"{envelope['negative_space_ratio']*100}% negative space"
    )
    
    # Color palette from material
    color_vocab = material.get("color_palette", [])
    
    return json.dumps({
        "visual_vocabulary": {
            "compositional_keywords": compositional_keywords,
            "geometric_directives": geometric_directives,
            "material_surface_keywords": material_keywords,
            "movement_behavior_keywords": movement_keywords,
            "color_palette": color_vocab,
            "surface_finish": material["surface_finish"],
            "reflectivity": material["reflectivity"]
        },
        "prompt_assembly_guidelines": {
            "preserve_geometric_precision": "Maintain exact coordinates, angles, and scales",
            "explicit_spatial_relationships": "State element positions relative to suspension point",
            "material_characteristics_first": "Establish surface and material before movement",
            "movement_as_potential": "Describe kinetic capacity, not frozen motion"
        },
        "usage": "Combine vocabularies into coherent image prompt preserving geometric specifications"
    }, indent=2)


# ============================================================================
# LAYER 3 INTERFACE: Creative Synthesis Preparation
# ============================================================================

@mcp.tool()
def prepare_mobile_synthesis_context(
    prompt: str,
    explicit_balance_type: str = "",
    explicit_element_count: int = 0,
    explicit_movements: str = "",
    explicit_material: str = ""
) -> str:
    """
    Prepare complete context for Claude to synthesize image prompt.
    
    Layer 3 Interface: Assembles deterministic parameters for creative synthesis
    
    Workflow:
    1. Extract parameters from prompt (Layer 2)
    2. Override with explicit parameters if provided
    3. Generate geometric specifications (Layer 2)
    4. Extract visual vocabulary (Layer 2)
    5. Package everything for Claude to synthesize natural language prompt
    
    Args:
        prompt: User's natural language request
        explicit_balance_type: Override detected balance type (optional)
        explicit_element_count: Override estimated element count (optional)
        explicit_movements: Override detected movements as JSON list (optional)
        explicit_material: Override detected material (optional)
        
    Returns:
        Complete synthesis context with all deterministic parameters assembled
    """
    # Step 1: Extract from prompt
    extracted = json.loads(map_prompt_to_mobile_parameters(prompt))
    
    # Step 2: Apply overrides
    balance_type = explicit_balance_type or extracted["detected_balance_type"]
    element_count = explicit_element_count or extracted["estimated_element_count"]
    movements = explicit_movements or json.dumps(extracted["detected_movements"])
    material = explicit_material or extracted["detected_material"]
    
    # Step 3: Generate geometric specs
    geo_specs = generate_mobile_geometric_specs(
        balance_type=balance_type,
        element_count=element_count,
        movement_types=movements,
        material_surface=material
    )
    
    # Step 4: Extract vocabulary
    vocab = extract_mobile_visual_vocabulary(geo_specs)
    
    # Step 5: Assemble synthesis context
    synthesis_context = {
        "original_prompt": prompt,
        "parameters_used": {
            "balance_type": balance_type,
            "element_count": element_count,
            "movements": json.loads(movements),
            "material": material
        },
        "geometric_specifications": json.loads(geo_specs),
        "visual_vocabulary": json.loads(vocab),
        "synthesis_instructions": {
            "task": "Translate geometric specifications and vocabulary into natural language image prompt",
            "requirements": [
                "PRESERVE all geometric coordinates, angles, and scales exactly",
                "DESCRIBE compositional parameters neutrally (never prescribe success/failure)",
                "TRANSLATE taxonomy to image generation vocabulary",
                "STATE explicit geometric relationships (e.g., 'Element at 35° angle, frame position [0.65, 0.72]')",
                "COMBINE movement characteristics as kinetic potential, not frozen action",
                "EMPHASIZE material surface qualities and light interaction"
            ],
            "format": "Single coherent paragraph with embedded geometric specifications",
            "style": "Technical precision meeting artistic description"
        },
        "cost_profile": {
            "layer_1_taxonomy_lookup": "0 tokens",
            "layer_2_deterministic_operations": "0 tokens",
            "layer_3_synthesis": "~150-200 tokens (Claude synthesis)",
            "total_llm_cost": "Single synthesis call only"
        }
    }
    
    return json.dumps(synthesis_context, indent=2)


# ============================================================================
# PHASE 2.6: RHYTHMIC COMPOSITION (0 tokens)
# ============================================================================
# Normalized 5D parameter space for mobile sculpture aesthetics.
# Each dimension captures a continuous aesthetic axis derived from
# the Layer 1 taxonomy (balance, movement, material, spatial).

MOBILE_PARAMETER_NAMES = [
    "balance_asymmetry",     # 0.0 = perfect symmetric, 1.0 = extreme asymmetric
    "kinetic_energy",        # 0.0 = still/frozen, 1.0 = maximum movement
    "structural_density",    # 0.0 = minimal wire/skeletal, 1.0 = dense heavy forms
    "surface_reflectivity",  # 0.0 = matte painted, 1.0 = mirror/chrome
    "hierarchical_depth"     # 0.0 = single tier flat, 1.0 = deep cascading tiers
]

MOBILE_CANONICAL_STATES = {
    "calder_classic": {
        "balance_asymmetry": 0.80,
        "kinetic_energy": 0.55,
        "structural_density": 0.60,
        "surface_reflectivity": 0.15,
        "hierarchical_depth": 0.65,
        "description": "Calder-style painted steel mobile, asymmetric counterbalance, "
                       "moderate rotation, bold primary colors, multi-tier"
    },
    "minimal_wire": {
        "balance_asymmetry": 0.45,
        "kinetic_energy": 0.35,
        "structural_density": 0.10,
        "surface_reflectivity": 0.70,
        "hierarchical_depth": 0.25,
        "description": "Sparse wire armature, gentle oscillation, exposed linear structure, "
                       "metallic gleam, single or two tiers"
    },
    "reflective_cascade": {
        "balance_asymmetry": 0.60,
        "kinetic_energy": 0.75,
        "structural_density": 0.50,
        "surface_reflectivity": 0.95,
        "hierarchical_depth": 0.90,
        "description": "Chrome/mirror surfaces, deep cascading tiers, vigorous movement, "
                       "environmental reflections, light multiplication"
    },
    "biomorphic_pendulum": {
        "balance_asymmetry": 0.55,
        "kinetic_energy": 0.65,
        "structural_density": 0.45,
        "surface_reflectivity": 0.30,
        "hierarchical_depth": 0.40,
        "description": "Organic curved forms, pendular swinging, natural materials, "
                       "earth tones, growth-like asymmetry"
    },
    "industrial_cantilever": {
        "balance_asymmetry": 0.35,
        "kinetic_energy": 0.50,
        "structural_density": 0.85,
        "surface_reflectivity": 0.20,
        "hierarchical_depth": 0.30,
        "description": "Heavy painted steel, horizontal cantilever extension, "
                       "distributed load, industrial fabrication, bold flat planes"
    },
    "floating_symmetric": {
        "balance_asymmetry": 0.10,
        "kinetic_energy": 0.25,
        "structural_density": 0.30,
        "surface_reflectivity": 0.55,
        "hierarchical_depth": 0.15,
        "description": "Gentle bobbing, near-perfect symmetry, light translucent materials, "
                       "weightless suspension quality, minimal tiers"
    }
}

# Phase 2.6 Rhythmic Presets — oscillation patterns between canonical states
MOBILE_RHYTHMIC_PRESETS = {
    "balance_shift": {
        "state_a": "calder_classic",
        "state_b": "minimal_wire",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 20,
        "description": "Smooth transition between bold asymmetric Calder forms and "
                       "transparent wire minimalism. Explores the tension between "
                       "visual weight and structural transparency."
    },
    "material_cycle": {
        "state_a": "industrial_cantilever",
        "state_b": "reflective_cascade",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 24,
        "description": "Cycles between matte industrial heaviness and mirror-surface "
                       "lightness. Material surface transforms from absorbing to "
                       "reflecting, density from heavy to moderate."
    },
    "motion_sweep": {
        "state_a": "biomorphic_pendulum",
        "state_b": "floating_symmetric",
        "pattern": "triangular",
        "num_cycles": 4,
        "steps_per_cycle": 18,
        "description": "Linear ramp between organic pendular energy and still "
                       "symmetric floating. Kinetic energy decays while symmetry "
                       "increases, then reverses."
    },
    "hierarchy_pulse": {
        "state_a": "minimal_wire",
        "state_b": "reflective_cascade",
        "pattern": "sinusoidal",
        "num_cycles": 2,
        "steps_per_cycle": 28,
        "description": "Slow deep oscillation between flat sparse structures and "
                       "deeply cascading reflective tiers. Hierarchical depth "
                       "breathes from minimal to maximal."
    },
    "kinetic_toggle": {
        "state_a": "calder_classic",
        "state_b": "floating_symmetric",
        "pattern": "square",
        "num_cycles": 5,
        "steps_per_cycle": 14,
        "description": "Sharp switch between dynamic Calder asymmetry and still "
                       "symmetric floating. Abrupt contrast in balance philosophy "
                       "and kinetic energy."
    }
}


def _generate_mobile_oscillation(
    num_steps: int,
    num_cycles: float,
    pattern: str
) -> np.ndarray:
    """Generate oscillation blending factor array [0.0, 1.0]."""
    t = np.linspace(0, 2 * np.pi * num_cycles, num_steps, endpoint=False)

    if pattern == "sinusoidal":
        return 0.5 * (1.0 + np.sin(t))
    elif pattern == "triangular":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 2.0 * t_norm, 2.0 * (1.0 - t_norm))
    elif pattern == "square":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown oscillation pattern: {pattern}")


def _generate_mobile_preset_trajectory(preset_config: dict) -> List[dict]:
    """
    Generate a full preset trajectory as list of state dicts.

    Each state dict maps MOBILE_PARAMETER_NAMES → float value.
    The trajectory forms a closed loop (endpoint ≈ start) when
    num_cycles is an integer.
    """
    state_a = MOBILE_CANONICAL_STATES[preset_config["state_a"]]
    state_b = MOBILE_CANONICAL_STATES[preset_config["state_b"]]

    total_steps = preset_config["num_cycles"] * preset_config["steps_per_cycle"]
    alpha = _generate_mobile_oscillation(
        total_steps, preset_config["num_cycles"], preset_config["pattern"]
    )

    vec_a = np.array([state_a[p] for p in MOBILE_PARAMETER_NAMES])
    vec_b = np.array([state_b[p] for p in MOBILE_PARAMETER_NAMES])

    trajectory_array = np.outer(1.0 - alpha, vec_a) + np.outer(alpha, vec_b)

    trajectory = []
    for row in trajectory_array:
        state = {p: float(row[i]) for i, p in enumerate(MOBILE_PARAMETER_NAMES)}
        trajectory.append(state)

    return trajectory


@mcp.tool()
def list_mobile_rhythmic_presets() -> str:
    """
    List all Phase 2.6 rhythmic presets for mobile sculpture.

    Layer 2: Pure taxonomy lookup (0 tokens)

    Each preset defines a temporal oscillation between two canonical
    mobile sculpture states, creating a rhythmic aesthetic pattern.

    Returns:
        JSON with preset names, descriptions, periods, and patterns
    """
    presets = {}
    for name, config in MOBILE_RHYTHMIC_PRESETS.items():
        presets[name] = {
            "period": config["steps_per_cycle"],
            "total_steps": config["num_cycles"] * config["steps_per_cycle"],
            "pattern": config["pattern"],
            "state_a": config["state_a"],
            "state_b": config["state_b"],
            "description": config["description"]
        }

    return json.dumps({
        "presets": presets,
        "count": len(presets),
        "parameter_space": MOBILE_PARAMETER_NAMES,
        "canonical_states": list(MOBILE_CANONICAL_STATES.keys()),
        "usage": "Use apply_mobile_rhythmic_preset(preset_name) to get trajectory"
    }, indent=2)


@mcp.tool()
def list_mobile_canonical_states() -> str:
    """
    List all canonical states in mobile sculpture parameter space.

    Layer 1: Pure taxonomy lookup (0 tokens)

    Returns:
        JSON with state coordinates and descriptions
    """
    states = {}
    for name, state in MOBILE_CANONICAL_STATES.items():
        states[name] = {
            p: state[p] for p in MOBILE_PARAMETER_NAMES
        }
        states[name]["description"] = state["description"]

    return json.dumps({
        "canonical_states": states,
        "parameter_names": MOBILE_PARAMETER_NAMES,
        "count": len(states)
    }, indent=2)


@mcp.tool()
def apply_mobile_rhythmic_preset(preset_name: str) -> str:
    """
    Apply a Phase 2.6 rhythmic preset and return the full trajectory.

    Layer 2: Deterministic trajectory generation (0 tokens)

    Generates the complete oscillation trajectory between two canonical
    states. The trajectory is a list of parameter-space states suitable
    for forced orbit integration or direct keyframe extraction.

    Args:
        preset_name: One of the available rhythmic presets

    Returns:
        JSON with trajectory states, pattern metadata, and period info
    """
    if preset_name not in MOBILE_RHYTHMIC_PRESETS:
        return json.dumps({
            "error": f"Unknown preset: {preset_name}",
            "available": list(MOBILE_RHYTHMIC_PRESETS.keys())
        }, indent=2)

    config = MOBILE_RHYTHMIC_PRESETS[preset_name]
    trajectory = _generate_mobile_preset_trajectory(config)

    return json.dumps({
        "preset_name": preset_name,
        "period": config["steps_per_cycle"],
        "total_steps": len(trajectory),
        "pattern": config["pattern"],
        "state_a": config["state_a"],
        "state_b": config["state_b"],
        "description": config["description"],
        "trajectory": trajectory,
        "parameter_names": MOBILE_PARAMETER_NAMES,
        "usage": "Each trajectory state maps to a complete mobile sculpture configuration"
    }, indent=2)


@mcp.tool()
def generate_mobile_rhythmic_sequence(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 3,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0
) -> str:
    """
    Generate custom rhythmic oscillation between two mobile sculpture states.

    Phase 2.6 Tool: Temporal composition for mobile sculpture aesthetics.
    Creates periodic transitions cycling between canonical states.

    Args:
        state_a_id: Starting canonical state
        state_b_id: Alternating canonical state
        oscillation_pattern: "sinusoidal" | "triangular" | "square"
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle (= period)
        phase_offset: Starting phase (0.0 = A, 0.5 = B)

    Returns:
        Sequence with states, pattern info, and phase points

    Cost: 0 tokens (Layer 2 deterministic)
    """
    if state_a_id not in MOBILE_CANONICAL_STATES:
        return json.dumps({
            "error": f"Unknown state: {state_a_id}",
            "available": list(MOBILE_CANONICAL_STATES.keys())
        }, indent=2)
    if state_b_id not in MOBILE_CANONICAL_STATES:
        return json.dumps({
            "error": f"Unknown state: {state_b_id}",
            "available": list(MOBILE_CANONICAL_STATES.keys())
        }, indent=2)

    config = {
        "state_a": state_a_id,
        "state_b": state_b_id,
        "pattern": oscillation_pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle
    }

    total_steps = num_cycles * steps_per_cycle
    t = np.linspace(0, 2 * np.pi * num_cycles, total_steps, endpoint=False)
    t = t + phase_offset * 2 * np.pi  # apply phase offset

    state_a = MOBILE_CANONICAL_STATES[state_a_id]
    state_b = MOBILE_CANONICAL_STATES[state_b_id]
    vec_a = np.array([state_a[p] for p in MOBILE_PARAMETER_NAMES])
    vec_b = np.array([state_b[p] for p in MOBILE_PARAMETER_NAMES])

    if oscillation_pattern == "sinusoidal":
        alpha = 0.5 * (1.0 + np.sin(t))
    elif oscillation_pattern == "triangular":
        t_norm = (t / (2 * np.pi)) % 1.0
        alpha = np.where(t_norm < 0.5, 2.0 * t_norm, 2.0 * (1.0 - t_norm))
    elif oscillation_pattern == "square":
        t_norm = (t / (2 * np.pi)) % 1.0
        alpha = np.where(t_norm < 0.5, 0.0, 1.0)
    else:
        return json.dumps({"error": f"Unknown pattern: {oscillation_pattern}"}, indent=2)

    trajectory_array = np.outer(1.0 - alpha, vec_a) + np.outer(alpha, vec_b)
    trajectory = []
    phase_values = []
    for i, row in enumerate(trajectory_array):
        state = {p: round(float(row[j]), 4) for j, p in enumerate(MOBILE_PARAMETER_NAMES)}
        trajectory.append(state)
        phase_values.append(round(float(alpha[i]), 4))

    return json.dumps({
        "state_a": state_a_id,
        "state_b": state_b_id,
        "pattern": oscillation_pattern,
        "period": steps_per_cycle,
        "num_cycles": num_cycles,
        "total_steps": total_steps,
        "phase_offset": phase_offset,
        "trajectory": trajectory,
        "phase_values": phase_values,
        "parameter_names": MOBILE_PARAMETER_NAMES
    }, indent=2)


# ============================================================================
# PHASE 2.7: ATTRACTOR VISUALIZATION PROMPT GENERATION (0 tokens)
# ============================================================================
# Maps parameter-space coordinates to image-generation-ready visual
# vocabulary through nearest-neighbor matching against canonical
# visual types derived from the mobile sculpture taxonomy.

MOBILE_VISUAL_TYPES = {
    "calder_industrial": {
        "coords": {
            "balance_asymmetry": 0.80,
            "kinetic_energy": 0.55,
            "structural_density": 0.65,
            "surface_reflectivity": 0.15,
            "hierarchical_depth": 0.60
        },
        "keywords": [
            "painted steel mobile sculpture",
            "bold primary colors on matte surfaces",
            "asymmetric counterbalanced arms",
            "visible wire armature connections",
            "dynamic tension between unequal forms",
            "industrial fabrication aesthetic",
            "suspended geometric shapes"
        ],
        "optical_properties": {
            "finish": "matte_painted",
            "light_interaction": "diffuse absorption",
            "dominant_colors": ["primary red", "cobalt blue", "cadmium yellow", "matte black"]
        }
    },
    "wire_skeletal": {
        "coords": {
            "balance_asymmetry": 0.45,
            "kinetic_energy": 0.35,
            "structural_density": 0.10,
            "surface_reflectivity": 0.70,
            "hierarchical_depth": 0.25
        },
        "keywords": [
            "wire armature sculpture",
            "transparent skeletal structure",
            "line-based spatial drawing",
            "minimal mass maximum gesture",
            "metallic gleam on thin rods",
            "negative space dominates form",
            "delicate equilibrium"
        ],
        "optical_properties": {
            "finish": "bare_metallic",
            "light_interaction": "specular highlights on thin edges",
            "dominant_colors": ["silver wire", "oxidized copper", "raw steel gray"]
        }
    },
    "mirror_cascade": {
        "coords": {
            "balance_asymmetry": 0.60,
            "kinetic_energy": 0.75,
            "structural_density": 0.50,
            "surface_reflectivity": 0.95,
            "hierarchical_depth": 0.90
        },
        "keywords": [
            "polished chrome mobile",
            "cascading reflective tiers",
            "environmental reflections on curved surfaces",
            "light multiplication through mirror planes",
            "kinetic energy rippling through levels",
            "dematerialized form through reflection",
            "deep hierarchical suspension"
        ],
        "optical_properties": {
            "finish": "mirror_polished",
            "light_interaction": "full environmental reflection",
            "dominant_colors": ["chrome silver", "reflected environment", "bright caustics"]
        }
    },
    "biomorphic_organic": {
        "coords": {
            "balance_asymmetry": 0.55,
            "kinetic_energy": 0.60,
            "structural_density": 0.45,
            "surface_reflectivity": 0.25,
            "hierarchical_depth": 0.40
        },
        "keywords": [
            "organic curved mobile forms",
            "biomorphic shapes in natural materials",
            "pendular swinging arcs",
            "growth-like asymmetry",
            "smooth flowing surfaces",
            "earth-toned resin and wood",
            "natural form references"
        ],
        "optical_properties": {
            "finish": "smooth_organic",
            "light_interaction": "warm diffuse with subtle grain",
            "dominant_colors": ["warm wood tones", "oxidized patina", "matte earth"]
        }
    }
}


def _extract_mobile_visual_vocabulary(
    state: dict,
    strength: float = 1.0
) -> dict:
    """
    Map a 5D parameter state to nearest canonical mobile visual type.

    Pure Layer 2 deterministic operation (0 tokens).

    Args:
        state: Dict with MOBILE_PARAMETER_NAMES keys → float values
        strength: Keyword weight multiplier [0.0, 1.0]

    Returns:
        Dict with nearest_type, distance, keywords, optical_properties
    """
    state_vec = np.array([state.get(p, 0.5) for p in MOBILE_PARAMETER_NAMES])

    best_type = None
    best_distance = float("inf")

    for type_name, type_spec in MOBILE_VISUAL_TYPES.items():
        type_vec = np.array([type_spec["coords"][p] for p in MOBILE_PARAMETER_NAMES])
        distance = float(np.linalg.norm(state_vec - type_vec))
        if distance < best_distance:
            best_distance = distance
            best_type = type_name

    matched = MOBILE_VISUAL_TYPES[best_type]

    # Weight keywords by strength
    if strength < 1.0:
        n_keywords = max(2, int(len(matched["keywords"]) * strength))
        keywords = matched["keywords"][:n_keywords]
    else:
        keywords = list(matched["keywords"])

    return {
        "nearest_type": best_type,
        "distance": round(best_distance, 4),
        "keywords": keywords,
        "optical_properties": matched["optical_properties"],
        "state_used": {p: round(state.get(p, 0.5), 4) for p in MOBILE_PARAMETER_NAMES}
    }


@mcp.tool()
def extract_mobile_attractor_vocabulary(
    state: str,
    strength: float = 1.0
) -> str:
    """
    Extract visual vocabulary from mobile sculpture parameter coordinates.

    Phase 2.7 Tool: Maps a 5D parameter state to the nearest canonical
    mobile visual type and returns image-generation-ready keywords.

    Uses nearest-neighbor matching against 4 visual types derived from
    the mobile sculpture taxonomy.

    Args:
        state: JSON dict with parameter coordinates (balance_asymmetry,
               kinetic_energy, structural_density, surface_reflectivity,
               hierarchical_depth)
        strength: Keyword weight multiplier [0.0, 1.0] (default 1.0)

    Returns:
        JSON with nearest_type, distance, keywords, optical_properties

    Cost: 0 tokens (pure Layer 2 computation)
    """
    state_dict = json.loads(state) if isinstance(state, str) else state
    result = _extract_mobile_visual_vocabulary(state_dict, strength)
    return json.dumps(result, indent=2)


# Preset attractors discovered / curated for mobile sculpture domain
MOBILE_ATTRACTOR_PRESETS = {
    "balance_shift_attractor": {
        "name": "Balance Shift — Calder ↔ Wire",
        "description": "Oscillation between bold Calder asymmetry and skeletal wire minimalism. "
                       "Explores visual weight vs structural transparency.",
        "basin_size": 0.12,
        "period": 20,
        "classification": "preset",
        "source_preset": "balance_shift",
        "representative_state": {
            "balance_asymmetry": 0.625,
            "kinetic_energy": 0.45,
            "structural_density": 0.35,
            "surface_reflectivity": 0.425,
            "hierarchical_depth": 0.45
        }
    },
    "material_cycle_attractor": {
        "name": "Material Cycle — Industrial ↔ Reflective",
        "description": "Surface transformation from matte industrial heaviness to "
                       "mirror-surface lightness. Density and reflectivity counter-cycle.",
        "basin_size": 0.10,
        "period": 24,
        "classification": "preset",
        "source_preset": "material_cycle",
        "representative_state": {
            "balance_asymmetry": 0.475,
            "kinetic_energy": 0.625,
            "structural_density": 0.675,
            "surface_reflectivity": 0.575,
            "hierarchical_depth": 0.60
        }
    },
    "motion_sweep_attractor": {
        "name": "Motion Sweep — Organic Pendulum ↔ Floating Still",
        "description": "Kinetic energy decays as symmetry increases. Organic curves "
                       "yield to weightless symmetry, then reawaken.",
        "basin_size": 0.09,
        "period": 18,
        "classification": "preset",
        "source_preset": "motion_sweep",
        "representative_state": {
            "balance_asymmetry": 0.325,
            "kinetic_energy": 0.45,
            "structural_density": 0.375,
            "surface_reflectivity": 0.425,
            "hierarchical_depth": 0.275
        }
    },
    "hierarchy_pulse_attractor": {
        "name": "Hierarchy Pulse — Flat ↔ Deep Cascade",
        "description": "Slow deep breathing of hierarchical depth. Minimal single-tier "
                       "wire expands into deeply cascading reflective tiers.",
        "basin_size": 0.08,
        "period": 28,
        "classification": "preset",
        "source_preset": "hierarchy_pulse",
        "representative_state": {
            "balance_asymmetry": 0.525,
            "kinetic_energy": 0.55,
            "structural_density": 0.30,
            "surface_reflectivity": 0.825,
            "hierarchical_depth": 0.575
        }
    },
    "kinetic_toggle_attractor": {
        "name": "Kinetic Toggle — Dynamic ↔ Still",
        "description": "Abrupt switch between Calder dynamism and symmetric stillness. "
                       "Sharp contrast in balance philosophy and energy.",
        "basin_size": 0.07,
        "period": 14,
        "classification": "preset",
        "source_preset": "kinetic_toggle",
        "representative_state": {
            "balance_asymmetry": 0.45,
            "kinetic_energy": 0.40,
            "structural_density": 0.45,
            "surface_reflectivity": 0.35,
            "hierarchical_depth": 0.40
        }
    }
}


@mcp.tool()
def list_mobile_attractor_presets() -> str:
    """
    List all available mobile sculpture attractor presets for visualization.

    Phase 2.7 Tool: Shows curated attractor configurations available for
    prompt generation.

    Returns:
        JSON with preset names, descriptions, basin sizes, classifications

    Cost: 0 tokens
    """
    presets = {}
    for key, preset in MOBILE_ATTRACTOR_PRESETS.items():
        presets[key] = {
            "name": preset["name"],
            "description": preset["description"],
            "basin_size": preset["basin_size"],
            "period": preset["period"],
            "classification": preset["classification"]
        }

    return json.dumps({
        "attractor_presets": presets,
        "count": len(presets),
        "usage": "Use generate_mobile_attractor_prompt(attractor_id) for image prompts"
    }, indent=2)


@mcp.tool()
def generate_mobile_attractor_prompt(
    attractor_id: str = "",
    custom_state: str = "",
    mode: str = "composite",
    style_modifier: str = "",
    keyframe_count: int = 4
) -> str:
    """
    Generate image generation prompt from attractor state or custom coordinates.

    Phase 2.7 Tool: Translates mathematical attractor coordinates into
    visual prompts suitable for image generation (ComfyUI, Stable Diffusion,
    DALL-E, etc.).

    Modes:
        composite: Single blended prompt from attractor state
        split_view: Separate prompt for state_a and state_b extremes
        sequence: Multiple keyframe prompts along the rhythmic trajectory

    Args:
        attractor_id: Preset attractor name (or "" with custom_state)
        custom_state: Optional JSON custom parameter coordinates dict
        mode: "composite" | "split_view" | "sequence"
        style_modifier: Optional prefix ("photorealistic", "oil painting", etc.)
        keyframe_count: Number of keyframes for sequence mode (default 4)

    Returns:
        JSON with prompt(s), vocabulary details, and attractor metadata

    Cost: 0 tokens (Layer 2 deterministic)
    """
    # Resolve state
    if custom_state:
        state_dict = json.loads(custom_state) if isinstance(custom_state, str) else custom_state
        attractor_meta = {
            "name": "Custom State",
            "description": "User-defined parameter coordinates",
            "classification": "custom"
        }
        source_preset = None
    elif attractor_id and attractor_id in MOBILE_ATTRACTOR_PRESETS:
        preset = MOBILE_ATTRACTOR_PRESETS[attractor_id]
        state_dict = preset["representative_state"]
        attractor_meta = {
            "name": preset["name"],
            "description": preset["description"],
            "basin_size": preset["basin_size"],
            "period": preset["period"],
            "classification": preset["classification"]
        }
        source_preset = preset.get("source_preset")
    else:
        return json.dumps({
            "error": f"Provide attractor_id or custom_state. Available: {list(MOBILE_ATTRACTOR_PRESETS.keys())}"
        }, indent=2)

    prefix = f"{style_modifier}, " if style_modifier else ""

    # --- COMPOSITE MODE ---
    if mode == "composite":
        vocab = _extract_mobile_visual_vocabulary(state_dict)
        prompt = prefix + ", ".join(vocab["keywords"])

        return json.dumps({
            "mode": "composite",
            "prompt": prompt,
            "attractor": attractor_meta,
            "vocabulary": vocab,
            "parameter_state": {p: round(state_dict.get(p, 0.5), 4) for p in MOBILE_PARAMETER_NAMES}
        }, indent=2)

    # --- SPLIT VIEW MODE ---
    elif mode == "split_view":
        if source_preset and source_preset in MOBILE_RHYTHMIC_PRESETS:
            config = MOBILE_RHYTHMIC_PRESETS[source_preset]
            state_a_coords = {p: MOBILE_CANONICAL_STATES[config["state_a"]][p]
                              for p in MOBILE_PARAMETER_NAMES}
            state_b_coords = {p: MOBILE_CANONICAL_STATES[config["state_b"]][p]
                              for p in MOBILE_PARAMETER_NAMES}
        else:
            # Without a source preset, use min/max extremes of the state
            state_a_coords = {p: max(0.0, state_dict.get(p, 0.5) - 0.2) for p in MOBILE_PARAMETER_NAMES}
            state_b_coords = {p: min(1.0, state_dict.get(p, 0.5) + 0.2) for p in MOBILE_PARAMETER_NAMES}

        vocab_a = _extract_mobile_visual_vocabulary(state_a_coords)
        vocab_b = _extract_mobile_visual_vocabulary(state_b_coords)

        prompt_a = prefix + ", ".join(vocab_a["keywords"])
        prompt_b = prefix + ", ".join(vocab_b["keywords"])

        return json.dumps({
            "mode": "split_view",
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "vocabulary_a": vocab_a,
            "vocabulary_b": vocab_b,
            "attractor": attractor_meta
        }, indent=2)

    # --- SEQUENCE MODE ---
    elif mode == "sequence":
        if source_preset and source_preset in MOBILE_RHYTHMIC_PRESETS:
            config = MOBILE_RHYTHMIC_PRESETS[source_preset]
            trajectory = _generate_mobile_preset_trajectory(config)
        else:
            # Synthesize a simple oscillation for custom states
            trajectory = []
            for i in range(keyframe_count * 5):
                t = i / (keyframe_count * 5)
                alpha = 0.5 * (1.0 + math.sin(2 * math.pi * t))
                frame = {}
                for p in MOBILE_PARAMETER_NAMES:
                    base = state_dict.get(p, 0.5)
                    frame[p] = base + 0.15 * math.sin(2 * math.pi * t + hash(p) % 7)
                    frame[p] = max(0.0, min(1.0, frame[p]))
                trajectory.append(frame)

        total = len(trajectory)
        step_size = max(1, total // keyframe_count)

        keyframes = []
        for k in range(keyframe_count):
            idx = min(k * step_size, total - 1)
            frame_state = trajectory[idx]
            vocab = _extract_mobile_visual_vocabulary(frame_state)
            prompt = prefix + ", ".join(vocab["keywords"])

            keyframes.append({
                "keyframe": k,
                "step": idx,
                "prompt": prompt,
                "vocabulary": vocab,
                "state": {p: round(frame_state.get(p, 0.5), 4) for p in MOBILE_PARAMETER_NAMES}
            })

        return json.dumps({
            "mode": "sequence",
            "keyframe_count": len(keyframes),
            "keyframes": keyframes,
            "attractor": attractor_meta,
            "total_trajectory_steps": total
        }, indent=2)

    else:
        return json.dumps({"error": f"Unknown mode: {mode}. Use composite, split_view, or sequence"}, indent=2)


@mcp.tool()
def generate_mobile_sequence_prompts(
    preset_name: str,
    keyframe_count: int = 4,
    style_modifier: str = ""
) -> str:
    """
    Generate keyframe prompts from a Phase 2.6 rhythmic preset.

    Phase 2.7 Tool: Extracts evenly-spaced keyframes from a rhythmic
    oscillation sequence and generates an image prompt for each.

    Useful for:
    - Storyboard generation from rhythmic compositions
    - Animation keyframe specification
    - Multi-panel visualization of temporal aesthetic evolution

    Args:
        preset_name: Phase 2.6 preset name
        keyframe_count: Number of keyframes to extract (default 4)
        style_modifier: Optional style prefix for all prompts

    Returns:
        JSON with keyframes, each containing step, state, prompt, vocabulary

    Cost: 0 tokens
    """
    if preset_name not in MOBILE_RHYTHMIC_PRESETS:
        return json.dumps({
            "error": f"Unknown preset: {preset_name}",
            "available": list(MOBILE_RHYTHMIC_PRESETS.keys())
        }, indent=2)

    config = MOBILE_RHYTHMIC_PRESETS[preset_name]
    trajectory = _generate_mobile_preset_trajectory(config)
    total = len(trajectory)
    step_size = max(1, total // keyframe_count)
    prefix = f"{style_modifier}, " if style_modifier else ""

    keyframes = []
    for k in range(keyframe_count):
        idx = min(k * step_size, total - 1)
        frame_state = trajectory[idx]
        vocab = _extract_mobile_visual_vocabulary(frame_state)
        prompt = prefix + ", ".join(vocab["keywords"])

        keyframes.append({
            "keyframe": k,
            "step": idx,
            "phase": round(idx / total, 3),
            "prompt": prompt,
            "vocabulary": vocab,
            "state": {p: round(frame_state.get(p, 0.5), 4) for p in MOBILE_PARAMETER_NAMES}
        })

    return json.dumps({
        "preset": preset_name,
        "period": config["steps_per_cycle"],
        "pattern": config["pattern"],
        "description": config["description"],
        "keyframe_count": len(keyframes),
        "keyframes": keyframes
    }, indent=2)


@mcp.tool()
def compute_mobile_distance(state_a: str, state_b: str) -> str:
    """
    Compute distance between two mobile sculpture states in parameter space.

    Layer 2: Pure distance computation (0 tokens)

    Args:
        state_a: JSON dict or canonical state name
        state_b: JSON dict or canonical state name

    Returns:
        Distance value and per-parameter breakdown
    """
    # Resolve states
    def _resolve(s):
        if s in MOBILE_CANONICAL_STATES:
            return {p: MOBILE_CANONICAL_STATES[s][p] for p in MOBILE_PARAMETER_NAMES}
        return json.loads(s) if isinstance(s, str) else s

    a = _resolve(state_a)
    b = _resolve(state_b)

    vec_a = np.array([a[p] for p in MOBILE_PARAMETER_NAMES])
    vec_b = np.array([b[p] for p in MOBILE_PARAMETER_NAMES])

    euclidean = float(np.linalg.norm(vec_a - vec_b))
    per_param = {p: round(abs(a[p] - b[p]), 4) for p in MOBILE_PARAMETER_NAMES}
    max_param = max(per_param, key=per_param.get)

    return json.dumps({
        "euclidean_distance": round(euclidean, 4),
        "per_parameter": per_param,
        "max_difference": {"parameter": max_param, "value": per_param[max_param]},
        "state_a_resolved": {p: round(a[p], 4) for p in MOBILE_PARAMETER_NAMES},
        "state_b_resolved": {p: round(b[p], 4) for p in MOBILE_PARAMETER_NAMES}
    }, indent=2)


# ============================================================================
# SERVER INFO (updated for Phase 2.6 + 2.7)
# ============================================================================

@mcp.tool()
def get_server_info() -> str:
    """Get information about the Mobile Sculpture MCP server."""
    return json.dumps({
        "server_name": "Mobile Sculpture Aesthetics MCP",
        "version": "2.0.0",
        "architecture": "three_layer_olog",
        "description": "Kinetic sculpture visual vocabulary based on Calder mobiles and balance physics, "
                       "with Phase 2.6 rhythmic composition and Phase 2.7 attractor visualization",
        "layers": {
            "layer_1": {
                "name": "Pure Taxonomy",
                "cost": "0 tokens",
                "operations": [
                    "Balance type specifications",
                    "Movement vocabularies",
                    "Material/surface taxonomies",
                    "Spatial relationship parameters",
                    "Canonical state coordinates"
                ]
            },
            "layer_2": {
                "name": "Deterministic Operations",
                "cost": "0 tokens",
                "operations": [
                    "Torque equilibrium physics calculations",
                    "Keyword extraction and parameter mapping",
                    "Geometric specification generation",
                    "Visual vocabulary extraction",
                    "Phase 2.6 rhythmic trajectory generation",
                    "Phase 2.7 attractor prompt generation",
                    "Distance computation in parameter space"
                ]
            },
            "layer_3": {
                "name": "Creative Synthesis Interface",
                "cost": "~150-200 tokens per synthesis",
                "operations": [
                    "Natural language prompt synthesis",
                    "Geometric precision preservation",
                    "Compositional description assembly"
                ]
            }
        },
        "phase_2_6_enhancements": {
            "rhythmic_composition": True,
            "parameter_space": MOBILE_PARAMETER_NAMES,
            "canonical_states": list(MOBILE_CANONICAL_STATES.keys()),
            "presets": {
                name: {
                    "period": config["steps_per_cycle"],
                    "pattern": config["pattern"],
                    "states": f"{config['state_a']} ↔ {config['state_b']}"
                }
                for name, config in MOBILE_RHYTHMIC_PRESETS.items()
            },
            "periods": sorted(set(c["steps_per_cycle"] for c in MOBILE_RHYTHMIC_PRESETS.values()))
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "visual_types": list(MOBILE_VISUAL_TYPES.keys()),
            "attractor_presets": list(MOBILE_ATTRACTOR_PRESETS.keys()),
            "prompt_modes": ["composite", "split_view", "sequence"],
            "supported_image_generators": [
                "ComfyUI", "Stable Diffusion", "DALL-E", "Midjourney"
            ]
        },
        "cost_savings": "~60-85% vs pure LLM approach (single synthesis call only)",
        "balance_physics": "Implements torque equilibrium: Σ(weight × distance) = 0",
        "geometric_precision": "Explicit coordinates, angles, scales for image generation",
        "domains_covered": [
            "Kinetic sculpture composition",
            "Balance dynamics",
            "Movement vocabularies",
            "Material/surface interaction",
            "Spatial relationships",
            "Rhythmic temporal aesthetics",
            "Attractor-based visualization"
        ],
        "usage_patterns": {
            "original": "map_prompt → generate_geometric_specs → extract_vocabulary → prepare_synthesis",
            "phase_2_6": "list_presets → apply_preset → extract keyframes",
            "phase_2_7": "list_attractors → generate_prompt(composite|split_view|sequence)"
        }
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
