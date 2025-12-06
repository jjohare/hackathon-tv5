/// FFI-Safe GPU Ontology Structures
///
/// This module provides FFI-safe structs that match CUDA kernel definitions
/// in `src/cuda/kernels/ontology_reasoning.cu` with byte-exact alignment.
///
/// CRITICAL: These structs MUST maintain exact binary compatibility with CUDA.
/// Any changes here require corresponding changes in the CUDA code.

use std::mem;

/// 3D float vector matching CUDA's float3
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
}

/// MediaOntologyNode - MUST match CUDA struct exactly (64-byte aligned)
///
/// CUDA struct location: src/cuda/kernels/ontology_reasoning.cu:37-55
///
/// Memory Layout:
/// - Offset 0-3:   graph_id (u32)
/// - Offset 4-7:   node_id (u32)
/// - Offset 8-11:  ontology_type (u32)
/// - Offset 12-15: constraint_flags (u32)
/// - Offset 16-27: position (float3)
/// - Offset 28-39: velocity (float3)
/// - Offset 40-43: mass (f32)
/// - Offset 44-47: radius (f32)
/// - Offset 48-51: parent_genre (u32)
/// - Offset 52-55: property_count (u32)
/// - Offset 56-59: cultural_flags (u32)
/// - Offset 60-63: mood_flags (u32)
/// - Offset 64-79: padding (4 x u32)
/// Total: 80 bytes
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct MediaOntologyNode {
    pub graph_id: u32,           // Multi-graph support
    pub node_id: u32,            // Unique identifier within graph
    pub ontology_type: u32,      // MEDIA_CONTENT, GENRE, MOOD, CONTEXT
    pub constraint_flags: u32,   // Active constraint types

    pub position: Float3,        // Semantic space coordinates
    pub velocity: Float3,        // Momentum in semantic space

    pub mass: f32,               // Importance weight
    pub radius: f32,             // Semantic influence radius

    pub parent_genre: u32,       // Genre hierarchy
    pub property_count: u32,     // Associated properties
    pub cultural_flags: u32,     // Cultural context identifiers
    pub mood_flags: u32,         // Mood/aesthetic identifiers

    pub padding: [u32; 4],       // Align to 64 bytes
}

// Compile-time size and alignment assertions
const _: () = assert!(mem::size_of::<MediaOntologyNode>() == 80);
const _: () = assert!(mem::align_of::<MediaOntologyNode>() == 64);

// Verify field offsets match CUDA
const _: () = {
    let offset_graph_id = unsafe {
        let base = mem::MaybeUninit::<MediaOntologyNode>::uninit();
        let ptr = base.as_ptr();
        &(*ptr).graph_id as *const _ as usize - ptr as usize
    };
    assert!(offset_graph_id == 0);
};

const _: () = {
    let offset_position = unsafe {
        let base = mem::MaybeUninit::<MediaOntologyNode>::uninit();
        let ptr = base.as_ptr();
        &(*ptr).position as *const _ as usize - ptr as usize
    };
    assert!(offset_position == 16);
};

impl MediaOntologyNode {
    pub const fn new() -> Self {
        Self {
            graph_id: 0,
            node_id: 0,
            ontology_type: 0,
            constraint_flags: 0,
            position: Float3::zero(),
            velocity: Float3::zero(),
            mass: 1.0,
            radius: 1.0,
            parent_genre: 0,
            property_count: 0,
            cultural_flags: 0,
            mood_flags: 0,
            padding: [0; 4],
        }
    }

    /// Verify struct is properly aligned for GPU
    pub fn is_aligned(&self) -> bool {
        (self as *const _ as usize) % 64 == 0
    }

    /// Size in bytes
    pub const fn size_bytes() -> usize {
        mem::size_of::<Self>()
    }
}

impl Default for MediaOntologyNode {
    fn default() -> Self {
        Self::new()
    }
}

/// MediaOntologyConstraint - MUST match CUDA struct exactly (64-byte aligned)
///
/// CUDA struct location: src/cuda/kernels/ontology_reasoning.cu:68-83
///
/// Memory Layout:
/// - Offset 0-3:   type (u32)
/// - Offset 4-7:   source_id (u32)
/// - Offset 8-11:  target_id (u32)
/// - Offset 12-15: graph_id (u32)
/// - Offset 16-19: strength (f32)
/// - Offset 20-23: distance (f32)
/// - Offset 24-27: mood_weight (f32)
/// - Offset 28-31: cultural_weight (f32)
/// - Offset 32-35: flags (u32)
/// - Offset 36-63: padding (7 x f32)
/// Total: 64 bytes
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct MediaOntologyConstraint {
    pub constraint_type: u32,    // Constraint type identifier
    pub source_id: u32,          // Source node ID
    pub target_id: u32,          // Target node ID
    pub graph_id: u32,           // Graph identifier

    pub strength: f32,           // Enforcement strength (0.0-1.0)
    pub distance: f32,           // Ideal semantic distance

    pub mood_weight: f32,        // Mood similarity influence
    pub cultural_weight: f32,    // Cultural context influence

    pub flags: u32,              // Additional constraint modifiers

    pub padding: [f32; 7],       // Align to 64 bytes
}

// Compile-time size and alignment assertions
const _: () = assert!(mem::size_of::<MediaOntologyConstraint>() == 64);
const _: () = assert!(mem::align_of::<MediaOntologyConstraint>() == 64);

impl MediaOntologyConstraint {
    pub const fn new() -> Self {
        Self {
            constraint_type: 0,
            source_id: 0,
            target_id: 0,
            graph_id: 0,
            strength: 1.0,
            distance: 1.0,
            mood_weight: 1.0,
            cultural_weight: 1.0,
            flags: 0,
            padding: [0.0; 7],
        }
    }

    /// Verify struct is properly aligned for GPU
    pub fn is_aligned(&self) -> bool {
        (self as *const _ as usize) % 64 == 0
    }

    /// Size in bytes
    pub const fn size_bytes() -> usize {
        mem::size_of::<Self>()
    }
}

impl Default for MediaOntologyConstraint {
    fn default() -> Self {
        Self::new()
    }
}

/// Constraint type constants (must match CUDA definitions)
pub mod constraint_types {
    // Core ontology constraints (OWL-based)
    pub const DISJOINT_GENRES: u32 = 1;
    pub const GENRE_HIERARCHY: u32 = 2;
    pub const CONTENT_EQUIVALENCE: u32 = 3;
    pub const INVERSE_RELATION: u32 = 4;
    pub const FUNCTIONAL_PROPERTY: u32 = 5;

    // Media-specific constraints (GMC-O extensions)
    pub const MOOD_CONSISTENCY: u32 = 6;
    pub const CULTURAL_ALIGNMENT: u32 = 7;
    pub const VIEWER_PREFERENCE: u32 = 8;
    pub const TEMPORAL_CONTEXT: u32 = 9;
    pub const CONTENT_SIMILARITY: u32 = 10;
}

/// Ontology type flags (must match CUDA definitions)
pub mod ontology_types {
    pub const MEDIA_CONTENT: u32 = 0x01;
    pub const GENRE: u32 = 0x02;
    pub const MOOD: u32 = 0x04;
    pub const CULTURAL: u32 = 0x08;
    pub const VIEWER_SEGMENT: u32 = 0x10;
    pub const TEMPORAL: u32 = 0x20;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_ontology_node_size() {
        assert_eq!(mem::size_of::<MediaOntologyNode>(), 80);
        assert_eq!(mem::align_of::<MediaOntologyNode>(), 64);
    }

    #[test]
    fn test_media_ontology_constraint_size() {
        assert_eq!(mem::size_of::<MediaOntologyConstraint>(), 64);
        assert_eq!(mem::align_of::<MediaOntologyConstraint>(), 64);
    }

    #[test]
    fn test_float3_size() {
        assert_eq!(mem::size_of::<Float3>(), 12);
        assert_eq!(mem::align_of::<Float3>(), 4);
    }

    #[test]
    fn test_node_alignment() {
        let node = MediaOntologyNode::default();
        assert!(node.is_aligned());
    }

    #[test]
    fn test_constraint_alignment() {
        let constraint = MediaOntologyConstraint::default();
        assert!(constraint.is_aligned());
    }

    #[test]
    fn test_field_offsets() {
        use std::mem::offset_of;

        // Verify MediaOntologyNode field offsets
        assert_eq!(offset_of!(MediaOntologyNode, graph_id), 0);
        assert_eq!(offset_of!(MediaOntologyNode, node_id), 4);
        assert_eq!(offset_of!(MediaOntologyNode, ontology_type), 8);
        assert_eq!(offset_of!(MediaOntologyNode, constraint_flags), 12);
        assert_eq!(offset_of!(MediaOntologyNode, position), 16);
        assert_eq!(offset_of!(MediaOntologyNode, velocity), 28);
        assert_eq!(offset_of!(MediaOntologyNode, mass), 40);
        assert_eq!(offset_of!(MediaOntologyNode, radius), 44);
        assert_eq!(offset_of!(MediaOntologyNode, parent_genre), 48);
        assert_eq!(offset_of!(MediaOntologyNode, property_count), 52);
        assert_eq!(offset_of!(MediaOntologyNode, cultural_flags), 56);
        assert_eq!(offset_of!(MediaOntologyNode, mood_flags), 60);
        assert_eq!(offset_of!(MediaOntologyNode, padding), 64);

        // Verify MediaOntologyConstraint field offsets
        assert_eq!(offset_of!(MediaOntologyConstraint, constraint_type), 0);
        assert_eq!(offset_of!(MediaOntologyConstraint, source_id), 4);
        assert_eq!(offset_of!(MediaOntologyConstraint, target_id), 8);
        assert_eq!(offset_of!(MediaOntologyConstraint, graph_id), 12);
        assert_eq!(offset_of!(MediaOntologyConstraint, strength), 16);
        assert_eq!(offset_of!(MediaOntologyConstraint, distance), 20);
        assert_eq!(offset_of!(MediaOntologyConstraint, mood_weight), 24);
        assert_eq!(offset_of!(MediaOntologyConstraint, cultural_weight), 28);
        assert_eq!(offset_of!(MediaOntologyConstraint, flags), 32);
        assert_eq!(offset_of!(MediaOntologyConstraint, padding), 36);
    }

    #[test]
    fn test_zero_initialization() {
        let node = MediaOntologyNode::default();
        assert_eq!(node.graph_id, 0);
        assert_eq!(node.node_id, 0);
        assert_eq!(node.mass, 1.0);
        assert_eq!(node.position.x, 0.0);
    }

    #[test]
    fn test_constraint_defaults() {
        let constraint = MediaOntologyConstraint::default();
        assert_eq!(constraint.constraint_type, 0);
        assert_eq!(constraint.strength, 1.0);
        assert_eq!(constraint.distance, 1.0);
    }
}
