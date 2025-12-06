/// Comprehensive FFI Safety Tests for GPU Ontology Structures
///
/// These tests verify binary compatibility between Rust and CUDA structures
/// through serialization, memory layout verification, and round-trip testing.

#[cfg(test)]
mod ffi_safety_tests {
    use super::super::ontology_ffi::*;
    use std::mem;
    use std::ptr;
    use std::slice;

    /// Test serialization to raw bytes and back
    #[test]
    fn test_node_serialization_roundtrip() {
        // Create test node with known values
        let mut original = MediaOntologyNode::new();
        original.graph_id = 42;
        original.node_id = 1337;
        original.ontology_type = ontology_types::MEDIA_CONTENT | ontology_types::GENRE;
        original.constraint_flags = 0xFF;
        original.position = Float3::new(1.0, 2.0, 3.0);
        original.velocity = Float3::new(0.5, 0.25, 0.125);
        original.mass = 10.5;
        original.radius = 2.5;
        original.parent_genre = 100;
        original.property_count = 5;
        original.cultural_flags = 0x0F;
        original.mood_flags = 0xAA;

        // Serialize to bytes
        let bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                &original as *const _ as *const u8,
                mem::size_of::<MediaOntologyNode>(),
            )
        };

        // Verify size
        assert_eq!(bytes.len(), 80);

        // Deserialize from bytes
        let deserialized: MediaOntologyNode = unsafe {
            ptr::read(bytes.as_ptr() as *const MediaOntologyNode)
        };

        // Verify all fields match
        assert_eq!(deserialized.graph_id, original.graph_id);
        assert_eq!(deserialized.node_id, original.node_id);
        assert_eq!(deserialized.ontology_type, original.ontology_type);
        assert_eq!(deserialized.constraint_flags, original.constraint_flags);
        assert_eq!(deserialized.position.x, original.position.x);
        assert_eq!(deserialized.position.y, original.position.y);
        assert_eq!(deserialized.position.z, original.position.z);
        assert_eq!(deserialized.velocity.x, original.velocity.x);
        assert_eq!(deserialized.velocity.y, original.velocity.y);
        assert_eq!(deserialized.velocity.z, original.velocity.z);
        assert_eq!(deserialized.mass, original.mass);
        assert_eq!(deserialized.radius, original.radius);
        assert_eq!(deserialized.parent_genre, original.parent_genre);
        assert_eq!(deserialized.property_count, original.property_count);
        assert_eq!(deserialized.cultural_flags, original.cultural_flags);
        assert_eq!(deserialized.mood_flags, original.mood_flags);
    }

    /// Test constraint serialization
    #[test]
    fn test_constraint_serialization_roundtrip() {
        let mut original = MediaOntologyConstraint::new();
        original.constraint_type = constraint_types::DISJOINT_GENRES;
        original.source_id = 10;
        original.target_id = 20;
        original.graph_id = 1;
        original.strength = 0.95;
        original.distance = 5.0;
        original.mood_weight = 0.6;
        original.cultural_weight = 0.8;
        original.flags = 0x123;

        let bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                &original as *const _ as *const u8,
                mem::size_of::<MediaOntologyConstraint>(),
            )
        };

        assert_eq!(bytes.len(), 64);

        let deserialized: MediaOntologyConstraint = unsafe {
            ptr::read(bytes.as_ptr() as *const MediaOntologyConstraint)
        };

        assert_eq!(deserialized.constraint_type, original.constraint_type);
        assert_eq!(deserialized.source_id, original.source_id);
        assert_eq!(deserialized.target_id, original.target_id);
        assert_eq!(deserialized.graph_id, original.graph_id);
        assert_eq!(deserialized.strength, original.strength);
        assert_eq!(deserialized.distance, original.distance);
        assert_eq!(deserialized.mood_weight, original.mood_weight);
        assert_eq!(deserialized.cultural_weight, original.cultural_weight);
        assert_eq!(deserialized.flags, original.flags);
    }

    /// Test extreme values (u32::MAX, f32::MAX, etc.)
    #[test]
    fn test_extreme_values() {
        let mut node = MediaOntologyNode::new();
        node.graph_id = u32::MAX;
        node.node_id = u32::MAX;
        node.ontology_type = u32::MAX;
        node.constraint_flags = u32::MAX;
        node.position = Float3::new(f32::MAX, f32::MIN, 0.0);
        node.velocity = Float3::new(f32::MIN_POSITIVE, f32::MAX, -f32::MAX);
        node.mass = f32::MAX;
        node.radius = f32::MIN_POSITIVE;
        node.parent_genre = u32::MAX;
        node.property_count = u32::MAX;
        node.cultural_flags = u32::MAX;
        node.mood_flags = u32::MAX;

        // Serialize and deserialize
        let bytes = unsafe {
            slice::from_raw_parts(
                &node as *const _ as *const u8,
                mem::size_of::<MediaOntologyNode>(),
            )
        };

        let deserialized: MediaOntologyNode = unsafe {
            ptr::read(bytes.as_ptr() as *const MediaOntologyNode)
        };

        // Verify extreme values preserved
        assert_eq!(deserialized.graph_id, u32::MAX);
        assert_eq!(deserialized.node_id, u32::MAX);
        assert_eq!(deserialized.mass, f32::MAX);
        assert_eq!(deserialized.radius, f32::MIN_POSITIVE);
    }

    /// Test array of nodes for batch processing
    #[test]
    fn test_node_array_layout() {
        const COUNT: usize = 10;
        let mut nodes = vec![MediaOntologyNode::default(); COUNT];

        // Initialize with sequential IDs
        for (i, node) in nodes.iter_mut().enumerate() {
            node.node_id = i as u32;
            node.graph_id = 1;
        }

        // Verify array is contiguous
        let first_ptr = &nodes[0] as *const _ as usize;
        let second_ptr = &nodes[1] as *const _ as usize;
        assert_eq!(second_ptr - first_ptr, 80);

        // Verify all alignments
        for node in &nodes {
            assert!(node.is_aligned());
        }
    }

    /// Test constraint array layout
    #[test]
    fn test_constraint_array_layout() {
        const COUNT: usize = 10;
        let mut constraints = vec![MediaOntologyConstraint::default(); COUNT];

        for (i, constraint) in constraints.iter_mut().enumerate() {
            constraint.source_id = i as u32;
            constraint.target_id = (i + 1) as u32;
        }

        let first_ptr = &constraints[0] as *const _ as usize;
        let second_ptr = &constraints[1] as *const _ as usize;
        assert_eq!(second_ptr - first_ptr, 64);

        for constraint in &constraints {
            assert!(constraint.is_aligned());
        }
    }

    /// Test zero-copy pointer casting
    #[test]
    fn test_zero_copy_casting() {
        let node = MediaOntologyNode::default();
        let ptr = &node as *const MediaOntologyNode;

        // Cast to u8 and back
        let u8_ptr = ptr as *const u8;
        let back_ptr = u8_ptr as *const MediaOntologyNode;

        unsafe {
            assert_eq!((*ptr).node_id, (*back_ptr).node_id);
        }
    }

    /// Test float3 internal layout
    #[test]
    fn test_float3_layout() {
        let f3 = Float3::new(1.0, 2.0, 3.0);

        let bytes = unsafe {
            slice::from_raw_parts(&f3 as *const _ as *const u8, mem::size_of::<Float3>())
        };

        assert_eq!(bytes.len(), 12);

        // Verify little-endian float layout
        let x_bytes = &bytes[0..4];
        let y_bytes = &bytes[4..8];
        let z_bytes = &bytes[8..12];

        let x = f32::from_le_bytes([x_bytes[0], x_bytes[1], x_bytes[2], x_bytes[3]]);
        let y = f32::from_le_bytes([y_bytes[0], y_bytes[1], y_bytes[2], y_bytes[3]]);
        let z = f32::from_le_bytes([z_bytes[0], z_bytes[1], z_bytes[2], z_bytes[3]]);

        assert_eq!(x, 1.0);
        assert_eq!(y, 2.0);
        assert_eq!(z, 3.0);
    }

    /// Test endianness consistency
    #[test]
    fn test_endianness() {
        let node = MediaOntologyNode::new();
        let id: u32 = 0x12345678;

        let mut test_node = node;
        test_node.node_id = id;

        let bytes = unsafe {
            slice::from_raw_parts(
                &test_node.node_id as *const _ as *const u8,
                mem::size_of::<u32>(),
            )
        };

        // Verify little-endian byte order (standard for x86_64 and ARM)
        if cfg!(target_endian = "little") {
            assert_eq!(bytes[0], 0x78);
            assert_eq!(bytes[1], 0x56);
            assert_eq!(bytes[2], 0x34);
            assert_eq!(bytes[3], 0x12);
        }
    }

    /// Test padding doesn't affect neighboring fields
    #[test]
    fn test_padding_isolation() {
        let mut node = MediaOntologyNode::new();
        node.mood_flags = 0xDEADBEEF;
        node.padding = [0xFFFFFFFF; 4];

        // Verify padding doesn't corrupt adjacent field
        assert_eq!(node.mood_flags, 0xDEADBEEF);

        // Verify we can read/write padding without issues
        node.padding[0] = 0x12345678;
        assert_eq!(node.padding[0], 0x12345678);
        assert_eq!(node.mood_flags, 0xDEADBEEF);
    }

    /// Test GPU memory coalescing alignment (64-byte)
    #[test]
    fn test_gpu_coalescing_alignment() {
        let nodes = vec![MediaOntologyNode::default(); 100];

        for (i, node) in nodes.iter().enumerate() {
            let addr = node as *const _ as usize;
            // Every node should be 64-byte aligned
            assert_eq!(
                addr % 64,
                0,
                "Node {} at address 0x{:x} not 64-byte aligned",
                i,
                addr
            );
        }
    }

    /// Benchmark-style test: 10K node serialization
    #[test]
    fn test_bulk_serialization_10k() {
        const COUNT: usize = 10_000;
        let mut nodes = Vec::with_capacity(COUNT);

        for i in 0..COUNT {
            let mut node = MediaOntologyNode::new();
            node.node_id = i as u32;
            node.graph_id = 1;
            node.position = Float3::new(i as f32, (i * 2) as f32, (i * 3) as f32);
            nodes.push(node);
        }

        // Serialize entire array
        let total_bytes = COUNT * mem::size_of::<MediaOntologyNode>();
        assert_eq!(total_bytes, COUNT * 80);

        let bytes = unsafe {
            slice::from_raw_parts(nodes.as_ptr() as *const u8, total_bytes)
        };

        assert_eq!(bytes.len(), total_bytes);

        // Verify first and last nodes
        let first: MediaOntologyNode =
            unsafe { ptr::read(bytes.as_ptr() as *const MediaOntologyNode) };
        assert_eq!(first.node_id, 0);

        let last_offset = (COUNT - 1) * 80;
        let last: MediaOntologyNode =
            unsafe { ptr::read(bytes[last_offset..].as_ptr() as *const MediaOntologyNode) };
        assert_eq!(last.node_id, (COUNT - 1) as u32);
    }

    /// Test that default initialization is safe
    #[test]
    fn test_safe_defaults() {
        let node = MediaOntologyNode::default();
        let constraint = MediaOntologyConstraint::default();

        // Defaults should be sensible, not arbitrary memory
        assert_eq!(node.graph_id, 0);
        assert_eq!(node.node_id, 0);
        assert_eq!(node.mass, 1.0);
        assert_eq!(node.radius, 1.0);

        assert_eq!(constraint.constraint_type, 0);
        assert_eq!(constraint.strength, 1.0);
        assert_eq!(constraint.distance, 1.0);
    }

    /// Platform-specific size verification
    #[test]
    fn test_platform_sizes() {
        // These should be identical across x86_64 and ARM64
        assert_eq!(mem::size_of::<u32>(), 4);
        assert_eq!(mem::size_of::<f32>(), 4);
        assert_eq!(mem::size_of::<Float3>(), 12);
        assert_eq!(mem::size_of::<MediaOntologyNode>(), 80);
        assert_eq!(mem::size_of::<MediaOntologyConstraint>(), 64);

        println!("Platform: {}", std::env::consts::ARCH);
        println!("Endian: {}", cfg!(target_endian = "little"));
        println!("Pointer size: {}", mem::size_of::<*const ()>());
    }
}

/// Integration tests simulating GPU transfers
#[cfg(test)]
mod gpu_transfer_simulation {
    use super::super::ontology_ffi::*;
    use std::mem;
    use std::slice;

    /// Simulate host-to-device transfer
    fn simulate_htod_transfer(nodes: &[MediaOntologyNode]) -> Vec<u8> {
        let size = nodes.len() * mem::size_of::<MediaOntologyNode>();
        let bytes =
            unsafe { slice::from_raw_parts(nodes.as_ptr() as *const u8, size) };
        bytes.to_vec()
    }

    /// Simulate device-to-host transfer
    fn simulate_dtoh_transfer(bytes: &[u8]) -> Vec<MediaOntologyNode> {
        let count = bytes.len() / mem::size_of::<MediaOntologyNode>();
        let mut nodes = Vec::with_capacity(count);

        for i in 0..count {
            let offset = i * mem::size_of::<MediaOntologyNode>();
            let node: MediaOntologyNode = unsafe {
                std::ptr::read(bytes[offset..].as_ptr() as *const MediaOntologyNode)
            };
            nodes.push(node);
        }

        nodes
    }

    #[test]
    fn test_roundtrip_transfer() {
        // Create test data
        let mut original_nodes = Vec::new();
        for i in 0..100 {
            let mut node = MediaOntologyNode::new();
            node.node_id = i;
            node.graph_id = 1;
            node.position = Float3::new(i as f32, (i * 2) as f32, (i * 3) as f32);
            node.mass = (i + 1) as f32;
            original_nodes.push(node);
        }

        // Simulate transfer to GPU
        let gpu_bytes = simulate_htod_transfer(&original_nodes);

        // Simulate GPU processing (no-op in this test)

        // Simulate transfer back from GPU
        let returned_nodes = simulate_dtoh_transfer(&gpu_bytes);

        // Verify data integrity
        assert_eq!(returned_nodes.len(), original_nodes.len());

        for (original, returned) in original_nodes.iter().zip(returned_nodes.iter()) {
            assert_eq!(original.node_id, returned.node_id);
            assert_eq!(original.graph_id, returned.graph_id);
            assert_eq!(original.position.x, returned.position.x);
            assert_eq!(original.position.y, returned.position.y);
            assert_eq!(original.position.z, returned.position.z);
            assert_eq!(original.mass, returned.mass);
        }
    }

    #[test]
    fn test_constraint_roundtrip() {
        let mut constraints = Vec::new();
        for i in 0..50 {
            let mut c = MediaOntologyConstraint::new();
            c.source_id = i;
            c.target_id = i + 1;
            c.constraint_type = constraint_types::GENRE_HIERARCHY;
            c.strength = 0.5 + (i as f32 * 0.01);
            constraints.push(c);
        }

        let bytes = unsafe {
            slice::from_raw_parts(
                constraints.as_ptr() as *const u8,
                constraints.len() * mem::size_of::<MediaOntologyConstraint>(),
            )
        };

        let returned: Vec<MediaOntologyConstraint> = (0..constraints.len())
            .map(|i| {
                let offset = i * mem::size_of::<MediaOntologyConstraint>();
                unsafe {
                    std::ptr::read(bytes[offset..].as_ptr() as *const MediaOntologyConstraint)
                }
            })
            .collect();

        for (orig, ret) in constraints.iter().zip(returned.iter()) {
            assert_eq!(orig.source_id, ret.source_id);
            assert_eq!(orig.target_id, ret.target_id);
            assert_eq!(orig.constraint_type, ret.constraint_type);
            assert!((orig.strength - ret.strength).abs() < 1e-6);
        }
    }
}
