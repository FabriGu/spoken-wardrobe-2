"""
Rigged Mesh Loader - Loads GLB meshes with skeleton and skin weights

Extracts:
- Mesh geometry (vertices, faces, normals, UVs)
- Skeleton structure (bones, hierarchy, transforms)
- Skin weights (which vertices influenced by which bones)
"""

import numpy as np
import pygltflib
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct


class SkeletonBone:
    """Represents a single bone in the skeleton"""
    def __init__(self, name: str, index: int, parent_idx: Optional[int] = None):
        self.name = name
        self.index = index
        self.parent_idx = parent_idx
        self.children_idx = []

        # Transforms
        self.local_transform = np.eye(4)  # Relative to parent
        self.inverse_bind_matrix = np.eye(4)  # Bind pose inverse
        self.world_transform = np.eye(4)  # Current world position


class RiggedMesh:
    """Container for rigged mesh data"""
    def __init__(self):
        # Geometry
        self.vertices = None
        self.faces = None
        self.normals = None
        self.uvs = None

        # Skeleton
        self.bones: List[SkeletonBone] = []
        self.bone_name_to_idx: Dict[str, int] = {}
        self.root_bone_idx = None

        # Skinning
        self.skin_weights = None  # (n_vertices, n_bones) sparse matrix
        self.skin_indices = None  # (n_vertices, max_influences) which bones

        # Metadata
        self.bounds = None
        self.center = None


class RiggedMeshLoader:
    """Loads rigged meshes from GLB files"""

    @staticmethod
    def load(glb_path: str) -> RiggedMesh:
        """Load a rigged mesh from GLB file"""
        glb_path = Path(glb_path)
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

        result = RiggedMesh()

        # Load with pygltflib for skeleton data
        gltf = pygltflib.GLTF2().load(str(glb_path))

        # Load with trimesh for geometry
        mesh = trimesh.load(str(glb_path), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # Take first geometry if multiple
            mesh = list(mesh.geometry.values())[0]

        # Extract geometry
        result.vertices = np.array(mesh.vertices, dtype=np.float32)
        result.faces = np.array(mesh.faces, dtype=np.int32)
        result.normals = np.array(mesh.vertex_normals, dtype=np.float32)

        # Try to get UVs
        if hasattr(mesh.visual, 'uv'):
            result.uvs = np.array(mesh.visual.uv, dtype=np.float32)

        # Compute bounds
        result.bounds = mesh.bounds
        result.center = mesh.centroid

        # Extract skeleton if present
        if gltf.skins and len(gltf.skins) > 0:
            RiggedMeshLoader._load_skeleton(gltf, result)
            RiggedMeshLoader._load_skin_weights(gltf, result)

        return result

    @staticmethod
    def _load_skeleton(gltf: pygltflib.GLTF2, result: RiggedMesh):
        """Extract skeleton structure from GLTF"""
        skin = gltf.skins[0]

        # Create bones
        for joint_idx in skin.joints:
            node = gltf.nodes[joint_idx]
            bone = SkeletonBone(
                name=node.name or f"bone_{joint_idx}",
                index=len(result.bones)
            )
            result.bones.append(bone)
            result.bone_name_to_idx[bone.name] = bone.index

        # Build hierarchy
        for i, joint_idx in enumerate(skin.joints):
            node = gltf.nodes[joint_idx]

            # Find parent
            if node.children:
                for child_idx in node.children:
                    # Check if child is a joint
                    if child_idx in skin.joints:
                        child_bone_idx = skin.joints.index(child_idx)
                        result.bones[child_bone_idx].parent_idx = i
                        result.bones[i].children_idx.append(child_bone_idx)

        # Find root (bone with no parent)
        for bone in result.bones:
            if bone.parent_idx is None:
                result.root_bone_idx = bone.index
                break

        # Load inverse bind matrices
        if skin.inverseBindMatrices is not None:
            matrices = RiggedMeshLoader._read_accessor(gltf, gltf.accessors[skin.inverseBindMatrices])
            matrices = matrices.reshape(-1, 4, 4)
            for i in range(len(result.bones)):
                result.bones[i].inverse_bind_matrix = matrices[i]

    @staticmethod
    def _load_skin_weights(gltf: pygltflib.GLTF2, result: RiggedMesh):
        """Extract skin weights from GLTF - must match vertex count"""
        # GLB files can have multiple meshes/primitives
        # We need to load skin weights for ALL vertices, matching trimesh vertex order

        all_joints = []
        all_weights = []

        # Iterate through all meshes and primitives
        for mesh in gltf.meshes:
            for primitive in mesh.primitives:
                # Check if this primitive has skin weights
                has_joints = hasattr(primitive.attributes, 'JOINTS_0') and primitive.attributes.JOINTS_0 is not None
                has_weights = hasattr(primitive.attributes, 'WEIGHTS_0') and primitive.attributes.WEIGHTS_0 is not None

                if has_joints and has_weights:
                    # Load joint indices
                    joints_accessor = gltf.accessors[primitive.attributes.JOINTS_0]
                    joints_data = RiggedMeshLoader._read_accessor(gltf, joints_accessor)

                    # Load weights
                    weights_accessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
                    weights_data = RiggedMeshLoader._read_accessor(gltf, weights_accessor)

                    all_joints.append(joints_data)
                    all_weights.append(weights_data)

        if not all_joints:
            print("Warning: No skin weights found in any mesh primitive")
            return

        # Concatenate all weights
        result.skin_indices = np.vstack(all_joints).astype(np.int32)
        result.skin_weights = np.vstack(all_weights).astype(np.float32)

        # Check if we have weights for all vertices
        if len(result.skin_weights) != len(result.vertices):
            print(f"Warning: Vertex count mismatch - {len(result.vertices)} verts but {len(result.skin_weights)} weights")
            print(f"Padding weights to match vertex count...")

            # Pad with identity (all weight to root bone)
            n_missing = len(result.vertices) - len(result.skin_weights)
            padding_weights = np.zeros((n_missing, 4), dtype=np.float32)
            padding_weights[:, 0] = 1.0  # Full weight to bone 0 (root)
            padding_indices = np.zeros((n_missing, 4), dtype=np.int32)

            result.skin_weights = np.vstack([result.skin_weights, padding_weights])
            result.skin_indices = np.vstack([result.skin_indices, padding_indices])

        # Normalize weights (ensure sum to 1.0)
        weight_sums = result.skin_weights.sum(axis=1, keepdims=True)
        result.skin_weights = np.divide(
            result.skin_weights,
            weight_sums,
            out=np.zeros_like(result.skin_weights),
            where=weight_sums != 0
        )

    @staticmethod
    def _read_accessor(gltf: pygltflib.GLTF2, accessor: pygltflib.Accessor) -> np.ndarray:
        """Read data from a GLTF accessor"""
        buffer_view = gltf.bufferViews[accessor.bufferView]

        # Get binary data - for GLB files, data is embedded
        # Use pygltflib's binary_blob() method
        data = gltf.binary_blob()

        # Determine data type
        component_type_map = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        dtype = component_type_map[accessor.componentType]

        # Determine component count
        type_size_map = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT2': 4,
            'MAT3': 9,
            'MAT4': 16,
        }
        components = type_size_map[accessor.type]

        # Extract data
        offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        count = accessor.count

        # Read and reshape
        byte_length = count * components * np.dtype(dtype).itemsize
        array_data = np.frombuffer(data[offset:offset + byte_length], dtype=dtype)

        if components > 1:
            array_data = array_data.reshape(count, components)

        return array_data

    @staticmethod
    def print_skeleton_info(mesh: RiggedMesh):
        """Print skeleton hierarchy for debugging"""
        if not mesh.bones:
            print("No skeleton found")
            return

        print(f"\n=== Skeleton: {len(mesh.bones)} bones ===")
        print(f"Root bone: {mesh.bones[mesh.root_bone_idx].name if mesh.root_bone_idx is not None else 'None'}")

        def print_bone_tree(bone_idx: int, indent: int = 0):
            bone = mesh.bones[bone_idx]
            print("  " * indent + f"├─ {bone.name} (idx: {bone.index})")
            for child_idx in bone.children_idx:
                print_bone_tree(child_idx, indent + 1)

        if mesh.root_bone_idx is not None:
            print_bone_tree(mesh.root_bone_idx)


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        glb_path = sys.argv[1]
    else:
        glb_path = "rigged_mesh/CAUCASIAN MAN.glb"

    print(f"Loading: {glb_path}")
    mesh = RiggedMeshLoader.load(glb_path)

    print(f"\n=== Geometry ===")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Bounds: {mesh.bounds}")
    print(f"Center: {mesh.center}")

    RiggedMeshLoader.print_skeleton_info(mesh)

    if mesh.skin_weights is not None:
        print(f"\n=== Skin Weights ===")
        print(f"Shape: {mesh.skin_weights.shape}")
        print(f"Max influences per vertex: {(mesh.skin_weights > 0).sum(axis=1).max()}")
        print(f"Avg influences per vertex: {(mesh.skin_weights > 0).sum(axis=1).mean():.2f}")
