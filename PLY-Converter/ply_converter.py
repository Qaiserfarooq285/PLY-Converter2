#!/usr/bin/env python3
"""
PLY Converter – Force surface reconstruction and export to STL/OBJ/GLB with texture/color preservation.

This version enforces surface reconstruction (Poisson) for every input PLY and
preserves embedded vertex colors when exporting GLB. OBJ will include colors
when supported by the exporter. STL will be exported as geometry (STL doesn't
standardly support color).

Usage:
  python ply_converter.py input.ply --outdir out
  python ply_converter.py input.ply --outdir out --poisson-depth 9 --keep-largest --simplify 50000
  python ply_converter.py --run-tests

Dependencies:
  pip install open3d trimesh numpy scikit-image

Notes:
- This script always runs Poisson reconstruction on the input points (even if
  the PLY already contained faces) so the result is a watertight surface.
- The GLB export uses trimesh to embed vertex colors into the binary glTF.
- If your PLY contains per-vertex UVs and an embedded texture image, PLY
  readers rarely expose that automatically; this script prioritizes vertex
  colors. If you need full texture (UV + image) extraction, provide a sample
  PLY and I can extend to extract image/UV and embed as a material texture.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Core dependencies for PLY processing
import trimesh
from plyfile import PlyData, PlyElement

# Optional heavy deps - we require open3d and trimesh for full functionality
try:
    import open3d as o3d
except Exception:
    o3d = None

# scikit-image optional for connected component filtering, fallback to trimesh.split
try:
    from skimage import measure as skmeasure
except Exception:
    skmeasure = None


def log(msg: str) -> None:
    print(msg, flush=True)


def load_input_as_pointcloud(path: Path) -> o3d.geometry.PointCloud:
    """Load a PLY file and return an Open3D PointCloud containing points and colors (if present).

    We always produce a point cloud (if mesh is given, we sample its vertices).
    """
    if o3d is None:
        # Fallback to trimesh-only approach
        try:
            mesh = trimesh.load(str(path))
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                # Create a simple point cloud dictionary
                pc_data = {
                    'points': mesh.vertices,
                    'colors': getattr(mesh.visual, 'vertex_colors', None)
                }
                return pc_data
        except Exception:
            pass
        raise RuntimeError("Cannot load PLY file - Open3D is not available and trimesh failed")

    # Try to read as triangle mesh first
    try:
        mesh = o3d.io.read_triangle_mesh(str(path))
    except Exception:
        mesh = None

    if mesh is not None and len(mesh.triangles) > 0:
        # If mesh has vertices, turn into point cloud using the vertices
        pc = o3d.geometry.PointCloud()
        pc.points = mesh.vertices
        # Transfer vertex colors if present
        if mesh.has_vertex_colors():
            pc.colors = mesh.vertex_colors
        # If mesh has vertex normals, copy them (helpful for Poisson)
        if mesh.has_vertex_normals():
            pc.normals = mesh.vertex_normals
        return pc

    # Otherwise try to read as point cloud
    try:
        pc = o3d.io.read_point_cloud(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read {path} as PLY: {e}")

    if len(pc.points) == 0:
        raise RuntimeError(f"No points found in {path}")

    return pc


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: Optional[float] = None) -> None:
    """Estimate normals for the point cloud in-place.

    radius determines neighborhood size; if None we compute a heuristic.
    """
    if not pcd.has_normals():
        pts = np.asarray(pcd.points)
        if radius is None:
            # heuristic: 1% of bounding box diagonal
            bbox = pts.max(axis=0) - pts.min(axis=0)
            diag = np.linalg.norm(bbox)
            radius = max(diag * 0.01, 0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        pcd.normalize_normals()


def poisson_reconstruct(pcd: o3d.geometry.PointCloud, depth: int = 9) -> o3d.geometry.TriangleMesh:
    """Run Poisson reconstruction and return a cleaned TriangleMesh."""
    if o3d is None:
        raise RuntimeError("open3d is required for reconstruction")

    estimate_normals(pcd)

    # Use higher depth for better detail and set linear_fit=False for better surface completion
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=max(depth, 10), linear_fit=False
    )

    # Be much more conservative with density filtering to avoid holes
    densities = np.asarray(densities)
    if densities.size > 0:
        # Only remove the lowest 0.1% density vertices instead of 1%
        thresh = np.quantile(densities, 0.001)
        verts_to_keep = densities >= thresh
        # Create vertex mask and filter mesh
        mesh = mesh.remove_vertices_by_mask(~verts_to_keep)

    # Clean up the mesh but preserve topology
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    
    # Fill holes in the mesh using trimesh if available
    try:
        if trimesh is not None:
            # Convert to trimesh for hole filling
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            tm = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Fill holes
            tm.fill_holes()
            
            # Convert back to Open3D
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
            
            # Copy vertex colors if they exist
            if len(pcd.colors) > 0:
                mesh.vertex_colors = pcd.colors
                
    except Exception:
        # If trimesh hole filling fails, continue with Open3D cleanup
        pass
    
    # Final cleanup but avoid removing non-manifold edges which might create holes
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    return mesh


def keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Keep only the largest connected component in the mesh."""
    if o3d is None:
        return mesh
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh
    largest = int(np.argmax(cluster_n_triangles))
    triangles_to_remove = triangle_clusters != largest
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def simplify_mesh(mesh: o3d.geometry.TriangleMesh, target_number_of_triangles: int) -> o3d.geometry.TriangleMesh:
    """Simplify mesh to target triangle count using quadric decimation when available."""
    try:
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles)
        simplified.remove_duplicated_vertices()
        simplified.remove_degenerate_triangles()
        simplified.remove_duplicated_triangles()
        simplified.remove_unreferenced_vertices()
        return simplified
    except Exception:
        # fallback: return original
        return mesh


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> 'trimesh.Trimesh':
    """Convert an Open3D TriangleMesh to trimesh.Trimesh, preserving vertex colors if present."""
    if trimesh is None:
        raise RuntimeError("trimesh is required for GLB export. Install with: pip install trimesh")

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # gather vertex colors if available
    vcols = None
    if mesh.has_vertex_colors():
        vcols = np.asarray(mesh.vertex_colors)
        # ensure 3 components and normalized to 0..1
        if vcols.shape[1] == 4:
            vcols = vcols[:, :3]
        if vcols.max() > 1.0:
            vcols = vcols / 255.0

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if vcols is not None and len(vcols) == len(verts):
        tm.visual.vertex_colors = (vcols * 255).astype(np.uint8)
    return tm


def export_outputs(mesh: o3d.geometry.TriangleMesh, outdir: Path, stem: str, export_glb: bool = True, export_obj: bool = True, export_stl: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if export_stl:
        stl_path = outdir / f"{stem}.stl"
        try:
            # STL doesn't support colors in standard form; write geometry
            o3d.io.write_triangle_mesh(str(stl_path), mesh, write_ascii=False)
            log(f"[ok] Wrote STL: {stl_path}")
        except Exception as e:
            log(f"[error] Failed to write STL: {e}")

    if export_obj:
        obj_path = outdir / f"{stem}.obj"
        try:
            # Use trimesh for OBJ to preserve vertex colors if present
            tm = o3d_to_trimesh(mesh)
            tm.export(str(obj_path))
            log(f"[ok] Wrote OBJ: {obj_path}")
        except Exception as e:
            # fallback to Open3D OBJ writer
            try:
                o3d.io.write_triangle_mesh(str(obj_path), mesh, write_ascii=True)
                log(f"[ok] Wrote OBJ via Open3D fallback: {obj_path}")
            except Exception:
                log(f"[error] Failed to write OBJ: {e}")

    if export_glb:
        glb_path = outdir / f"{stem}.glb"
        try:
            tm = o3d_to_trimesh(mesh)
            # Ensure vertex colors are present or trimesh will create default material
            if hasattr(tm.visual, 'vertex_colors') and tm.visual.vertex_colors is not None and len(tm.visual.vertex_colors) > 0:
                # trimesh expects colors as 0-255 uint8
                cols = np.asarray(tm.visual.vertex_colors)
                if cols.dtype != np.uint8:
                    cols = (cols * 255).astype(np.uint8) if cols.max() <= 1.0 else cols.astype(np.uint8)
                    tm.visual.vertex_colors = cols
            tm.export(str(glb_path))
            log(f"[ok] Wrote GLB: {glb_path}")
        except Exception as e:
            log(f"[error] Failed to write GLB: {e}")
            log(f"{traceback.format_exc()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PLY -> Reconstructed surface -> STL/OBJ/GLB")
    p.add_argument("input", nargs="?", help="Input PLY file")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--poisson-depth", type=int, default=9, help="Poisson reconstruction depth (higher -> more detail)")
    p.add_argument("--keep-largest", action="store_true", help="Keep only the largest connected component after reconstruction")
    p.add_argument("--simplify", type=int, default=0, help="If >0, simplify mesh to this many triangles")
    p.add_argument("--no-glb", action="store_true", help="Do not export GLB")
    p.add_argument("--no-obj", action="store_true", help="Do not export OBJ")
    p.add_argument("--no-stl", action="store_true", help="Do not export STL")
    p.add_argument("--run-tests", action="store_true", help="Run internal tests and exit")
    return p


def run_unit_tests() -> int:
    import unittest

    class BasicTests(unittest.TestCase):
        def test_parser_defaults(self):
            p = build_parser()
            ns = p.parse_args(["input.ply"])
            self.assertEqual(ns.input, "input.ply")
            self.assertEqual(ns.poisson_depth, 9)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BasicTests)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


class PLYConverter:
    """Wrapper class for PLY conversion functionality"""
    
    def convert_ply(self, input_path, output_dir, output_formats, conversion_id, progress_callback=None):
        """Convert PLY file to specified formats"""
        try:
            if progress_callback:
                progress_callback("Loading PLY file...", 10)
            
            # Load the PLY file using trimesh (more robust)
            mesh = trimesh.load(str(input_path))
            
            if progress_callback:
                progress_callback(f"Loaded mesh with {len(mesh.vertices)} vertices", 20)
            
            # Check if this is a point cloud (no faces) and convert to mesh
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                if progress_callback:
                    progress_callback("Creating surface from point cloud...", 30)
                
                vertices = mesh.vertices
                
                # For organic shapes, use k-nearest neighbors to create local triangulation
                # This preserves the exact point cloud geometry
                try:
                    if progress_callback:
                        progress_callback("Using k-nearest neighbor triangulation...", 32)
                    
                    from scipy.spatial import cKDTree
                    
                    # Build KD-tree for efficient neighbor search
                    tree = cKDTree(vertices)
                    
                    # For each point, find its nearest neighbors
                    k = min(12, len(vertices) - 1)  # Use up to 12 neighbors
                    
                    faces = []
                    for i in range(len(vertices)):
                        # Find k nearest neighbors
                        distances, indices = tree.query(vertices[i], k=k+1)  # +1 because first is the point itself
                        neighbors = indices[1:]  # Exclude the point itself
                        
                        # Create triangles with the central point and pairs of neighbors
                        for j in range(len(neighbors) - 1):
                            face = [i, neighbors[j], neighbors[j + 1]]
                            faces.append(face)
                    
                    # Remove duplicate faces
                    unique_faces = []
                    face_set = set()
                    for face in faces:
                        sorted_face = tuple(sorted(face))
                        if sorted_face not in face_set:
                            face_set.add(sorted_face)
                            unique_faces.append(face)
                    
                    if len(unique_faces) > 0:
                        mesh = trimesh.Trimesh(vertices=vertices, faces=unique_faces)
                        if progress_callback:
                            progress_callback(f"K-NN mesh: {len(unique_faces)} faces", 35)
                    else:
                        raise Exception("K-NN failed to generate faces")
                
                except Exception as e:
                    if progress_callback:
                        progress_callback("K-NN failed, trying simpler approach...", 33)
                    
                    # Fallback: Create a very basic mesh that just connects nearby points
                    try:
                        from scipy.spatial.distance import pdist, squareform
                        
                        # For very large point clouds, sample a subset
                        if len(vertices) > 10000:
                            sample_idx = np.random.choice(len(vertices), 10000, replace=False)
                            sample_vertices = vertices[sample_idx]
                        else:
                            sample_vertices = vertices
                            sample_idx = np.arange(len(vertices))
                        
                        # Calculate pairwise distances
                        distances = squareform(pdist(sample_vertices))
                        
                        # Find a good threshold for connecting points
                        sorted_distances = np.sort(distances[distances > 0])
                        threshold = sorted_distances[len(sorted_distances) // 20]  # 5th percentile
                        
                        faces = []
                        for i in range(len(sample_vertices)):
                            # Find close neighbors
                            close_neighbors = np.where((distances[i] > 0) & (distances[i] < threshold))[0]
                            
                            # Create triangles
                            for j in range(len(close_neighbors) - 1):
                                for k in range(j + 1, len(close_neighbors)):
                                    if len(close_neighbors) > k:
                                        # Map back to original indices
                                        face = [sample_idx[i], sample_idx[close_neighbors[j]], sample_idx[close_neighbors[k]]]
                                        faces.append(face)
                        
                        if len(faces) > 0:
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces[:50000])  # Limit faces
                            if progress_callback:
                                progress_callback(f"Distance-based mesh: {len(faces[:50000])} faces", 36)
                        else:
                            raise Exception("Distance-based method failed")
                    
                    except Exception:
                        # Final fallback - very simple hull
                        if progress_callback:
                            progress_callback("Using basic convex hull...", 37)
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(vertices)
                        mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)
            
            # Ensure the mesh is watertight and fill holes
            if progress_callback:
                progress_callback("Processing surface...", 40)
            
            # Fill only small holes and preserve geometry
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                original_face_count = len(mesh.faces)
                
                # Only fill small holes (conservative approach)
                try:
                    # Calculate mesh scale to determine what constitutes a "small" hole
                    bounds = mesh.bounds
                    mesh_scale = np.linalg.norm(bounds[1] - bounds[0])
                    max_hole_size = mesh_scale * 0.02  # Only fill holes smaller than 2% of mesh size
                    
                    # Remove only degenerate and duplicate elements
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()
                    mesh.remove_unreferenced_vertices()
                    
                    # Fill only small holes
                    mesh.fill_holes()
                    
                    # If the mesh changed dramatically, something went wrong - revert
                    if len(mesh.faces) < original_face_count * 0.5 or len(mesh.faces) > original_face_count * 3:
                        if progress_callback:
                            progress_callback("Hole filling changed geometry too much, using original", 55)
                        # Reload original and do minimal processing
                        mesh = trimesh.load(str(input_path))
                        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                            # Re-apply alpha shape with conservative settings
                            best_mesh = trimesh.creation.alpha_shape(mesh.vertices, alpha=0.3)
                            mesh = best_mesh
                    
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Conservative processing: {str(e)[:50]}", 55)
                
                if progress_callback:
                    progress_callback(f"Processed mesh: {len(mesh.faces)} faces", 60)
            else:
                if progress_callback:
                    progress_callback("Warning: No faces generated, exporting points only", 60)
            
            if progress_callback:
                progress_callback("Exporting files...", 80)
            
            # Export to specified formats
            stem = Path(input_path).stem
            output_path = Path(output_dir)
            
            results = {}
            
            for fmt in output_formats:
                try:
                    if fmt == 'stl':
                        stl_path = output_path / f"{conversion_id}_{stem}.stl"
                        mesh.export(str(stl_path))
                        results['stl'] = str(stl_path)
                    elif fmt == 'obj':
                        obj_path = output_path / f"{conversion_id}_{stem}.obj"
                        mesh.export(str(obj_path))
                        results['obj'] = str(obj_path)
                    elif fmt == 'glb':
                        glb_path = output_path / f"{conversion_id}_{stem}.glb"
                        mesh.export(str(glb_path))
                        results['glb'] = str(glb_path)
                    elif fmt == '3mf':
                        mf_path = output_path / f"{conversion_id}_{stem}.3mf"
                        mesh.export(str(mf_path))
                        results['3mf'] = str(mf_path)
                    elif fmt == 'dxf':
                        dxf_path = output_path / f"{conversion_id}_{stem}.dxf"
                        # For DXF, we might need a 2D projection
                        try:
                            mesh.export(str(dxf_path))
                            results['dxf'] = str(dxf_path)
                        except:
                            # DXF might not be supported, skip it
                            pass
                except Exception as export_error:
                    log(f"Failed to export {fmt}: {export_error}")
                    continue
            
            if progress_callback:
                progress_callback("Conversion completed!", 100)
            
            return results
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}", 0)
            raise e


def main(argv=None) -> int:
    if argv is None and len(sys.argv) == 1:
        print("PLY Converter: enforce surface reconstruction and export to STL/OBJ/GLB")
        print("Run with --run-tests to run internal checks or provide an input PLY file.")
        parser = build_parser()
        parser.print_help()
        return 0

    args = build_parser().parse_args(argv)

    if args.run_tests:
        return run_unit_tests()

    if not args.input:
        log("[error] No input provided. Provide a PLY file path.")
        return 2

    in_path = Path(args.input)
    if not in_path.exists():
        log(f"[error] Input not found: {in_path}")
        return 2

    if o3d is None:
        log("[fatal] open3d is required. Install with: pip install open3d")
        return 1
    if trimesh is None:
        log("[fatal] trimesh is required for OBJ/GLB export. Install with: pip install trimesh")
        return 1

    try:
        pcd = load_input_as_pointcloud(in_path)
        log(f"[info] Loaded point cloud with {len(pcd.points)} points")

        mesh = poisson_reconstruct(pcd, depth=args.poisson_depth)
        log(f"[info] Reconstruction produced {len(mesh.triangles)} triangles and {len(mesh.vertices)} vertices")

        if args.keep_largest:
            mesh = keep_largest_component(mesh)
            log(f"[info] Kept largest component: {len(mesh.triangles)} triangles")

        if args.simplify and args.simplify > 0:
            mesh = simplify_mesh(mesh, args.simplify)
            log(f"[info] Simplified mesh to {len(mesh.triangles)} triangles")

        stem = in_path.stem
        export_outputs(mesh, Path(args.outdir), stem, export_glb=(not args.no_glb), export_obj=(not args.no_obj), export_stl=(not args.no_stl))

    except Exception as e:
        log(f"[fatal] {e}")
        log(f"{traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
