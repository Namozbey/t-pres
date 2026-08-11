from editing.alignpc import align_pcs_sim3
from editing.mesh2pc import sample_mesh_surface


def run_alignment(mesh, pc):
    sample_mesh_surface(mesh_path=mesh)
    M = align_pcs_sim3(source_ply=pc, target_ply="state/generated.ply")
    return M

if __name__ == "__main__":
    M = run_alignment("input/gen_mesh.glb","input/bg_free_full_pc2.ply")
    print(M)