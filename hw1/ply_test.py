import numpy as np
import ply
import filecmp

if __name__ == "__main__":
    ply_path_read = "./data/triangle_sample"
    ply_path_write = "./data/triangle_sample_copy"

    # Data:
    points = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    colors = np.array([[0, 0, 155], [0, 0, 155], [0, 0, 155]])
    triangles = np.array([[2, 1, 0]])

    # initialize with file path
    sample1 = ply.Ply(ply_path=ply_path_read + ".ply")
    sample1.write(ply_path_write + "_1.ply")

    # initialize with data
    sample2 = ply.Ply(points=points, normals=normals, colors=colors, triangles=triangles)
    sample2.write(ply_path_write + "_2.ply")

    # read from file
    sample3 = ply.Ply()
    sample3.read(ply_path_read + ".ply")
    sample3.write(ply_path_write + "_3.ply")

    # three ways should generate equal solutions
    assert filecmp.cmp(ply_path_write + "_1.ply", ply_path_write + "_2.ply", shallow=False)
    assert filecmp.cmp(ply_path_write + "_1.ply", ply_path_write + "_3.ply", shallow=False)
    assert filecmp.cmp(ply_path_write + "_2.ply", ply_path_write + "_3.ply", shallow=False)
