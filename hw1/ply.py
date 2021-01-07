import numpy as np
import os
import warnings


class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point.. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point.. Defaults to None.
        """
        super().__init__()

        self.triangles = triangles
        self.points = points
        self.normals = normals
        self.colors = colors
        self.validate_gate()

        if ply_path:
            return self.read(ply_path)

        return

    def validate_gate(self):
        """Check whether the attributes are good.

        Raises:
            ValueError: If vertices are not 3-D if any.
            ValueError: If vertices' dimensionality cannot match if any.
            ValueError: If triangles don't have 3 vertices if any.
        """

        if isinstance(self.points, np.ndarray) or isinstance(self.normals, np.ndarray) or isinstance(self.colors, np.ndarray):
            if not (self.points.shape[1] == self.normals.shape[1] == self.colors.shape[1] == 3):
                raise ValueError("Vertices are not 3-D")
            elif not (self.points.shape[0] == self.normals.shape[0] == self.colors.shape[0]):
                raise ValueError("Vertices' dimensionality cannot match")

        if isinstance(self.triangles, np.ndarray):
            if not self.triangles.shape[1] == 3:
                raise ValueError("Triangles don't have 3 vertices")

        return

    def clear(self):
        """Clear the attributes' values."""
        self.triangles = None
        self.points = None
        self.normals = None
        self.colors = None

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """

        self.validate_gate()
        if os.path.exists(ply_path):
            warnings.warn("Caution: This will cause an overwrite!")

        with open(ply_path, "w+") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            if isinstance(self.points, np.ndarray):
                f.write("element vertex " + str(self.points.shape[0]) + "\n")   # number of points
                f.write("property float x\n")  # first entry of a point.
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property float nx\n")  # first normal component of the point.
                f.write("property float ny\n")
                f.write("property float nz\n")
                f.write("property uchar red\n")  # red component of the point color.
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

                if isinstance(self.triangles, np.ndarray):
                    f.write("element face " + str(self.triangles.shape[0]) + "\n") # number of faces
                    f.write("property list uchar int vertex_index\n")

            f.write("end_header\n")

            if isinstance(self.points, np.ndarray):
                for i in range(self.points.shape[0]):
                    f.write(str(self.points[i, 0]) + " " + str(self.points[i, 1]) + " " + str(self.points[i, 2]) + " ")
                    f.write(str(self.normals[i, 0]) + " " + str(self.normals[i, 1]) + " " + str(self.normals[i, 2]) + " ")
                    f.write(str(int(self.colors[i, 0])) + " " + str(int(self.colors[i, 1])) + " " + str(int(self.colors[i, 2])))
                    # if i == 0:
                    #     f.write(" # x y z nx ny nz red green blue\n")
                    # else:
                    #     f.write("\n")
                    f.write("\n")

                if isinstance(self.triangles, np.ndarray):
                    for i in range(self.triangles.shape[0]):
                        f.write(str(3) + " " + str(int(self.triangles[i, 0])) + " " + str(int(self.triangles[i, 1])) + " " + str(int(self.triangles[i, 2])))
                        # if i == 0:
                        #     f.write(" # (number of vertices in the face) (point index 1) ...\n")
                        # else:
                        #     f.write("\n")
                        f.write("\n")

        return

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """

        self.clear()
        if not os.path.exists(ply_path):
            raise ValueError(f"Path - {ply_path} - doesn't exist")

        with open(ply_path, "r+") as f:
            ver_num, tri_num = 0, 0
            data_field = False
            points, normals, colors, triangles = [], [], [], []

            for lines in f.readlines():
                tokens = lines.split()
                if len(tokens) == 0:
                    continue

                if data_field == False:
                    if tokens[0] == "element":
                        if tokens[1] == "vertex":
                            ver_num = int(tokens[2])
                        elif tokens[1] == "face":
                            tri_num = int(tokens[2])
                        else:
                            raise TypeError("Element cannot found")
                    elif tokens[0] == "end_header":
                        data_field = True
                        continue
                else:
                    if ver_num != 0:
                        points.append([float(t) for t in tokens[0:3]])
                        normals.append([float(t) for t in tokens[3:6]])
                        colors.append([float(t) for t in tokens[6:9]])
                        ver_num -= 1
                    elif tri_num != 0:
                        triangles.append([float(t) for t in tokens[1:4]])
                        tri_num -= 1

            self.points = np.array(points)
            self.normals = np.array(normals)
            self.colors = np.array(colors)
            self.triangles = np.array(triangles)

        return self.validate_gate()
