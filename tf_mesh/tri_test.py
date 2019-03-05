from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes

from tf_mesh import tri


def tetrahedron():
  vertices = tf.random.normal(shape=(4, 3))
  faces = tf.constant(
    [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=tf.int64)
  return vertices, faces


class TriTest(tf.test.TestCase):
  @run_in_graph_and_eager_modes
  def test_triangulated_faces(self):
    face_values = tf.range(9, dtype=tf.int64)
    face_lengths = tf.constant((5, 4), dtype=tf.int64)
    tri_faces = tri.triangulated_faces(face_values, face_lengths)
    expected = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [5, 6, 7],
        [5, 7, 8],
    ])

    np.testing.assert_equal(self.evaluate(tri_faces), expected)

  @run_in_graph_and_eager_modes
  def test_compute_face_normals(self):
    faces = tf.constant([[0, 1, 2], [0, 2, 1]])
    vertices = tf.constant([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=tf.float32)
    normals = tri.compute_face_normals(vertices, faces)
    np.testing.assert_allclose(
      self.evaluate(normals), [[1, 0, 0], [-1, 0, 0]])
    vertices = 2*vertices
    normals = tri.compute_face_normals(vertices, faces)
    np.testing.assert_allclose(
      self.evaluate(normals), [[4, 0, 0], [-4, 0, 0]])

  @run_in_graph_and_eager_modes
  def test_compute_vertex_normals(self):
    faces = tf.constant([[0, 1, 2], [0, 3, 1]])
    vertices = tf.constant([
      [0, 0, 0],
      [0, 1, 0],
      [0, 1, 1],
      [1, 1, 0]], dtype=tf.float32)
    n_vertices = tf.shape(vertices)[0]
    face_normals = tri.compute_face_normals(vertices, faces)
    vertex_normals = tri.compute_vertex_normals(
      faces, face_normals, n_vertices)
    np.testing.assert_equal(vertex_normals.shape, vertices.shape)
    np.testing.assert_equal(
      self.evaluate(vertex_normals),
      np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
      ], dtype=np.float32))

  @run_in_graph_and_eager_modes
  def test_sample_faces(self):
      x_offset = 10
      vertices = np.array([
          [x_offset, 0, 0],
          [x_offset, 0, 2],
          [x_offset, 1, 2],
      ], dtype=np.float32)
      faces = np.array([
          [0, 1, 2]
      ], dtype=np.int64)
      n_points = 1000
      face_lengths, barycentric_coords, points, point_normals = \
        tri.sample_faces(vertices, faces, n_points)
      points = self.evaluate(points)
      face_lengths = self.evaluate(face_lengths)
      np.testing.assert_equal(points.shape, (n_points, 3))
      np.testing.assert_equal(face_lengths, (n_points,))
      x, y, z = points.T
      np.testing.assert_allclose(x, x_offset)
      self.assertTrue(np.all(z >= 2*y))
      self.assertTrue(np.all(y >= 0))
      self.assertTrue(np.all(z <= 2))



if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  unittest.main()
