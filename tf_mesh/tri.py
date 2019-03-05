"""Utilities for manipulating triangular meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def normalized(x, axis=-1):
  return x / tf.norm(x, ord=2, axis=axis, keepdims=True)


def _split_n(x, n, axis=0):
  n = tf.cast(n, dtype=tf.int32)
  return tf.split(x, (n, tf.shape(x, out_type=tf.int32)[axis]-n), axis=axis)


def _dequeue(x, axis=0):
  leading, rest = _split_n(x, 1, axis=axis)
  return tf.squeeze(leading, axis=axis), rest


def triangulated_faces(face_values, face_lengths):
  """
  Convert ragged representation to (n_faces, 3) tensor of triangular faces.

  Args:
    face_values: rank 1 int tensor of vertex indices of all faces concatenated
    face_lengths: rank 1 int tensor of the length of each face

  Returns:
    tri_faces: (n_tri_faces, 3) regular tensor of triangular faces.
  """
  n = tf.size(face_lengths, out_type=tf.int32)

  def tri_faces(face):
    f1 = face[1:-1]
    f2 = face[2:]
    f0 = tf.fill(tf.shape(f1), face[0])
    return tf.stack((f0, f1, f2), axis=-1)

  def cond(face_values, face_lengths, i, acc):
    return i < n

  def body(face_values, face_lengths, i, acc):
    next_count, face_lengths = _dequeue(face_lengths)
    next_face, face_values = _split_n(face_values, next_count)
    faces = tri_faces(next_face)
    acc = acc.write(i, faces)
    return face_values, face_lengths, i+1, acc

  element_shape = (None, 3)
  acc = tf.TensorArray(
    dtype=tf.int64, size=n, dynamic_size=False,
    element_shape=element_shape, infer_shape=False)

  acc = tf.while_loop(
    cond, body,
    [face_values, face_lengths, tf.zeros((), dtype=n.dtype), acc],
    shape_invariants=[
      tf.TensorShape((None,)),
      tf.TensorShape((None,)),
      tf.TensorShape(()),
      tf.TensorShape(None),
    ]
    )[-1]
  return acc.concat()


def compute_face_normals(vertices, faces):
  tris = tf.gather(vertices, faces)
  a, b, c = tf.unstack(tris, axis=-2)
  face_normals = tf.cross(b - a, c - a)
  return face_normals


def compute_vertex_normals(faces, face_normals, n_vertices=None):
  if n_vertices is None:
    n_vertices = tf.reduce_max(faces)
  face_normals = tf.tile(tf.expand_dims(face_normals, -2), (1, 3, 1))
  faces = tf.expand_dims(faces, -1)
  shape = tf.stack((tf.cast(n_vertices, faces.dtype), 3), axis=0)
  return tf.scatter_nd(faces, face_normals, shape=shape)


def sample_barycentric_coordinates(n_samples, dtype=tf.float32):
  """Get random barycentric coordinates for uniform sampling.

  Args:
    n_samples: number of samples to take
    dtype: float dtype of returned array

  Returns:
    `(n_samples, 3)` float array of barycentric coordinates corresponding to
      points uniformly sampled across a face.
  """
  # see e.g. https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
  r0, r1 = tf.unstack(
      tf.random.uniform(shape=(2, n_samples), dtype=dtype), axis=0)
  root_r0 = tf.sqrt(r0)
  return tf.stack([(1 - root_r0), root_r0 * (1 - r1), root_r0 * r1], axis=-1)


def _barycentric_interpolate(values, barycentric_coords):
  assert(len(values.shape) == len(barycentric_coords.shape) + 1)
  assert(values.shape[-2] == barycentric_coords.shape[-1])
  return tf.reduce_sum(
    values * tf.expand_dims(barycentric_coords, axis=-1), axis=-2)


def _categorical_barycentric_interpolate(
      vertex_values, faces, face_lengths, barycentric_coords):

    n = tf.shape(faces, out_type=tf.int32)[0]

    def cond(face, face_lengths, barycentric_coords, i, acc):
      return i < n

    def body(faces, face_lengths, barycentric_coords, i, acc):
      next_face, faces = _dequeue(faces)
      next_count, face_lengths = _dequeue(face_lengths)
      next_coords, barycentric_coords = _split_n(
          barycentric_coords, next_count)
      next_acc = _barycentric_interpolate(
        tf.expand_dims(tf.gather(vertex_values, next_face), axis=0),
        next_coords)
      acc = acc.write(i, next_acc)
      return faces, face_lengths, barycentric_coords, i+1, acc

    n_out = vertex_values.shape[-1]
    acc0 = tf.TensorArray(
      dtype=vertex_values.dtype, size=n, dynamic_size=False,
      element_shape=(None, n_out), infer_shape=False)

    i0 = tf.zeros((), dtype=n.dtype)
    acc = tf.while_loop(
      cond, body,
      [faces, face_lengths, barycentric_coords, i0, acc0],
      shape_invariants=[
        tf.TensorShape((None, 3)),
        tf.TensorShape((None,)),
        tf.TensorShape((None, 3)),
        tf.TensorShape(()),
        tf.TensorShape(None),
      ])[-1]
    return acc.concat()


def sample_faces(vertices, faces, n_total, include_normals=True):
  """Sample points uniformly over a triangular mesh.

  Sampling is based on the area of each

  Args:
    vertices: (nv, 3) float array of 3D indices
    faces: (nf, 3) non-negative int array of vertex indices. All entries must be
        less than nv.
    n_total: number of points to sample
    include_normals: if True, also computes normals by interpolating vertex
      normals

  Returns:
    face_lengths: `(nf,)` int array of counts of samples from each face, summing
      to `n_total`
    barycentric_coords: `(n_total, 3)` barycentric coordinates for each point
    points: `(n_total, 3)` cartesian coordinates of each point
    point_normals: (if `include_normals`) `(n_total, 3)` normals or each point,
      calculated as the interpolation of vertex normals.
  """
  vertices = tf.convert_to_tensor(vertices)
  dtype = faces.dtype
  if n_total == 0:
    return tf.zeros(shape=(0, 3), dtype=vertices.dtype)
  if len(faces) == 0:
    raise ValueError('Cannot sample points from zero faces.')
  face_normals = compute_face_normals(vertices, faces)
  areas = tf.linalg.norm(face_normals, ord=2, axis=-1)
  n_faces = tf.shape(faces, out_type=dtype)[0]
  face_samples = tf.squeeze(tf.random.categorical(
    tf.expand_dims(tf.log(areas), axis=0), n_total, dtype=dtype), axis=0)
  indices = face_samples
  indices = tf.expand_dims(indices, axis=-1)
  updates = tf.ones((n_total,), dtype=dtype)
  face_lengths = tf.scatter_nd(
    indices, updates, shape=tf.expand_dims(n_faces, axis=0))
  barycentric_coords = sample_barycentric_coordinates(n_total)
  points = _categorical_barycentric_interpolate(
    vertices, faces, face_lengths, barycentric_coords)
  if include_normals:
    vertex_normals = compute_vertex_normals(
      faces, face_normals, n_vertices=tf.shape(vertices)[0])
    # vertex_normals = normalized(vertex_normals)
    point_normals = _categorical_barycentric_interpolate(
      vertex_normals, faces, face_lengths, barycentric_coords)
    point_normals = normalized(point_normals)
    return face_lengths, barycentric_coords, points, point_normals
  else:
    return face_lengths, barycentric_coords, points
