from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def normalize(x, axis=-1):
  x /= np.linalg.norm(x, ord=2, axis=axis, keepdims=True)


def triangulated_faces(face_values, face_lengths):
  n = np.sum(face_lengths) - 2 * len(face_lengths)
  out = np.empty((n, 3), dtype=face_values.dtype)
  faces = np.split(face_values, np.cumsum(face_lengths[:-1]))
  assert(len(faces[-1]) == face_lengths[-1])

  start = 0
  for face in faces:
    end = start + len(face) - 2
    out[start:end, 0] = face[0]
    out[start:end, 1] = face[1:-1]
    out[start:end, 2] = face[2:]
    start = end
  return out


def compute_face_normals(vertices, faces):
  tris = vertices[faces]
  a, b, c = np.split(tris, (1, 2), axis=-2)  # pylint: disable=unbalanced-tuple-unpacking
  face_normals = np.cross(b - a, c - a)
  return np.squeeze(face_normals, axis=-2)


def compute_vertex_normals(faces, face_normals, n_vertices=None):
  if n_vertices is None:
    n_vertices = np.max(faces)
  vertex_normals = np.zeros((n_vertices, 3), dtype=face_normals.dtype)
  for face, normal in zip(faces, face_normals):
    vertex_normals[face] += normal
  return vertex_normals


def sample_barycentric_coordinates(n_samples, dtype=np.float32):
  r0, r1 = np.reshape(np.random.uniform(size=2*n_samples), (2, n_samples))
  root_r0 = np.sqrt(r0)
  return np.stack([(1 - root_r0), root_r0 * (1 - r1), root_r0 * r1], axis=-1)


def _barycentric_interpolate(values, barycentric_coords):
  assert(len(values.shape) == len(barycentric_coords.shape) + 1)
  assert(values.shape[-2] == barycentric_coords.shape[-1])
  return np.sum(values * np.expand_dims(barycentric_coords, axis=-1), axis=-2)


def _categorical_barycentric_interpolate(
      vertex_values, faces, face_lengths, barycentric_coords):
  faces = np.repeat(faces, face_lengths, axis=0)
  vertex_values = vertex_values[faces]
  return _barycentric_interpolate(vertex_values, barycentric_coords)


def sample_faces(vertices, faces, n_total, include_normals=True):
  vertices = np.asarray(vertices)
  if n_total == 0:
    return np.empty(shape=(0, 3), dtype=vertices.dtype)
  if len(faces) == 0:
    raise ValueError('Cannot sample points from zero faces.')
  face_normals = compute_face_normals(vertices, faces)
  areas = np.linalg.norm(face_normals, ord=2, axis=-1)
  area_total = np.sum(areas)
  if not np.isfinite(area_total):
    raise ValueError('Total area not finite')
  # np.random.multinomial has issues if areas sum greater than 1, even by a bit
  areas /= (area_total * (1 + 1e-5))
  face_lengths = np.random.multinomial(n_total, areas)
  barycentric_coords = sample_barycentric_coordinates(n_total)
  points = _categorical_barycentric_interpolate(
      vertices, faces, face_lengths, barycentric_coords)
  if include_normals:
    vertex_normals = compute_vertex_normals(
        faces, face_normals, n_vertices=vertices.shape[0])
    point_normals = _categorical_barycentric_interpolate(
        vertex_normals, faces, face_lengths, barycentric_coords)
    normalize(point_normals)
    return face_lengths, barycentric_coords, points, point_normals
  else:
    return face_lengths, barycentric_coords, points
