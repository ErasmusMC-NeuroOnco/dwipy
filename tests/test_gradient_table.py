import os
import tempfile

import numpy as np

import dwipy.mrtrix.gradient_table


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "grads")


def default_grad_tests(grads: np.ndarray):
    # Grad should always be an array
    assert isinstance(grads, np.ndarray)
    # Dim of 2
    assert grads.ndim == 2
    # Where the second dim is the bvecs (3) + val
    assert grads.shape[1] == 4
    # The bvals should be ints
    assert all(np.mod(grads[:, 3], 1) == 0)


def default_bval_tests(bvals: np.ndarray):
    # Should be numpy array
    assert isinstance(bvals, np.ndarray)
    # only one dimension
    assert bvals.ndim == 1
    # bvals should be ints
    assert all(np.mod(bvals, 1) == 0)


def default_bvec_tests(bvecs: np.ndarray):
    # should be numpy array
    assert isinstance(bvecs, np.ndarray)
    # 2 dimensional matrix
    assert bvecs.ndim == 2
    # second shape should be 3 for the different vectors
    assert bvecs.shape[1] == 3


##############
# Grads loading tests
#############


def test_dwi_trace_grad_loading():
    n_grads = 2
    correct_grads = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1000]])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_trace.grads")

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    default_grad_tests(gradient_table.gradients)
    assert gradient_table.n_entries == n_grads
    np.testing.assert_allclose(gradient_table.gradients, correct_grads)


def test_dwi_single_grad_loading():
    n_grads = 1
    correct_grads = np.asarray([[0, 0, 0, 0]])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_single_bval.grads")

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    default_grad_tests(gradient_table.gradients)
    assert gradient_table.n_entries == n_grads
    np.testing.assert_allclose(gradient_table.gradients, correct_grads)


def test_dwi_mixed_grad_loading():
    n_grads = 3
    correct_grads = np.asarray([[0, 0, 0, 1000], [0, 0, 0, 0], [0, 0, 0, 500]])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_mixed_bval.grads")

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    default_grad_tests(gradient_table.gradients)
    assert gradient_table.n_entries == n_grads
    np.testing.assert_allclose(gradient_table.gradients, correct_grads)


def test_dwi_trace_float_grad_loading():
    correct_grads = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]])
    n_grads = 2
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_trace_float_bvals.grads")

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    default_grad_tests(gradient_table.gradients)
    assert gradient_table.n_entries == n_grads
    np.testing.assert_allclose(gradient_table.gradients, correct_grads)


def test_dwi_with_bvals_loading():
    correct_grads = np.asarray([[0, 0, 0, 0], [1, 0, 0, 1000], [0, 1, 0, 1000], [0, 0, 1, 1000]])
    n_grads = 4
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_with_bvecs.grads")

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    default_grad_tests(gradient_table.gradients)
    assert gradient_table.n_entries == n_grads
    np.testing.assert_allclose(gradient_table.gradients, correct_grads)


# ##############
# # Bval extraction from loaded grads tests
# #############


def test_bval_from_dwi_trace_grads():
    correct_bvals = np.asarray([0, 1000])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_trace.grads")
    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    bvals = gradient_table.bvals

    default_bval_tests(bvals)
    # We have two bvals
    assert bvals.shape == correct_bvals.shape
    np.testing.assert_allclose(bvals, correct_bvals)


def test_bval_from_float_bvals():
    correct_bvals = np.asarray([0, 0])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_trace_float_bvals.grads")
    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    bvals = gradient_table.bvals

    default_bval_tests(bvals)
    # We have two bvals
    assert bvals.shape == correct_bvals.shape
    np.testing.assert_allclose(bvals, correct_bvals)


##############
# bvec extraction from loaded grads tests
#############
def test_bvecs_from_dwi_trace_grads():
    correct_bvecs = np.asarray([[0, 0, 0], [0, 0, 0]])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_trace.grads")
    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    bvecs = gradient_table.bvecs

    default_bvec_tests(bvecs)
    assert bvecs.shape == correct_bvecs.shape
    np.testing.assert_allclose(bvecs, correct_bvecs)


def test_bvecs_from_dwi_bvec_grads():
    correct_bvecs = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    grad_file = os.path.join(TEST_DATA_DIR, "DWI_with_bvecs.grads")
    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    bvecs = gradient_table.bvecs

    default_bvec_tests(bvecs)
    assert bvecs.shape == correct_bvecs.shape
    np.testing.assert_allclose(bvecs, correct_bvecs)


#########
# Gradient table from matrix tests
#########


def test_gradient_table_from_matrix_trace():
    correct_gradient_table = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1000]])

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_matrix(correct_gradient_table)

    np.testing.assert_allclose(gradient_table.gradients, correct_gradient_table)


def test_gradient_table_from_matrix_with_floats():
    init_gradient_table = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1000.2]])
    correct_gradient_table = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1000]])

    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_matrix(init_gradient_table)

    np.testing.assert_allclose(gradient_table.gradients, correct_gradient_table)


#######
## Writing tests
#####


def test_write_dwi_trace_grads():
    correct_gradient_table = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1000]])
    grad_file = tempfile.mktemp()
    gradient_table = dwipy.mrtrix.gradient_table.GradientTable.from_matrix(correct_gradient_table)

    gradient_table.write_to_file(grad_file)
    gradient_table_from_file = dwipy.mrtrix.gradient_table.GradientTable.from_file(grad_file)

    assert os.path.exists(grad_file)
    np.testing.assert_allclose(gradient_table.gradients, gradient_table_from_file.gradients)
    np.testing.assert_allclose(correct_gradient_table, gradient_table_from_file.gradients)
