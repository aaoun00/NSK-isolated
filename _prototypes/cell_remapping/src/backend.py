import numpy as np
import scipy
import scipy.linalg
import scipy.special as special
from scipy.sparse import issparse, coo_matrix, csr_matrix
import warnings
import time




str_type_error = "All array should be from the same type/backend. Current types are : {}"


def get_backend(*args):
    """Returns the proper backend for a list of input arrays

        Also raises TypeError if all arrays are not from the same backend
    """
    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one parameter")
    # check all same type
    if not len(set(type(a) for a in args)) == 1:
        raise ValueError(str_type_error.format([type(a) for a in args]))

    if isinstance(args[0], np.ndarray):
        return NumpyBackend()
    # elif isinstance(args[0], torch_type):
    #     return TorchBackend()
    # elif isinstance(args[0], jax_type):
    #     return JaxBackend()
    # elif isinstance(args[0], cp_type):  # pragma: no cover
    #     return CupyBackend()
    # elif isinstance(args[0], tf_type):
    #     return TensorflowBackend()
    else:
        raise ValueError("Unknown type of non implemented backend.")

def to_numpy(*args):
    """Returns numpy arrays from any compatible backend"""

    if len(args) == 1:
        return get_backend(args[0]).to_numpy(args[0])
    else:
        return [get_backend(a).to_numpy(a) for a in args]


class Backend():
    """
    Backend abstract class.
    Implementations: :py:class:`JaxBackend`, :py:class:`NumpyBackend`, :py:class:`TorchBackend`,
    :py:class:`CupyBackend`, :py:class:`TensorflowBackend`

    - The `__name__` class attribute refers to the name of the backend.
    - The `__type__` class attribute refers to the data structure used by the backend.
    """

    __name__ = None
    __type__ = None
    __type_list__ = None

    rng_ = None

    def __str__(self):
        return self.__name__

    # convert batch of tensors to numpy
    def to_numpy(self, *arrays):
        """Returns the numpy version of tensors"""
        if len(arrays) == 1:
            return self._to_numpy(arrays[0])
        else:
            return [self._to_numpy(array) for array in arrays]


    # convert a tensor to numpy
    def _to_numpy(self, a):
        """Returns the numpy version of a tensor"""
        raise NotImplementedError()

    # convert batch of arrays from numpy
    def from_numpy(self, *arrays, type_as=None):
        """Creates tensors cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        if len(arrays) == 1:
            return self._from_numpy(arrays[0], type_as=type_as)
        else:
            return [self._from_numpy(array, type_as=type_as) for array in arrays]


    # convert an array from numpy
    def _from_numpy(self, a, type_as=None):
        """Creates a tensor cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        raise NotImplementedError()

    def set_gradients(self, val, inputs, grads):
        """Define the gradients for the value val wrt the inputs """
        raise NotImplementedError()


    def zeros(self, shape, type_as=None):
        r"""
        Creates a tensor full of zeros.

        This function follows the api from :any:`numpy.zeros`

        See: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
        """
        raise NotImplementedError()


    def ones(self, shape, type_as=None):
        r"""
        Creates a tensor full of ones.

        This function follows the api from :any:`numpy.ones`

        See: https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        """
        raise NotImplementedError()


    def arange(self, stop, start=0, step=1, type_as=None):
        r"""
        Returns evenly spaced values within a given interval.

        This function follows the api from :any:`numpy.arange`

        See: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        """
        raise NotImplementedError()


    def full(self, shape, fill_value, type_as=None):
        r"""
        Creates a tensor with given shape, filled with given value.

        This function follows the api from :any:`numpy.full`

        See: https://numpy.org/doc/stable/reference/generated/numpy.full.html
        """
        raise NotImplementedError()


    def eye(self, N, M=None, type_as=None):
        r"""
        Creates the identity matrix of given size.

        This function follows the api from :any:`numpy.eye`

        See: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
        """
        raise NotImplementedError()


    def sum(self, a, axis=None, keepdims=False):
        r"""
        Sums tensor elements over given dimensions.

        This function follows the api from :any:`numpy.sum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        raise NotImplementedError()


    def cumsum(self, a, axis=None):
        r"""
        Returns the cumulative sum of tensor elements over given dimensions.

        This function follows the api from :any:`numpy.cumsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        """
        raise NotImplementedError()


    def max(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        """
        raise NotImplementedError()


    def min(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amin.html
        """
        raise NotImplementedError()


    def maximum(self, a, b):
        r"""
        Returns element-wise maximum of array elements.

        This function follows the api from :any:`numpy.maximum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        """
        raise NotImplementedError()


    def minimum(self, a, b):
        r"""
        Returns element-wise minimum of array elements.

        This function follows the api from :any:`numpy.minimum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        """
        raise NotImplementedError()


    def dot(self, a, b):
        r"""
        Returns the dot product of two tensors.

        This function follows the api from :any:`numpy.dot`

        See: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        """
        raise NotImplementedError()


    def abs(self, a):
        r"""
        Computes the absolute value element-wise.

        This function follows the api from :any:`numpy.absolute`

        See: https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
        """
        raise NotImplementedError()


    def exp(self, a):
        r"""
        Computes the exponential value element-wise.

        This function follows the api from :any:`numpy.exp`

        See: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        """
        raise NotImplementedError()


    def log(self, a):
        r"""
        Computes the natural logarithm, element-wise.

        This function follows the api from :any:`numpy.log`

        See: https://numpy.org/doc/stable/reference/generated/numpy.log.html
        """
        raise NotImplementedError()


    def sqrt(self, a):
        r"""
        Returns the non-ngeative square root of a tensor, element-wise.

        This function follows the api from :any:`numpy.sqrt`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        raise NotImplementedError()


    def power(self, a, exponents):
        r"""
        First tensor elements raised to powers from second tensor, element-wise.

        This function follows the api from :any:`numpy.power`

        See: https://numpy.org/doc/stable/reference/generated/numpy.power.html
        """
        raise NotImplementedError()


    def norm(self, a):
        r"""
        Computes the matrix frobenius norm.

        This function follows the api from :any:`numpy.linalg.norm`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        raise NotImplementedError()


    def any(self, a):
        r"""
        Tests whether any tensor element along given dimensions evaluates to True.

        This function follows the api from :any:`numpy.any`

        See: https://numpy.org/doc/stable/reference/generated/numpy.any.html
        """
        raise NotImplementedError()


    def isnan(self, a):
        r"""
        Tests element-wise for NaN and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isnan`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        """
        raise NotImplementedError()


    def isinf(self, a):
        r"""
        Tests element-wise for positive or negative infinity and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isinf`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
        """
        raise NotImplementedError()


    def einsum(self, subscripts, *operands):
        r"""
        Evaluates the Einstein summation convention on the operands.

        This function follows the api from :any:`numpy.einsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        raise NotImplementedError()


    def sort(self, a, axis=-1):
        r"""
        Returns a sorted copy of a tensor.

        This function follows the api from :any:`numpy.sort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        raise NotImplementedError()


    def argsort(self, a, axis=None):
        r"""
        Returns the indices that would sort a tensor.

        This function follows the api from :any:`numpy.argsort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        """
        raise NotImplementedError()


    def searchsorted(self, a, v, side='left'):
        r"""
        Finds indices where elements should be inserted to maintain order in given tensor.

        This function follows the api from :any:`numpy.searchsorted`

        See: https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError()


    def flip(self, a, axis=None):
        r"""
        Reverses the order of elements in a tensor along given dimensions.

        This function follows the api from :any:`numpy.flip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
        """
        raise NotImplementedError()


    def clip(self, a, a_min, a_max):
        """
        Limits the values in a tensor.

        This function follows the api from :any:`numpy.clip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        """
        raise NotImplementedError()


    def repeat(self, a, repeats, axis=None):
        r"""
        Repeats elements of a tensor.

        This function follows the api from :any:`numpy.repeat`

        See: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        """
        raise NotImplementedError()


    def take_along_axis(self, arr, indices, axis):
        r"""
        Gathers elements of a tensor along given dimensions.

        This function follows the api from :any:`numpy.take_along_axis`

        See: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
        """
        raise NotImplementedError()


    def concatenate(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along an existing dimension.

        This function follows the api from :any:`numpy.concatenate`

        See: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        """
        raise NotImplementedError()


    def zero_pad(self, a, pad_width):
        r"""
        Pads a tensor.

        This function follows the api from :any:`numpy.pad`

        See: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        raise NotImplementedError()


    def argmax(self, a, axis=None):
        r"""
        Returns the indices of the maximum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        raise NotImplementedError()


    def argmin(self, a, axis=None):
        r"""
        Returns the indices of the minimum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        """
        raise NotImplementedError()


    def mean(self, a, axis=None):
        r"""
        Computes the arithmetic mean of a tensor along given dimensions.

        This function follows the api from :any:`numpy.mean`

        See: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        raise NotImplementedError()


    def std(self, a, axis=None):
        r"""
        Computes the standard deviation of a tensor along given dimensions.

        This function follows the api from :any:`numpy.std`

        See: https://numpy.org/doc/stable/reference/generated/numpy.std.html
        """
        raise NotImplementedError()


    def linspace(self, start, stop, num):
        r"""
        Returns a specified number of evenly spaced values over a given interval.

        This function follows the api from :any:`numpy.linspace`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        """
        raise NotImplementedError()


    def meshgrid(self, a, b):
        r"""
        Returns coordinate matrices from coordinate vectors (Numpy convention).

        This function follows the api from :any:`numpy.meshgrid`

        See: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        """
        raise NotImplementedError()


    def diag(self, a, k=0):
        r"""
        Extracts or constructs a diagonal tensor.

        This function follows the api from :any:`numpy.diag`

        See: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        """
        raise NotImplementedError()


    def unique(self, a):
        r"""
        Finds unique elements of given tensor.

        This function follows the api from :any:`numpy.unique`

        See: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        """
        raise NotImplementedError()


    def logsumexp(self, a, axis=None):
        r"""
        Computes the log of the sum of exponentials of input elements.

        This function follows the api from :any:`scipy.special.logsumexp`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        raise NotImplementedError()


    def stack(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along a new dimension.

        This function follows the api from :any:`numpy.stack`

        See: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        """
        raise NotImplementedError()


    def outer(self, a, b):
        r"""
        Computes the outer product between two vectors.

        This function follows the api from :any:`numpy.outer`

        See: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        """
        raise NotImplementedError()


    def reshape(self, a, shape):
        r"""
        Gives a new shape to a tensor without changing its data.

        This function follows the api from :any:`numpy.reshape`

        See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        """
        raise NotImplementedError()


    def seed(self, seed=None):
        r"""
        Sets the seed for the random generator.

        This function follows the api from :any:`numpy.random.seed`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.seed.html
        """
        raise NotImplementedError()


    def rand(self, *size, type_as=None):
        r"""
        Generate uniform random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html
        """
        raise NotImplementedError()


    def randn(self, *size, type_as=None):
        r"""
        Generate normal Gaussian random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html
        """
        raise NotImplementedError()


    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        r"""
        Creates a sparse tensor in COOrdinate format.

        This function follows the api from :any:`scipy.sparse.coo_matrix`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        """
        raise NotImplementedError()


    def issparse(self, a):
        r"""
        Checks whether or not the input tensor is a sparse tensor.

        This function follows the api from :any:`scipy.sparse.issparse`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html
        """
        raise NotImplementedError()


    def tocsr(self, a):
        r"""
        Converts this matrix to Compressed Sparse Row format.

        This function follows the api from :any:`scipy.sparse.coo_matrix.tocsr`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html
        """
        raise NotImplementedError()


    def eliminate_zeros(self, a, threshold=0.):
        r"""
        Removes entries smaller than the given threshold from the sparse tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.eliminate_zeros`

        See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html
        """
        raise NotImplementedError()


    def todense(self, a):
        r"""
        Converts a sparse tensor to a dense tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.toarray`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html
        """
        raise NotImplementedError()


    def where(self, condition, x, y):
        r"""
        Returns elements chosen from x or y depending on condition.

        This function follows the api from :any:`numpy.where`

        See: https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """
        raise NotImplementedError()


    def copy(self, a):
        r"""
        Returns a copy of the given tensor.

        This function follows the api from :any:`numpy.copy`

        See: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
        """
        raise NotImplementedError()


    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        r"""
        Returns True if two arrays are element-wise equal within a tolerance.

        This function follows the api from :any:`numpy.allclose`

        See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        raise NotImplementedError()


    def dtype_device(self, a):
        r"""
        Returns the dtype and the device of the given tensor.
        """
        raise NotImplementedError()


    def assert_same_dtype_device(self, a, b):
        r"""
        Checks whether or not the two given inputs have the same dtype as well as the same device
        """
        raise NotImplementedError()


    def squeeze(self, a, axis=None):
        r"""
        Remove axes of length one from a.

        This function follows the api from :any:`numpy.squeeze`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        """
        raise NotImplementedError()


    def bitsize(self, type_as):
        r"""
        Gives the number of bits used by the data type of the given tensor.
        """
        raise NotImplementedError()


    def device_type(self, type_as):
        r"""
        Returns CPU or GPU depending on the device where the given tensor is located.
        """
        raise NotImplementedError()


    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        r"""
        Executes a benchmark of the given callable with the given arguments.
        """
        raise NotImplementedError()

    def solve(self, a, b):
        r"""
        Solves a linear matrix equation, or system of linear scalar equations.

        This function follows the api from :any:`numpy.linalg.solve`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        """
        raise NotImplementedError()


    def trace(self, a):
        r"""
        Returns the sum along diagonals of the array.

        This function follows the api from :any:`numpy.trace`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.trace.html
        """
        raise NotImplementedError()


    def inv(self, a):
        r"""
        Computes the inverse of a matrix.

        This function follows the api from :any:`scipy.linalg.inv`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html
        """
        raise NotImplementedError()


    def sqrtm(self, a):
        r"""
        Computes the matrix square root. Requires input to be definite positive.

        This function follows the api from :any:`scipy.linalg.sqrtm`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
        """
        raise NotImplementedError()


    def isfinite(self, a):
        r"""
        Tests element-wise for finiteness (not infinity and not Not a Number).

        This function follows the api from :any:`numpy.isfinite`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
        """
        raise NotImplementedError()


    def array_equal(self, a, b):
        r"""
        True if two arrays have the same shape and elements, False otherwise.

        This function follows the api from :any:`numpy.array_equal`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        """
        raise NotImplementedError()


    def is_floating_point(self, a):
        r"""
        Returns whether or not the input consists of floats
        """
        raise NotImplementedError()


class NumpyBackend(Backend):
    """
    NumPy implementation of the backend

    - `__name__` is "numpy"
    - `__type__` is np.ndarray
    """

    __name__ = 'numpy'
    __type__ = np.ndarray
    __type_list__ = [np.array(1, dtype=np.float32),
                     np.array(1, dtype=np.float64)]

    rng_ = np.random.RandomState()

    def _to_numpy(self, a):
        return a

    def _from_numpy(self, a, type_as=None):
        if type_as is None:
            return a
        elif isinstance(a, float):
            return a
        else:
            return a.astype(type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        # No gradients for numpy
        return val


    def zeros(self, shape, type_as=None):
        if type_as is None:
            return np.zeros(shape)
        else:
            return np.zeros(shape, dtype=type_as.dtype)


    def ones(self, shape, type_as=None):
        if type_as is None:
            return np.ones(shape)
        else:
            return np.ones(shape, dtype=type_as.dtype)


    def arange(self, stop, start=0, step=1, type_as=None):
        return np.arange(start, stop, step)


    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return np.full(shape, fill_value)
        else:
            return np.full(shape, fill_value, dtype=type_as.dtype)


    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return np.eye(N, M)
        else:
            return np.eye(N, M, dtype=type_as.dtype)


    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis, keepdims=keepdims)


    def cumsum(self, a, axis=None):
        return np.cumsum(a, axis)


    def max(self, a, axis=None, keepdims=False):
        return np.max(a, axis, keepdims=keepdims)


    def min(self, a, axis=None, keepdims=False):
        return np.min(a, axis, keepdims=keepdims)


    def maximum(self, a, b):
        return np.maximum(a, b)


    def minimum(self, a, b):
        return np.minimum(a, b)


    def dot(self, a, b):
        return np.dot(a, b)


    def abs(self, a):
        return np.abs(a)


    def exp(self, a):
        return np.exp(a)


    def log(self, a):
        return np.log(a)


    def sqrt(self, a):
        return np.sqrt(a)


    def power(self, a, exponents):
        return np.power(a, exponents)


    def norm(self, a):
        return np.sqrt(np.sum(np.square(a)))


    def any(self, a):
        return np.any(a)


    def isnan(self, a):
        return np.isnan(a)


    def isinf(self, a):
        return np.isinf(a)


    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, *operands)


    def sort(self, a, axis=-1):
        return np.sort(a, axis)


    def argsort(self, a, axis=-1):
        return np.argsort(a, axis)


    def searchsorted(self, a, v, side='left'):
        if a.ndim == 1:
            return np.searchsorted(a, v, side)
        else:
            # this is a not very efficient way to make numpy
            # searchsorted work on 2d arrays
            ret = np.empty(v.shape, dtype=int)
            for i in range(a.shape[0]):
                ret[i, :] = np.searchsorted(a[i, :], v[i, :], side)
            return ret


    def flip(self, a, axis=None):
        return np.flip(a, axis)


    def outer(self, a, b):
        return np.outer(a, b)


    def clip(self, a, a_min, a_max):
        return np.clip(a, a_min, a_max)


    def repeat(self, a, repeats, axis=None):
        return np.repeat(a, repeats, axis)


    def take_along_axis(self, arr, indices, axis):
        return np.take_along_axis(arr, indices, axis)


    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis)


    def zero_pad(self, a, pad_width):
        return np.pad(a, pad_width)


    def argmax(self, a, axis=None):
        return np.argmax(a, axis=axis)


    def argmin(self, a, axis=None):
        return np.argmin(a, axis=axis)


    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)


    def std(self, a, axis=None):
        return np.std(a, axis=axis)


    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num)


    def meshgrid(self, a, b):
        return np.meshgrid(a, b)


    def diag(self, a, k=0):
        return np.diag(a, k)


    def unique(self, a):
        return np.unique(a)


    def logsumexp(self, a, axis=None):
        return special.logsumexp(a, axis=axis)


    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis)


    def reshape(self, a, shape):
        return np.reshape(a, shape)


    def seed(self, seed=None):
        if seed is not None:
            self.rng_.seed(seed)


    def rand(self, *size, type_as=None):
        return self.rng_.rand(*size)


    def randn(self, *size, type_as=None):
        return self.rng_.randn(*size)


    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if type_as is None:
            return coo_matrix((data, (rows, cols)), shape=shape)
        else:
            return coo_matrix((data, (rows, cols)), shape=shape, dtype=type_as.dtype)


    def issparse(self, a):
        return issparse(a)


    def tocsr(self, a):
        if self.issparse(a):
            return a.tocsr()
        else:
            return csr_matrix(a)


    def eliminate_zeros(self, a, threshold=0.):
        if threshold > 0:
            if self.issparse(a):
                a.data[self.abs(a.data) <= threshold] = 0
            else:
                a[self.abs(a) <= threshold] = 0
        if self.issparse(a):
            a.eliminate_zeros()
        return a


    def todense(self, a):
        if self.issparse(a):
            return a.toarray()
        else:
            return a


    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return np.where(condition)
        else:
            return np.where(condition, x, y)


    def copy(self, a):
        return a.copy()


    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


    def dtype_device(self, a):
        if hasattr(a, "dtype"):
            return a.dtype, "cpu"
        else:
            return type(a), "cpu"


    def assert_same_dtype_device(self, a, b):
        # numpy has implicit type conversion so we automatically validate the test
        pass


    def squeeze(self, a, axis=None):
        return np.squeeze(a, axis=axis)


    def bitsize(self, type_as):
        return type_as.itemsize * 8


    def device_type(self, type_as):
        return "CPU"


    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        results = dict()
        for type_as in self.__type_list__:
            inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
            for _ in range(warmup_runs):
                callable(*inputs)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                callable(*inputs)
            t1 = time.perf_counter()
            key = ("Numpy", self.device_type(type_as), self.bitsize(type_as))
            results[key] = (t1 - t0) / n_runs
        return results

    def solve(self, a, b):
        return np.linalg.solve(a, b)


    def trace(self, a):
        return np.trace(a)


    def inv(self, a):
        return scipy.linalg.inv(a)


    def sqrtm(self, a):
        return scipy.linalg.sqrtm(a)


    def isfinite(self, a):
        return np.isfinite(a)


    def array_equal(self, a, b):
        return np.array_equal(a, b)


    def is_floating_point(self, a):
        return a.dtype.kind == "f"

