import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CompressionResult:
    compressed_bytes: bytes | np.ndarray
    original_shape: tuple
    original_dtype: np.dtype
    compressed_size: int
    original_size: int
    compression_ratio: float
    compression_time: float


# ---------------------------------------------------------------------------
# SZ3 Compressor (lossy, error-bounded)
# pip install pysz
# ---------------------------------------------------------------------------
class SZ3Compressor:
    """Lossy compressor using SZ3 with configurable error bounds.

    Parameters
    ----------
    error_bound_mode : str
        One of 'abs', 'rel', 'abs_and_rel', 'abs_or_rel', 'psnr', 'l2norm'.
    abs_error_bound : float
        Absolute error bound (used when mode includes 'abs').
    rel_error_bound : float
        Relative error bound (used when mode includes 'rel').
    psnr_error_bound : float
        PSNR error bound (used when mode is 'psnr').
    l2norm_error_bound : float
        L2-norm error bound (used when mode is 'l2norm').
    algorithm : str or None
        SZ3 prediction algorithm: 'interp', 'lorenzo', 'interp_lorenzo',
        'lossless', or None for default.
    """

    _MODE_MAP = None
    _ALGO_MAP = None

    def __init__(
        self,
        error_bound_mode: str = "abs",
        abs_error_bound: float = 1e-3,
        rel_error_bound: float = 1e-3,
        psnr_error_bound: float = 80.0,
        l2norm_error_bound: float = 1e-3,
        algorithm: str | None = None,
    ):
        try:
            import pysz
        except ImportError:
            raise ImportError("pysz library is required for SZ3Compressor. Install with 'pip install pysz'.")

        if SZ3Compressor._MODE_MAP is None:
            SZ3Compressor._MODE_MAP = {
                "abs": pysz.szErrorBoundMode.ABS,
                "rel": pysz.szErrorBoundMode.REL,
                "abs_and_rel": pysz.szErrorBoundMode.ABS_AND_REL,
                "abs_or_rel": pysz.szErrorBoundMode.ABS_OR_REL,
                "psnr": pysz.szErrorBoundMode.PSNR,
                "l2norm": pysz.szErrorBoundMode.L2NORM,
            }
            SZ3Compressor._ALGO_MAP = {
                "interp": pysz.szAlgorithm.INTERP,
                "lorenzo": pysz.szAlgorithm.LORENZO_REG,
                "interp_lorenzo": pysz.szAlgorithm.INTERP_LORENZO,
                "lossless": pysz.szAlgorithm.LOSSLESS,
            }

        self.pysz = pysz
        self.error_bound_mode = error_bound_mode
        self.abs_error_bound = abs_error_bound
        self.rel_error_bound = rel_error_bound
        self.psnr_error_bound = psnr_error_bound
        self.l2norm_error_bound = l2norm_error_bound
        self.algorithm = algorithm

    def _make_config(self, shape: tuple):
        """Build an SZ3 configuration object for the given array shape."""
        config = self.pysz.szConfig()
        mode_key = self.error_bound_mode.lower()
        config.errorBoundMode = self._MODE_MAP[mode_key]
        config.absErrorBound = self.abs_error_bound
        config.relErrorBound = self.rel_error_bound
        config.psnrErrorBound = self.psnr_error_bound
        config.l2normErrorBound = self.l2norm_error_bound
        if self.algorithm is not None:
            config.cmprAlgo = self._ALGO_MAP[self.algorithm]
        config.setDims(*shape)
        return config

    def compress(self, data: np.ndarray) -> CompressionResult:
        """Compress a numpy array using SZ3 and return a CompressionResult."""
        data = np.ascontiguousarray(data)
        config = self._make_config(data.shape)
        sz = self.pysz.sz(config)

        t0 = time.perf_counter()
        compressed, ratio = sz.compress(data, config)
        dt = time.perf_counter() - t0

        return CompressionResult(
            compressed_bytes=compressed,
            original_shape=data.shape,
            original_dtype=data.dtype,
            compressed_size=compressed.nbytes,
            original_size=data.nbytes,
            compression_ratio=ratio,
            compression_time=dt,
        )

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress a CompressionResult back to a numpy array."""
        config = self._make_config(result.original_shape)
        sz = self.pysz.sz(config)
        decompressed, _ = sz.decompress(
            result.compressed_bytes, result.original_dtype, result.original_shape
        )
        return decompressed

    def compress_and_decompress(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, CompressionResult]:
        """Compress and immediately decompress, returning both the reconstructed array and compression metadata."""
        result = self.compress(data)
        decompressed = self.decompress(result)
        return decompressed, result


# ---------------------------------------------------------------------------
# ZFP Compressor (lossy, error-bounded)
# pip install zfpy
# ---------------------------------------------------------------------------
class ZFPCompressor:
    """Lossy compressor using ZFP with configurable compression modes.

    Parameters
    ----------
    mode : str
        Compression mode: 'tolerance' (fixed-accuracy), 'rate' (fixed-rate),
        'precision' (fixed-precision), or 'lossless'.
    tolerance : float
        Absolute error tolerance (used when mode is 'tolerance').
    rate : float
        Bits per value (used when mode is 'rate').
    precision : int
        Uncompressed bits per value to preserve (used when mode is 'precision').
    """

    def __init__(
        self,
        mode: str = "tolerance",
        tolerance: float = 1e-3,
        rate: float = 16.0,
        precision: int = 32,
    ):
        try:
            import zfpy
        except ImportError:
            raise ImportError("zfpy library is required for ZFPCompressor. Install with 'pip install zfpy'.")

        self.zfpy = zfpy
        self.mode = mode
        self.tolerance = tolerance
        self.rate = rate
        self.precision = precision

    def _compress_kwargs(self) -> dict:
        """Return the keyword arguments for ``zfpy.compress_numpy`` based on the active mode."""
        if self.mode == "tolerance":
            return {"tolerance": self.tolerance}
        elif self.mode == "rate":
            return {"rate": self.rate}
        elif self.mode == "precision":
            return {"precision": self.precision}
        elif self.mode == "lossless":
            return {}
        else:
            raise ValueError(f"Unsupported ZFP mode: {self.mode}")

    def compress(self, data: np.ndarray) -> CompressionResult:
        """Compress a numpy array using ZFP and return a CompressionResult."""
        data = np.ascontiguousarray(data)
        kwargs = self._compress_kwargs()

        t0 = time.perf_counter()
        compressed = self.zfpy.compress_numpy(data, **kwargs)
        dt = time.perf_counter() - t0

        compressed_size = len(compressed)
        original_size = data.nbytes
        return CompressionResult(
            compressed_bytes=compressed,
            original_shape=data.shape,
            original_dtype=data.dtype,
            compressed_size=compressed_size,
            original_size=original_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else float("inf"),
            compression_time=dt,
        )

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress a CompressionResult back to a numpy array."""
        return self.zfpy.decompress_numpy(result.compressed_bytes)

    def compress_and_decompress(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, CompressionResult]:
        """Compress and immediately decompress, returning both the reconstructed array and compression metadata."""
        result = self.compress(data)
        decompressed = self.decompress(result)
        return decompressed, result


# ---------------------------------------------------------------------------
# MGARD Compressor (lossy, error-bounded, multi-level)
# Built from source: https://github.com/CODARcode/MGARD
# Python bindings via pybind11 (mgard_binding module)
# ---------------------------------------------------------------------------
class MGARDCompressor:
    """Lossy compressor using MGARD-X with error-bounded compression.

    Parameters
    ----------
    tolerance : float
        Error tolerance.
    s : float
        Smoothness parameter. Use 0.0 for L-infinity norm (default),
        or a finite positive value for a weighted Sobolev norm.
    mode : str
        Error bound type: 'abs' (absolute) or 'rel' (relative).
    """

    _DTYPE_MAP = {"float32": "float32", "float64": "float64"}

    def __init__(
        self, tolerance: float = 1e-3, s: float = 0.0, mode: str = "abs"
    ):
        try:
            import mgard_binding
        except ImportError:
            raise ImportError("mgard_binding module is required for MGARDCompressor." \
            "As of now, you need to build the MGARD-X C++ library from source and create Python bindings using pybind11.")

        self.mgard = mgard_binding
        self.tolerance = tolerance
        self.s = s
        self.mode = mode

    def compress(self, data: np.ndarray) -> CompressionResult:
        """Compress a numpy array using MGARD-X and return a CompressionResult."""
        data = np.ascontiguousarray(data)
        if data.dtype not in (np.float32, np.float64):
            raise ValueError("MGARD only supports float32 and float64 arrays")

        t0 = time.perf_counter()
        compressed, compressed_size = self.mgard.compress(
            data, self.tolerance, self.s, self.mode
        )
        dt = time.perf_counter() - t0

        original_size = data.nbytes
        return CompressionResult(
            compressed_bytes=compressed,
            original_shape=data.shape,
            original_dtype=data.dtype,
            compressed_size=compressed_size,
            original_size=original_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else float("inf"),
            compression_time=dt,
        )

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress a CompressionResult back to a numpy array."""
        dtype_str = self._DTYPE_MAP.get(str(result.original_dtype))
        if dtype_str is None:
            raise ValueError(f"Unsupported dtype: {result.original_dtype}")
        decompressed = self.mgard.decompress(
            result.compressed_bytes, dtype_str, tuple(result.original_shape)
        )
        return decompressed

    def compress_and_decompress(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, CompressionResult]:
        """Compress and immediately decompress, returning both the reconstructed array and compression metadata."""
        result = self.compress(data)
        decompressed = self.decompress(result)
        return decompressed, result


# ---------------------------------------------------------------------------
# Blosc2 Compressor (lossless by default, lossy via TRUNC_PREC filter)
# pip install blosc2
# ---------------------------------------------------------------------------
class Blosc2Compressor:
    """Fast multi-threaded compressor using Blosc2.

    Lossless by default. Enable lossy compression by setting
    ``trunc_prec_bits`` to truncate floating-point mantissa bits.

    Parameters
    ----------
    codec : str
        Compression codec: 'zstd', 'lz4', 'lz4hc', 'blosclz', 'zlib'.
    clevel : int
        Compression level (0-9, higher = smaller but slower).
    filter : str
        Pre-compression filter: 'shuffle', 'bitshuffle', 'bytedelta',
        'nofilter', or 'delta'.
    trunc_prec_bits : int or None
        If set, truncates mantissa to this many bits for lossy compression.
        For float32 (23 mantissa bits), lower values = more lossy.
        For float64 (52 mantissa bits), same principle applies.
        Set to None for lossless compression.
    nthreads : int
        Number of threads for compression/decompression.
    """

    _CODEC_MAP = None
    _FILTER_MAP = None

    def __init__(
        self,
        codec: str = "zstd",
        clevel: int = 9,
        filter: str = "shuffle",
        trunc_prec_bits: int | None = None,
        nthreads: int = 1,
    ):
        try:
            import blosc2
        except ImportError:
            raise ImportError("blosc2 library is required for Blosc2Compressor. Install with 'pip install blosc2'.")

        if Blosc2Compressor._CODEC_MAP is None:
            Blosc2Compressor._CODEC_MAP = {
                "blosclz": blosc2.Codec.BLOSCLZ,
                "lz4": blosc2.Codec.LZ4,
                "lz4hc": blosc2.Codec.LZ4HC,
                "zlib": blosc2.Codec.ZLIB,
                "zstd": blosc2.Codec.ZSTD,
            }
            Blosc2Compressor._FILTER_MAP = {
                "nofilter": blosc2.Filter.NOFILTER,
                "shuffle": blosc2.Filter.SHUFFLE,
                "bitshuffle": blosc2.Filter.BITSHUFFLE,
                "delta": blosc2.Filter.DELTA,
                "bytedelta": blosc2.Filter.BYTEDELTA,
            }

        self.blosc2 = blosc2
        self.codec = codec
        self.clevel = clevel
        self.filter = filter
        self.trunc_prec_bits = trunc_prec_bits
        self.nthreads = nthreads

    def _make_cparams(self, typesize: int):
        """Build Blosc2 compression parameters for the given element byte size."""
        filters = []
        filters_meta = []

        if self.trunc_prec_bits is not None:
            filters.append(self.blosc2.Filter.TRUNC_PREC)
            filters_meta.append(self.trunc_prec_bits)

        filters.append(self._FILTER_MAP[self.filter])
        filters_meta.append(0)

        return self.blosc2.CParams(
            codec=self._CODEC_MAP[self.codec],
            clevel=self.clevel,
            typesize=typesize,
            filters=filters,
            filters_meta=filters_meta,
            nthreads=self.nthreads,
        )

    def compress(self, data: np.ndarray) -> CompressionResult:
        """Compress a numpy array using Blosc2 and return a CompressionResult."""
        data = np.ascontiguousarray(data)
        cparams = self._make_cparams(data.dtype.itemsize)
        dparams = self.blosc2.DParams(nthreads=self.nthreads)

        t0 = time.perf_counter()
        compressed = self.blosc2.pack_array2(data, cparams=cparams, dparams=dparams)
        dt = time.perf_counter() - t0

        compressed_size = len(compressed)
        original_size = data.nbytes
        return CompressionResult(
            compressed_bytes=compressed,
            original_shape=data.shape,
            original_dtype=data.dtype,
            compressed_size=compressed_size,
            original_size=original_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else float("inf"),
            compression_time=dt,
        )

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress a CompressionResult back to a numpy array."""
        return self.blosc2.unpack_array2(result.compressed_bytes)

    def compress_and_decompress(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, CompressionResult]:
        """Compress and immediately decompress, returning both the reconstructed array and compression metadata."""
        result = self.compress(data)
        decompressed = self.decompress(result)
        return decompressed, result


# ---------------------------------------------------------------------------
# Utility: run all compressors on a single array and return metrics
# ---------------------------------------------------------------------------
def compute_metrics(original: np.ndarray, decompressed: np.ndarray) -> dict:
    """Compute reconstruction quality metrics (MSE, MAE, max absolute error, relative L2 error) between original and decompressed arrays."""
    diff = original.astype(np.float64) - decompressed.astype(np.float64)
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_abs = np.max(np.abs(diff))
    orig_norm = np.linalg.norm(original.astype(np.float64).ravel())
    rel_l2 = np.linalg.norm(diff.ravel()) / orig_norm if orig_norm > 0 else float("inf")
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_abs_error": float(max_abs),
        "rel_l2_error": float(rel_l2),
    }


def benchmark_compressor(compressor, data: np.ndarray) -> dict:
    """Compress, decompress, and return metrics + compression stats."""
    decompressed, result = compressor.compress_and_decompress(data)
    metrics = compute_metrics(data, decompressed)
    return decompressed, {
        "compression_ratio": result.compression_ratio,
        "compressed_size_bytes": result.compressed_size,
        "original_size_bytes": result.original_size,
        "compression_time_s": result.compression_time,
        **metrics,
    }

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = np.random.randn(100, 64, 64).astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min())
    # print(data[50].shape)
    # i = 50
    # data = np.random.rand(64, 64).astype(np.float32)
    compressor = SZ3Compressor(error_bound_mode="abs", abs_error_bound=1e-3)
    decompressed, results = benchmark_compressor(compressor, data[50])

    print(results)
