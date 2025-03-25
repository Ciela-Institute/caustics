# Project Code Changes and Performance Enhancements

This document outlines the changes made to the project codebase, along with their locations, descriptions, and the
corresponding performance improvements.

---

## [1]. **Modified chunk_size of lens_source simulator**

**File:** `src/caustics/sims/lens_source.py`  
**Location:** `LensSource.__call__()`  
**Description:**

- Spawning multiple `vmap` for a simple lens is inefficient since the computation is already vectorized.
- Modified `chunk_size` to be a large number (2^24) so that it can be changed if memory issues are experienced, but is
  not used otherwise.
- Using `vmap` without splitting the workload still calls `torch/_functorch/vmap.py(328): _concat_chunked_outputs`.
  Using the `vmap` calls only when necessary (only if `chunk_size` is not `None`) should accelerate `lens_source` by ~
  20%. (Passing `chunk_size=None` to vmap has a 5us overhead, so it might be optimal to keep it like that instead of
  having 4 boolean checks.)

**Performance Impact:**

- Setup: Lens: SIE, Source: Sersic, LensLight: Sersic, 100x100 pixels, quad_level=3 -> ~5x faster

## [2]. **Translate_rotate simultaneous assignment**

**File:** `caustics/src/caustics/utils.py`  
**Location:** `translate_rotate()`  
**Description:**

- Remove unnecessary variable assignment.
- Replaced unnecessary negation and addition by a single subtraction.
- The JIT compiler should do this on its own, so this is of no concern for JITed code.

**Performance Impact:**

- Same setup as previously -> saves ~4ms (initial computation took ~16ms)

## [3]. **k_sersic polynomial efficient computation**

**File:** `caustics/src/caustics/light/func/sersic.py`  
**Location:** `k_sersic()`  
**Description:**

- Apply Horner’s rule for fast polynomial computation.
- Only keep integer divisions and a single tensor division.

**Performance Impact:**

- ~2x faster (~350us vs ~700us)

## [4]. **Simplified sie deflection angles computation**

**File:** `caustics/src/caustics/lenses/func/sie.py`  
**Location:** `reduced_deflection_angle_sie()`  
**Description:**

- Simplify the `q==1.0` numerical instability fix by removing a subtraction.
- Regrouped duplicate computation into intermediary variables.

**Performance Impact:**

- Saves ~500us (from a ~4ms computation)

## [5]. **NFW Helper Methods Optimization**

**File:** `caustics/src/caustics/lenses/func/nfw.py`  
**Location:** `_h_nfw()`, `_g_nfw()`, `_f_nfw()`  
**Description:**

- Simplify control flow of PyTorch where computation by precomputing values.
- Precomputing reduces the overhead of OpenXLA when not using JIT.

**Performance Impact:**

- Saves ~50us (from a ~1.5ms computation)

## [6]. **Removing unnecessary computation in NFW**

**File:** `caustics/src/caustics/lenses/func/nfw.py`  
**Location:** `scale_radius_nfw()`, `scale_density_nfw()`, `_f_nfw()`  
**Description:**

- Removed unnecessary multiplication in `scale_radius_nfw()`.
- Added intermediary variable for `(c + 1)` in `scale_density_nfw()`.

**Performance Impact:**

- Saves ~20us (one elementary operation)

## [7]. **Vectorizing EPL R_Omega**

**File:** `caustics/src/caustics/lenses/func/epl.py`  
**Location:** `_r_omega()`
**Description:**

- Vectorized the function. The function is much faster at the cost of more memory usage. The memory usage scales
  linearly with the number of iterations done, but computation speed scales faster than memory usage.
- Added a `memory_efficient` flag to the function that reverts to the old computation when set to `True`.
- Added a `steps` variable that interpolates between the two implementations by doing the vectorized computation in
  chunks. This implementation trades speed for memory usage.

**Performance Impact:**

- ~6-8x faster when used with 18 iterations to the cost of ~4-5x the memory usage. 

## [8]. **Interp2d Overhaul**

**File:** `caustics/src/caustics/utils.py`  
**Location:** `interp2d()`
**Description:**

- `torch.nn.functional.grid_sample()` has exactly the same cuda kernels as the old `interp2d` function, so replacing it with the provided torch function is faster. However, this torch function is has no forward differentiation defined, so we fall back to the old version when needed.
- There are multiple checks in the new function that optimize speed vs accuracy. Some parameters given to `grid_sample` make it so that the gradient doesn't propagate as expected (`align_corner=False`), but using this flag is faster.
- The function removes the usage of padding when the image has the same dimensions as the grid given, which speeds up the computation by a significant amount.
- The `interp2d` function now accepts a batch input and both `x` and `y` don't need to be flattened beforehand.
- Added the option to use a `bicubic` interpolation on top of the `nearest` and `bilinear`options.
- `interp2d` doesn't support channels, but this feature could be easily added if necessary.

**Performance Impact:**

Using a 128x128 grid and a 128x128 image.

### (`align_corners=True` and padding)
| Method            | Old Implementation (µs) ± Std | Peak Memory (MB) | New Implementation (µs) ± Std | Peak Memory (MB) | Speedup |
|-------------------|----------------------------|------------------|----------------------------|------------------|---------|
| **Nearest, Zeros** | 805.76 ± 643.35            | 2.85             | 421.86 ± 344.01            | 2.59             | ~1.91×  |
| **Linear, Zeros**  | 936.05 ± 554.61            | 3.70             | 400.97 ± 214.58            | 2.59             | ~2.33×  |
| **Nearest, Clamp** | 463.56 ± 280.11            | 2.84             | 446.36 ± 158.05            | 2.71             | ~1.04×  |
| **Linear, Clamp**  | 969.37 ± 575.50            | 3.69             | 234.94 ± 63.62             | 2.71             | ~4.13×  |

### `align_corners=False` and *not* using padding (difference in speed and memory in these runs are machine specific)
| Method  | Old Implementation (µs) ± Std | Peak Memory (MB) | New Implementation (µs) ± Std | Peak Memory (MB) | Speedup |
|---------|----------------------------|------------------|----------------------------|------------------|---------|
| **Nearest, Zeros**  | 658.60 ± 669.29            | 4.64             | 164.82 ± 474.26            | 4.29             | ~3.99×  |
| **Linear, Zeros**| 1070.75 ± 823.17           | 5.49             | 105.95 ± 48.75             | 4.29             | ~10.10× |
| **Nearest, Clamp**  | 366.89 ± 262.72            | 4.62             | 172.56 ± 181.32            | 4.43             | ~2.13×  |
| **Linear, Clamp** | 1159.63 ± 699.47           | 5.47             | 217.82 ± 365.38            | 4.43             | ~5.32×  |

## [9]. **Interp3d Overhaul**

**File:** `caustics/src/caustics/utils.py`  
**Location:** `interp3d()`  
**Description:**

- `torch.nn.functional.grid_sample()` also accept 5d inputs, so it is updated similarly to `interp2d`.


## [10]. **Use new `interp2d` in `pixelated_convergence`**

**File:** `caustics/src/caustics/lenses/pixelated_convergence.py`  
**Location:** `reduced_deflection_angle_pixelated_convergence()`  
**Description:**

- With the use of a batch possible with the new `interp2d`, the `reduced_deflection_angle_pixelated_convergence` was modified to use a batch in the fft, convolution and interpolation.

**Performance Impact:**

- Depends on the size of the image, but when small, `interp2d` takes most of the computation time and the speedup is ~4x.
- 
---

# Total Performance Gains

- Setup: Lens: SIE, Source: Sersic, LensLight: Sersic, 100x100 pixels, quad_level=3
    - Before: 34.8 ms ± 4.82 ms per loop (mean ± std. dev. of 10 runs, 10 loops each)
    - After: 3.96 ms ± 418 μs per loop (mean ± std. dev. of 10 runs, 10 loops each)
