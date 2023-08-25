In order to use phytorch (which enables gradients through cosmological parameters), do the following (assumes PyTorch > 2.0):

- Install Boost: https://www.boost.org/
- `pip install ninja`
- git clone [torchdiffeq](https://github.com/rtqichen/torchdiffeq), cd into torchdiffeq, and run `pip install -e .`
- git clone [phytorch](https://github.com/kosiokarchev/phytorch), cd into phytorch, and run `pip install -e .`
- In `(your local environment here)/lib/python3.9/site-packages/torch/include/c10/util/complex_utils.h`, change the following:
```
template <typename T> 
bool isnan(const c10::complex<T>& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

} // namespace std
```
To:
```
template <typename T> C10_HOST_DEVICE
bool isnan(const c10::complex<T>& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

} // namespace std
```

- In `phytorch/phytorch-extensions/common/complex.h`, change
```
DEF_COMPLEX_CHECK isnan(complex<T> a) { return isnan(a.real()) or isnan(a.imag()); }
```
To:
```
//DEF_COMPLEX_CHECK isnan(complex<T> a) { return isnan(a.real()) or isnan(a.imag()); }
```
- Run the following:
```
cd phytorch-extensions
python setup.py build_ext -b ../phytorch/extensions
```
