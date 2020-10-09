#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "progress_bar.h"
namespace py = pybind11;

const int OnPoints = 0;
const int OnIntervals = 1;

template<typename T, int Mode>
void
check_shape(py::array_t<T > xb, py::array_t<T > xq, py::array_t<T > rg) {
  auto xb_ = xb.template unchecked<2>();
  auto xq_ = xq.template unchecked<2>();
  auto rg_ = rg.template unchecked<2>();
  if (xb_.shape(1) != xq_.shape(1))
    throw std::runtime_error("Incompatible dimensions for xb and xq");
  if constexpr(Mode == OnPoints) {
    if (rg_.shape(0) != xq_.shape(0))
      throw std::runtime_error("Incompatible dimensions for rg and xq");
  }
  if constexpr(Mode == OnIntervals) {
    if (rg_.shape(0) != xb_.shape(0))
      throw std::runtime_error("Incompatible dimensions for rg and xb");
  }
}


template<typename T, int Mode>
std::vector<std::vector<int > >
interval_search(py::array_t<T > xb, py::array_t<T > xq, py::array_t<T > rg) {

  auto xb_ = xb.template unchecked<2>();
  auto xq_ = xq.template unchecked<2>();
  auto rg_ = rg.template unchecked<2>();
  const int nx = xb_.shape(0);
  const int nq = xq_.shape(0);
  const int d = xb_.shape(1);

  std::vector<std::vector<int > > res(nq);

  check_shape<T, Mode>(xb, xq, rg);

  ProgressBar progress_bar(nq, "interval search");

#pragma omp parallel for
  for (int qi = 0; qi < nq; ++qi) {
    for (int xi = 0; xi < nx; ++xi) {
      bool should_in = true;
      for (int di = 0; di < d; ++di) {
        T diff = xb_(xi, di) - xq_(qi, di);
        T range;
        if constexpr(Mode == OnPoints) {
          range = rg_(qi, di);
        }
        if constexpr(Mode == OnIntervals) {
          range = rg_(xi, di);
        }
        if (diff > range || diff < -range) {
          should_in = false;
          break;
        }
      }
      if (should_in) {
        res[qi].push_back(xi);
      }
    }
#pragma omp critical
    {
      ++progress_bar;
    }
  }
  return res;
}

template<typename T, int Mode>
std::vector<std::vector<int > >
sparse_interval_search(py::array_t<T > xb,
                       py::array_t<T > xq,
                       py::array_t<T > rg,
                       std::vector<std::vector<int > > idx) {

  auto xb_ = xb.template unchecked<2>();
  auto xq_ = xq.template unchecked<2>();
  auto rg_ = rg.template unchecked<2>();
  const int nx = xb_.shape(0);
  const int nq = xq_.shape(0);
  const int d = xb_.shape(1);

  std::vector<std::vector<int > > res(nq);

  check_shape<T, Mode>(xb, xq, rg);

  ProgressBar progress_bar(nq, "interval search");

#pragma omp parallel for
  for (int qi = 0; qi < nq; ++qi) {
    for (int xi = 0; xi < nx; ++xi) {
      bool should_in = true;
      for (int di : idx[qi]) {
        T diff = xb_(xi, di) - xq_(qi, di);
        T range;
        if constexpr(Mode == OnPoints) {
          range = rg_(qi, di);
        }
        if constexpr(Mode == OnIntervals) {
          range = rg_(xi, di);
        }
        if (diff > range || diff < -range) {
          should_in = false;
          break;
        }
      }
      if (should_in) {
        res[qi].push_back(xi);
      }
    }
#pragma omp critical
    {
      ++progress_bar;
    }
  }
  return res;
}

template<typename T, int Mode>
py::array_t<T >
weighted_dist(py::array_t<T > xb,
              py::array_t<T > xq,
              py::array_t<T > rg, int p) {

  auto xb_ = xb.template unchecked<2>();
  auto xq_ = xq.template unchecked<2>();
  auto rg_ = rg.template unchecked<2>();
  const int nx = xb_.shape(0);
  const int nq = xq_.shape(0);
  const int d = xb_.shape(1);

  auto res = py::array_t<T >(nq * nx);
  py::buffer_info buf = res.request();
  auto ptr = (T *) buf.ptr;

  check_shape<T, Mode>(xb, xq, rg);

  ProgressBar progress_bar(nq, "weighted dist");

#pragma omp parallel for
  for (int qi = 0; qi < nq; ++qi) {
    T* ptr_i = ptr + qi * nx;
    for (int xi = 0; xi < nx; ++xi, ++ptr_i) {
      T dist_sum = 0;
      for (int di = 0; di < d; ++di) {
        T diff = xb_(xi, di) - xq_(qi, di);
        T range;
        if constexpr(Mode == OnPoints) {
          range = rg_(qi, di);
        }
        if constexpr(Mode == OnIntervals) {
          range = rg_(xi, di);
        }
        if (diff < 0) {
            diff = - diff / range;
        } else {
            diff = diff / range;
        }
        T p_diff = diff;
        for (int order = 1; order < p; order++) {
            p_diff *= diff;
        }
        dist_sum += p_diff;
      }
      *ptr_i = dist_sum;
    }
#pragma omp critical
    {
        ++progress_bar;
    }
  }
  return res;
}

template<typename T>
bool interval_check(const T* xb_,
                    const T* xq_,
                    const T* rg_, 
                    const int d, 
                    const float scale) {
  bool should_in = true;
  for (int di = 0; di < d; ++di) {
    T diff = *xb_++ - *xq_++;
    T range = scale * (*rg_++);
    if (diff > range || diff < -range) {
      should_in = false;
      break;
    }
  }
  return should_in;
}

template<typename T>
int range_count(const T* xb_, 
                const T* xq_,
                const T* rg_, 
                const int d,
                const int nx,  
                const float scale) {
  int count = 0;
  for (int i = 0; i < nx; ++i, xb_+=d) {
    if (interval_check(xb_, xq_, rg_, d, scale)) {
      ++count;
    }
  }
  return count;
}

template<typename T>
T _find_scale(const T* xb_, 
              const T* xq_,
              const T* rg_, 
              const int d,
              const int nx,  
              const int lower, 
              const int upper) {
    T min_scale = 0 ;
    T max_scale = 1.0;
    constexpr int max_iter = 64;
    for (int i = 0; i < max_iter; ++i) {
      auto count = range_count(xb_, xq_, rg_, d, nx, max_scale * 2.0);
      if (count >= lower) {
        if (i > 0) {
          min_scale = max_scale;
        }
        max_scale *= 2;
        break;
      }
      max_scale *= 2;
      if (i == max_iter - 1) {
        std::cerr << "count: " << count 
                  << "\tmax_scale : " << max_scale 
                  << "failed to find apprropriate scale, too little result\n";
        return max_scale;
      }
    }

    for (int i = 0; i < max_iter; ++i) {
      T middle_scale = (min_scale + max_scale) / 2.;
      int middle_count = range_count(xb_, xq_, rg_, d, nx, middle_scale);
      if (middle_count > upper) {
        max_scale = middle_scale;
      } 
      else if (middle_count < lower) {
        min_scale = middle_scale;
      }
      else {
        return middle_scale;
      }

      if (i == max_iter - 1) {
        std::cerr << "middle_count: " << middle_count 
                  << "\tmiddle_scale : " <<  middle_scale 
                  << "\tfailed to find apprropriate scale, too many iters\n";
        return max_scale;
      }
    }
    return min_scale;
}

template<typename T>
py::array_t<T >
find_scale(py::array_t<T > xb, 
           py::array_t<T > xq, 
           py::array_t<T > rg, 
           const int lower, 
           const int upper) {

  auto xb_ = xb.template unchecked<2>();
  auto xq_ = xq.template unchecked<2>();
  auto rg_ = rg.template unchecked<2>();
  const int nx = xb_.shape(0);
  const int nq = xq_.shape(0);
  const int d = rg_.shape(1);
  const T* xb_ptr = (T*) xb.request().ptr;
  const T* xq_ptr = (T*) xq.request().ptr;
  const T* rg_ptr = (T*) rg.request().ptr;

  auto res = py::array_t<T >(nq);
  py::buffer_info buf = res.request();
  auto ptr = (T *) buf.ptr;

  check_shape<T, OnPoints>(xb, xq, rg);

  ProgressBar progress_bar(nq, "find scale");

#pragma omp parallel for
  for (int qi = 0; qi < nq; ++qi) {

    ptr[qi] = _find_scale<T >(xb_ptr, xq_ptr + qi * d, rg_ptr + qi * d, d, nx, lower, upper);
#pragma omp critical
    {
        ++progress_bar;
    }
  }
  return res;
}

PYBIND11_MODULE(cgt, m) {
    m.doc() = "pybind11 cgt plugin"; // optional module docstring
    m.def("interval_search_on_float_point",
          &interval_search<float, OnPoints>,
          "A function which execute query by intervals"
    );
    m.def("interval_search_on_double_point",
          &interval_search<double, OnPoints>,
          "A function which execute query by intervals"
    );
    m.def("interval_search_on_float_interval",
          &interval_search<float, OnIntervals>,
          "A function which execute query by intervals"
    );
    m.def("interval_search_on_double_interval",
          &interval_search<double, OnIntervals>,
          "A function which execute query by intervals"
    );

    m.def("sparse_interval_search_on_float_point",
          &sparse_interval_search<float, OnPoints>,
          "A function which execute sparse query by intervals"
    );
    m.def("sparse_interval_search_on_double_point",
          &sparse_interval_search<double, OnPoints>,
          "A function which execute sparse query by intervals"
    );
    m.def("sparse_interval_search_on_float_interval",
          &sparse_interval_search<float, OnIntervals>,
          "A function which execute sparse query by intervals"
    );
    m.def("sparse_interval_search_on_double_interval",
          &sparse_interval_search<double, OnIntervals>,
          "A function which execute sparse query by intervals"
    );

    m.def("weighted_dist_on_float_point",
          &weighted_dist<float, OnPoints>,
          "A function which calculate weighted p-distance"
    );
    m.def("weighted_dist_on_double_point",
          &weighted_dist<double, OnPoints>,
          "A function which calculate weighted p-distance"
    );
    m.def("weighted_dist_on_float_interval",
          &weighted_dist<float, OnIntervals>,
          "A function which calculate weighted p-distance"
    );
    m.def("weighted_dist_on_double_interval",
          &weighted_dist<double, OnIntervals>,
          "A function which calculate weighted p-distance"
    );

    m.def("find_scale_on_float_interval",
          &find_scale<float >,
          "A function which find_scale of the witdth of range"
    );
    m.def("find_scale_on_double_interval",
          &find_scale<double >,
          "A function which find_scale of the witdth of range"
    );
}

