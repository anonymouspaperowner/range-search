//
// Created by xydai on 2020/6/28.
//
#include <limits>
#include <cstring>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

template<typename T, int p>
T power(T x) {
  if (p == 0) {
    return 1;
  }
  else if (p == 1) {
    return x;
  }
  else if (p % 2 == 0) {
    T root = power<T, p / 2>(x);
    return root * root;
  }
  else { // if (p % 2 == 1) {
    T root = power<T, p / 2>(x);
    return root * root * x;
  }
}

template<typename T, int p>
void _vq(py::array_t<T > xs,
        py::array_t<T > centroid, 
        py::array_t<int > codes, 
        py::array_t<T  > dists) {
  auto xs_ = xs.template unchecked<2>();
  auto centroid_ = centroid.template unchecked<2>();

  const int nx = xs_.shape(0);
  const int d = xs_.shape(1);
  const int ks = centroid_.shape(0);
  if (d != centroid_.shape(1))
    throw std::runtime_error("Incompatible dimensions for x and centroid");
  
  auto codes_ptr = (int *) codes.request().ptr;
  auto dists_ptr = (T *) dists.request().ptr;

#pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    int min_id = -1;
    T min_dist = std::numeric_limits<T>::max();
    for (int k = 0; k < ks; ++k) {
      T dist = 0;
      for (int di = 0; di < d; ++di) {
        auto distortion = std::abs(centroid_(k, di) - xs_(i, di));
        dist += power<T, p>(distortion);
      }
      if (dist < min_dist) {
        min_id = k;
        min_dist = dist;
      }
    }
    codes_ptr[i] = min_id;
    dists_ptr[i] = min_dist;
  }
}

template<typename T, int p>
void _update_centroids(py::array_t<T > xs,
                       py::array_t<T > centroid,
                       py::array_t<int > codes, int iteration) {
  auto xs_ = xs.template unchecked<2>();
  auto codes_ = codes.template unchecked<1>();
  auto centroid_ = centroid.template mutable_unchecked<2>();

  const int nx = xs_.shape(0);
  const int d = xs_.shape(1);
  const int ks = centroid_.shape(0);

  if (d != centroid_.shape(1))
    throw std::runtime_error("Incompatible dimensions for x and centroid");

#pragma omp parallel for
  for (int di = 0; di < d; ++di) {
    std::vector<T> gradient(ks);
    std::vector<T> hessian(ks);

    for (int iter = 0; iter < iteration; ++iter) {
      std::memset(gradient.data(), 0, sizeof(T) * ks);
      std::memset(hessian.data(), 0, sizeof(T) * ks);
      // calculate gradient and hess
      for (int i = 0; i < nx; ++i) {
        auto k = codes_(i);
        auto distortion = centroid_(k, di) - xs_(i, di);
        auto power_pm2 = power<T, p-2>(distortion);
        if (p % 2 == 0 || distortion > 0) {
          gradient[k] += /*p * */ distortion * power_pm2;
          hessian[k] += /*p * */  (p -1) * power_pm2;
        } else {
          gradient[k ] -= /*p * */ distortion * power_pm2;
          hessian[k] -= /*p * */  (p -1) * power_pm2;
        }
      }
      // update centroids
      bool should_break = true;
      for (int k = 0; k < ks; ++k) {
        auto g = gradient[k];
        auto h = hessian[k];

        if (h != 0) {
          centroid_(k, di) -= g / h;
          should_break = false;
        }
      }
      if (should_break) {
        break;
      }
    }
  }
}


template<typename T>
void vq(py::array_t<T > xs,
        py::array_t<T > centroid,
        py::array_t<int > codes,
        py::array_t<T  > dists, int p) {
  if (p == 1) {
    _vq<T, 1>(xs, centroid, codes, dists);
  }
  else if (p == 2) {
    _vq<T, 2>(xs, centroid, codes, dists);
  }
  else if (p == 3) {
    _vq<T, 3>(xs, centroid, codes, dists);
  }
  else if (p == 4) {
    _vq<T, 4>(xs, centroid, codes, dists);
  }
  else if (p == 6) {
    _vq<T, 6>(xs, centroid, codes, dists);
  }
  else if (p == 8) {
    _vq<T, 8>(xs, centroid, codes, dists);
  }
  else if (p == 16) {
    _vq<T, 16>(xs, centroid, codes, dists);
  }
  else {
    throw std::runtime_error("un supported p-value in vq_lp.");
  }
}


template<typename T>
void update_centroids(py::array_t<T > xs,
                      py::array_t<T > centroid,
                      py::array_t<int > codes, int p, int iteration) {
  if (p == 1) {
    _update_centroids<T, 1>(xs, centroid, codes, iteration);
  }
  else if (p == 2) {
    _update_centroids<T, 2>(xs, centroid, codes, iteration);
  }
  else if (p == 3) {
    _update_centroids<T, 3>(xs, centroid, codes, iteration);
  }
  else if (p == 4) {
    _update_centroids<T, 4>(xs, centroid, codes, iteration);
  }
  else if (p == 6) {
    _update_centroids<T, 6>(xs, centroid, codes, iteration);
  }
  else if (p == 8) {
    _update_centroids<T, 8>(xs, centroid, codes, iteration);
  }
  else if (p == 16) {
    _update_centroids<T, 16>(xs, centroid, codes, iteration);
  }
  else {
    throw std::runtime_error("un supported p-value in vq_lp update_centroids.");
  }
}


PYBIND11_MODULE(cpp_vq_lp, m) {
  m.doc() = "pybind11 p-vq_p plugin"; // optional module docstring
  m.def("vq_p_on_float",
        &vq<float>,
        "A function which execute vq_p by lp distance"
  );
  m.def("vq_p_on_double",
        &vq<double>,
        "A function which execute vq_p by lp distance"
  );

  m.def("update_centroids_on_float",
        &update_centroids<float>,
        "A function which execute vq_p by lp distance"
  );
  m.def("update_centroids_on_double",
        &update_centroids<double>,
        "A function which execute vq_p by lp distance"
  );
}
