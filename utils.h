//
// Copyright (c) 2020 xinyan. All rights reserved.
// Created on 2020/4/2.
//

#ifndef PROGRESS_BAR_H_
#define PROGRESS_BAR_H_

#pragma once
#include <sstream>
#include <string>
#include <chrono>
#include <cassert>
#include <random>

#define Debug
#define verbose
//#define MULTI_THREAD

#define DELTA 0
//#define DELTA 0.001

template <typename T >
inline void check_range(T v, T a, T b) {
#ifdef Debug
  if (v < a || a >= b) {
    if (v < a) {
      std::cerr << " v_" << v << " is smaller than " << a;
    } else {
      std::cerr << " v_" << v << " is bigger than " << b;
    }
    throw std::runtime_error("exceed range");
  }
#endif
}

using std::string;
using std::vector;
using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

typedef float ValueType;
typedef int int_t;

class StopW {
  std::chrono::steady_clock::time_point time_begin;
 public:
  StopW() {
    reset();
  }

  long getElapsedTimeMicro(bool reset_time) {
    steady_clock::time_point time_end = steady_clock::now();
    long elapsed = duration_cast<microseconds>(time_end - time_begin).count();
    if (reset_time) {
      reset();
    }
    return elapsed;
  }

  void reset() {
    time_begin = steady_clock::now();
  }
};

class ProgressBar {
 public:
  ProgressBar(int len, string message): len_(len), cur_(0), star_(0) {
    std::cout << "0%   10   20   30   40   50   60   70   80   90   100%\t"
              << message
              << std::endl
              << "|----|----|----|----|----|----|----|----|----|----|"
              << std::endl;
  }

  ProgressBar& update(int i) {
    cur_ += i;
    int num_star = static_cast<int >(1.0 * cur_ / len_ * 50 + 1);
    if (num_star > star_) {
      for (int j = 0; j < num_star-star_; ++j) {
        std::cout << '*';
      }
      star_ = num_star;
      if (num_star == 51) {
        std::cout << std::endl;
      }
      std::cout << std::flush;
    }

    return *this;
  }

  ProgressBar& operator++() {
    return update(1);
  }

  ProgressBar& operator+=(int i) {
    return update(i);
  }

 private:
  int len_;
  int cur_;
  int star_;
};

bool exists_test(const char* name) {
  std::ifstream f(name);
  return f.good();
}



template <typename DataType>
void load_data(DataType*& data, int_t& dimension, int_t &cardinality, std::string input_path)
{
  std::ifstream fin(input_path.c_str(), std::ios::binary | std::ios::ate);
  if (!fin) {
    std::cout << "cannot open file " << input_path << std::endl;
    exit(1);
  }

  size_t fileSize = fin.tellg();
  fin.seekg(0, fin.beg);
  if (fileSize == 0) {
    std::cout << "file size is 0 " << input_path << std::endl;
    exit(1);
  }

  int dim;
  fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
  dimension = (size_t)dim;
  size_t bytesPerRecord = dimension * sizeof(DataType) + 4;
  if (fileSize % bytesPerRecord != 0) {
    std::cout << "File not aligned" << std::endl;
    exit(1);
  }
  cardinality = fileSize / bytesPerRecord;
  data = new DataType[cardinality * dimension];
  fin.read((char*)data, sizeof(DataType) * dimension);

  for (int i = 1; i < cardinality; ++i) {
    fin.read((char*)&dim, 4);
    assert(dim == dimension);
    fin.read((char*)(data + i * dimension), sizeof(DataType) * dimension);
  }
  fin.close();
}

template<typename T>
bool interval_check(const T* xb_, const T* xq_,
                    const T* rg_, const int_t d) {
  bool should_in = true;
  for (int di = 0; di < d; ++di) {
    T diff = *xb_++ - *xq_++;
    T range = *rg_++;
    if (diff > range || diff < -range) {
      should_in = false;
      break;
    }
  }
  return should_in;
}

template<typename T>
bool interval_check(const T* xb_, const T* x_rg_, 
                    const T* xq_, const T* q_rg_, const int_t d) {
  bool should_in = true;
  for (int di = 0; di < d; ++di) {
    T diff = *xb_++ - *xq_++;
    T range = *x_rg_++ + *q_rg_++;
    if (diff > range || diff < -range) {
      should_in = false;
      break;
    }
  }
  return should_in;
}

template<typename T, class Container>
std::vector<int_t >
interval_prune(const T* xb_, const T* xq_, const T* rg_,
               const Container& xs_id, const int_t d) {
  std::vector<int > res;
  for (int xi : xs_id) {
    if (interval_check(xb_ + xi * d, xq_, rg_, d)) {
      res.push_back(xi);
    }
  }
  return res;
}

template<typename T>
std::vector<std::vector<int_t > >
interval_search(const T* xb_, const T* xq_, const T* rg_,
                const int_t nx, const int_t nq, const int_t d) {
  std::vector<std::vector<int_t > > res(nq);

#ifdef verbose
  ProgressBar progress_bar(nq, "interval search");
#endif
#ifdef MULTI_THREAD
#pragma omp parallel for
#endif
  for (int qi = 0; qi < nq; ++qi) {
    for (int xi = 0; xi < nx; ++xi) {
      bool should_in = true;
      for (int di = 0; di < d; ++di) {

        T diff = xb_[xi * d + di] - xq_[qi * d + di];
        T range = rg_[qi * d + di];

        if (diff > range || diff < -range) {
          should_in = false;
          break;
        }
      }
      if (should_in) {
        res[qi].push_back(xi);
      }
    }
#ifdef verbose
#pragma omp critical
    {
      ++progress_bar;
    }
#endif
  }
  return res;
}

template <size_t p>
float int_power(float t) {
  if constexpr (p == 1) {
    return std::abs(t);
  }
  if constexpr (p == 2) {
    return t * t;
  }
  if constexpr (p == 3) {
    t = std::abs(t);
    return t * t * t;
  }
  if constexpr (p == 4) {
    t = t * t;
    return t * t;
  }
  if constexpr (p == 8) {
    t = t * t;
    t = t * t;
    return t * t;
  }
  if constexpr (p == 16) {
    t = t * t;
    t = t * t;
    t = t * t;
    return t * t;
  }
}
template<size_t p = 2, bool WEIGHTED = true>
float weighted_dist(const float * q, const float* w, const float* x, const int_t d) {
  float dist = 0.;
  for (int i = 0; i < d; ++i) {
    float t = (*q++ - *x++);
    if constexpr (WEIGHTED) {
      t /= (DELTA + *w++);
    }
    dist += int_power<p>(t);
  }
  return dist;
}

template <typename T>
vector<int_t> arg_sort(T* v, int_t n) {
  // initialize original index locations
  vector<int_t> idx(n);
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

template<typename T>
void normal_random_fill(T* ptr, int n, T mean, T std) {
  static std::random_device rd;
  static std::mt19937_64 eng(rd());
  static std::normal_distribution<T > dist(mean, std);
  for (int i = 0; i < n; ++i) {
    ptr[i] = dist(eng);
  }
}

template<typename T>
void uniform_random_fill(T* ptr, int n, T lower, T upper) {
  static std::random_device rd;
  static std::mt19937_64 eng(rd());
  static std::uniform_real_distribution<T > dist(lower, upper);
  for (int i = 0; i < n; ++i) {
    ptr[i] = dist(eng);
  }
}

vector<vector<int_t > > read_list(const char* file_name) {
  string line;
  std::ifstream file_stream(file_name);
  vector<vector<int_t > > res;

  while(getline(file_stream, line)) {
    std::istringstream iss(line);
    res.emplace_back();
    int_t id;
    while (iss >> id) {
      res.back().push_back(id);
    }
  }
  return res;
}

void write_list(const char* file_name, vector<vector<int_t > > lists) {
  std::ofstream file_stream(file_name);
  if (file_stream.is_open()) {
    for (vector<int_t> ids : lists) {
      if (!ids.empty()) {
        file_stream << ids[0];
      }
      for (int i = 1; i < ids.size(); ++i) {
        file_stream << " " << ids[i];
      }
      file_stream << "\n";
    }
    file_stream.close();
  }
  else {
    std::cout << "Unable to open file : " << file_name << std::endl;
  }
}

template <bool Compute=false>
vector<vector<int_t > >
load_gt(const char* fg, const float* xb,
        const float* xq, const float* rg,
        int_t nb, int_t nq, int_t dimension) {
  if constexpr (Compute) {
    return interval_search<float >(xb, xq, rg, nb, nq, dimension);
  }
  if (exists_test(fg)) {
    return read_list(fg);
  } else {
    StopW stop_w;
    auto ground_truth = interval_search<float >(xb, xq, rg, nb, nq, dimension);
    std::cout << "average brute force search time : "
              << stop_w.getElapsedTimeMicro(true) / nq << std::endl;
    write_list(fg, ground_truth);
    return ground_truth;
  }
}
#endif //PROGRESS_BAR_H_
