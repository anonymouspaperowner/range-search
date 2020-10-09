#include <boost/functional/hash.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <unordered_map>
#include <vector>
#include <set>
#include "utils.h"
#include "rss.h"

using namespace std;

extern int_t nb;
extern int_t nq;
extern int_t dimension;
extern float *xb;
extern float *xq;
extern float *rg;
extern vector<vector<int > > gt;

using std::shared_ptr;
using std::vector;
using std::unordered_map;


template<typename T>
class Hasher {
 public:
  size_t operator()(const std::vector<T>& vec) const {
    size_t seed = 0;
    for (T v : vec) {
      boost::hash_combine(seed, v);
    }
    return seed;
  }
};

class E2LSH {
 private:
  int k_;
  int d_;
  float r_;
  int size_;
  shared_ptr<float > bias_;
  shared_ptr<float > projectors_;
  unordered_map<vector<int>, vector<int>, Hasher<int > >  hash_map_;
 public:
  E2LSH(int k, int d, float r) : k_(k), d_(d), r_(r), size_(0),
                                 bias_(new float[k]),
                                 projectors_(new float[k * d]) {
    uniform_random_fill(bias_.get(), k, 0.f, r);
    normal_random_fill(projectors_.get(), k * d, 0.f, 1.f);
  }

  size_t size() const {
    return hash_map_.size();
  }

  void add(const float* x, int n) {
    for (int i = 0; i < n; ++i) {
      auto key = hash(x + i * d_);
      hash_map_[key].push_back(size_++);
    }
  }

  const vector<int >& search(const float* x) {
    auto key = hash(x);
    return hash_map_[key];
  }

  set<int > range_search(const float* x, const float* r) {
    vector<int > hv_upper(k_);
    vector<int > hv_lower(k_);
    vector<int > hv(k_);
    double distinct_cnt = 1;
    for (int i = 0; i < k_; ++i) {
      float projected_x = 0.;
      float projected_r = 0.;
      for (int di = 0; di < d_; ++di) {
        projected_x += x[di] * projectors_.get()[i * d_ + di];
        projected_r += r[di] * std::abs(projectors_.get()[i * d_ + di]);
      }
      hv_lower[i] = std::ceil((projected_x - projected_r + bias_.get()[i])  / r_) ;
      hv_upper[i] = std::ceil((projected_x + projected_r + bias_.get()[i])  / r_) ;
      hv[i] = std::ceil((projected_x + bias_.get()[i])  / r_) ;

      distinct_cnt *=  (hv_upper[i] - hv_lower[i] + 1);
    }
//    std::cout << "distinct_cnt : " << distinct_cnt << "\n";

    const vector<int >& res = hash_map_[hv];
    std::set<int > s(res.begin(), res.end());

    for(int i = 0; i < k_;  i++) {
      if (hv_lower[i] > hv[i]) {
        hv[i]--;
        const vector<int >& r = hash_map_[hv];
        s.insert(r.begin(), r.end());
        hv[i]++;
      }
      if (hv_upper[i] < hv[i]) {
        hv[i]++;
        const vector<int >& r = hash_map_[hv];
        s.insert(r.begin(), r.end());
        hv[i]--;
      }
    }
    return s;
  }

  vector<int > hash(const float* x) {
    vector<int > hv(k_);
    for (int i = 0; i < k_; ++i) {
      float projected = std::inner_product(x, x + d_, projectors_.get() + i * d_, 0.);
      hv[i] = std::ceil((projected + bias_.get()[i])  / r_) ;
    }
    return hv;
  }
};

class SRP {
  typedef unsigned long long HV;
 private:
  int k_;
  int d_;
  int size_;
  shared_ptr<float > projectors_;
  unordered_map<HV, vector<int> >  hash_map_;
 public:
  SRP(int k, int d, float r) : k_(k), d_(d), size_(0),
                        projectors_(new float[k * d]) {
    normal_random_fill(projectors_.get(), k * d, 0.f, 1.f);
  }

  size_t size() const {
    return hash_map_.size();
  }

  void add(const float* x, int n) {
    for (int i = 0; i < n; ++i) {
      auto key = hash(x + i * d_);
      hash_map_[key].push_back(size_++);
    }
  }

  const vector<int >& search(const float* x) {
    auto key = hash(x);
    return hash_map_[key];
  }

  set<int > range_search(const float* x, const float* r) {
    const vector<int >& vct = search(x);
    return set<int >(vct.begin(), vct.end());
  }

  HV hash(const float* x) {
    HV hv = 0;
    for (int i = 0; i < k_; ++i) {
      hv *= 2;
      float projected = std::inner_product(x, x + d_, projectors_.get() + i * d_, 0.);
      hv |= projected > 0;
    }
    return hv;
  }
};

template<class LSHTable >
class LSH {
 private:
  vector<LSHTable > lsh_;

 public:
  LSH(int l, int k, int d, float r) {
    for (int i = 0 ; i < l; i++) {
      lsh_.emplace_back(k, d, r);
    }
  }

  float add(const float* x, int n) {
#pragma omp parallel for
    for(int i = 0; i < lsh_.size(); i++) {
      lsh_[i].add(x, n);
    }

    float sum_size = 0;
    for (const auto& lsh : lsh_) {
      sum_size += lsh.size();
    }
    return sum_size / lsh_.size();
  }

  std::set<int > search(const float* x, int l) {
    std::set<int > s;
    for(int i = 0; i < l;  i++) {
      auto r = lsh_[i].search(x);
      s.insert(r.begin(), r.end());
    }
    return s;
  }

  std::set<int > range_search(const float* x, const float* r, int l) {
    std::set<int > s;
    for(int i = 0; i < l;  i++) {
      auto res = lsh_[i].range_search(x, r);
      s.insert(res.begin(), res.end());
    }
    return s;
  }
};
