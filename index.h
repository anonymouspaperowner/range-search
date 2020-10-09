//
// Created by xydai on 2020/6/22.
//

#ifndef SS_INTERVAL__IMI_H_
#define SS_INTERVAL__IMI_H_
#include <array>
#include <utility>
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <queue>
#include <functional>
#include "utils.h"
#include "heap.h"

using std::array;
using std::vector;
using std::pair;
using std::function;
using std::shared_ptr;
using std::multiplies;
using std::accumulate;
using std::priority_queue;

#define M_IMI 2
using Coord = array<int_t, M_IMI>;
using FuncPointRange = function<bool ((int_t, int_t))>;
using FuncIntervalOverlap = function<bool (const float*, const float*, int_t, int_t, int_t)>;
using FuncWeightedDistance = function<float (const float*, int_t, int_t, int_t)>;

enum SearchMode {
  Threshold,
  TopK,
  Precision
};

// stands for inverted multi index
template<size_t K>
class IMISequence {
 public:
  explicit IMISequence(function<float (Coord)> dist)
                     : dist_(std::move(dist)), visit_{{false}, {false}} {
    Coord zero{0, 0};
    en_heap(zero);
  }

  bool has_next() const {
    return !heap_.empty();
  }

  pair<float, Coord > next() {
    const float dist = heap_.top().first;
    const Coord nearest = heap_.top().second;
    heap_.pop();
    visit_[nearest[0]][nearest[1]] = true;

#pragma unroll
    for (int i = 0; i < M_IMI; ++i) {
      int_t new_index = nearest[i] + 1;
      if(new_index  < K) { // next element exists
        Coord next_en = nearest;
        next_en[i] = new_index;
        if (should_en(next_en)) {
          en_heap(next_en);
        }
      }
    }
    return std::make_pair(dist, nearest);
  }

  void en_heap(const Coord &coord) {
    heap_.emplace(dist_(coord), coord);
  }

  bool should_en(const Coord &coord) {
#pragma unroll
    for (int i = 0; i < M_IMI; ++i) {
      if (coord[i] >= 1) { // preCoord existed
        Coord pre_one = coord;
        pre_one[i]--;
        if (!visit_[pre_one[0]][pre_one[1]]) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  bool                                 visit_[K][K];
  function<float (Coord)>              dist_;
  priority_queue<pair<float, Coord > > heap_;
};


template <int_t K, int_t M_PQ>
class ProductQuantization {
 protected:
  int_t                    d_;
  int_t                    nb_;
  array<int_t, M_PQ>       dims_;
  array<int_t, M_PQ+1>     offset_;
  const int_t*             codes_;
  const float*             centroids_;
  const float*             xb_;
  float*                   bucket_median_; // M x K x D
  float*                   bucket_width_; // M x K x D
 public:
  explicit ProductQuantization(int_t d, int_t nb,
                               const int_t* codes,
                               const float* centroids,
                               const float* xb)
                             : d_(d), nb_(nb), codes_(codes),
                               centroids_(centroids), xb_(xb),
                               bucket_median_(nullptr),
                               bucket_width_(nullptr) {
    init_dims_();
    init_bound_();
  }

  ~ProductQuantization() {
    delete [] bucket_median_;
    delete [] bucket_width_;
  }

  void init_bound_() {
    const int_t max_d_ = dims_[0];
    bucket_median_ = new float[max_d_ * K * M_PQ];
    bucket_width_ = new float[max_d_ * K * M_PQ];

    std::memcpy(bucket_median_, centroids_, max_d_ * K * M_PQ * sizeof(float));
    std::memset(bucket_width_, 0, max_d_ * K * M_PQ * sizeof(float));
    for (int i = 0 ; i < nb_; ++i) {
#pragma unroll
      for (int m = 0 ; m < M_PQ; ++m) {
        int k = codes_[i * M_PQ + m];
        for (int di = 0; di < dims_[m]; ++di) {
          float value = xb_[i * d_ + di + offset_[m]];
          int_t c_idx = m * K * max_d_ + k * max_d_ + di;
          float upper_bound = bucket_median_[c_idx] + bucket_width_[c_idx];
          float lower_bound = bucket_median_[c_idx] - bucket_width_[c_idx];
          if (value > upper_bound) {
            upper_bound = value;
          } else if (value < lower_bound) {
            lower_bound = value;
          }
          bucket_median_[c_idx] = (upper_bound + lower_bound) / 2.f;
          bucket_width_[c_idx] = (upper_bound - lower_bound) / 2.f;
        }
      }
    }
  }

  void init_dims_() {
    auto sub_dimension = d_ / M_PQ;
    auto reminder = d_ % M_PQ;

    for (int i = 0; i < M_PQ; ++i) {
      dims_[i] = sub_dimension + (i < reminder ? 1 : 0);
    }

    offset_[0] = 0;
    for (int i = 1; i <= M_PQ; ++i) {
      offset_[i] = offset_[i-1] + dims_[i-1];
    }
  }

  const int_t* get_code(int_t i) const {
    return codes_ + M_PQ * i;
  }

  bool exact_interval_check(int m, int i, int query_id,
                            const FuncIntervalOverlap& is_interval_overlap) const  {
    const int_t max_d_ = dims_[0];
    int_t c_idx_i = m * K * max_d_ + i * max_d_;
    return is_interval_overlap(this->bucket_median_ + c_idx_i,
        this->bucket_width_ + c_idx_i, this->dims_[m], this->offset_[m], query_id);
  }

  shared_ptr<float> compute_dist(int_t query_id, const FuncWeightedDistance& distance) const {
    shared_ptr<float> dists(new float[M_PQ * K]);
    const int_t max_d_ = dims_[0];
    float* dist = dists.get();
    const float* centroids = centroids_;
    for (int m = 0; m < M_PQ; ++m) {
      auto sub_d = dims_[m];
      auto offset = offset_[m];
      for (int k = 0; k < K; ++k) {
        *(dist++) = distance(centroids, sub_d, offset, query_id);
        centroids += max_d_;
      }
    }
    return dists;
  }

  vector<int_t> exact_pq_search(int_t *num_probe, int_t query_id,
                                const FuncPointRange& is_point_in_range,
                                const FuncIntervalOverlap& is_interval_overlap) const {
    int_t num_probe_items_ = 0;
    vector<int_t > valid_items;

    bool possible_PQ[M_PQ][K];
#pragma unroll
    for (int m = 0; m < M_PQ;  ++m) {
      for (int i = 0 ; i < K; ++i) {
        possible_PQ[m][i] = this->exact_interval_check(m, i, query_id, is_interval_overlap);
      }
    }

    for (int_t item_id = 0; item_id < nb_; item_id++) {
      bool pass_verify = true;
      {
        const int_t* code_k = this->get_code(item_id);
        for (int m = 0; m < M_PQ; ++m) {
          auto k = code_k[m];
          if (!possible_PQ[m][k]) {
            pass_verify = false;
            break;
          }
        }
      }
      if (pass_verify) {
        num_probe_items_++;
        if (is_point_in_range(item_id, query_id)) {
          valid_items.push_back(item_id);
        }
      }
    }
    *num_probe = num_probe_items_;
    return valid_items;
  }

  vector<int_t> brute_force_search(int_t *num_probe, int query_id,
                                   const FuncPointRange& is_point_in_range) const {
    vector<int_t > valid_items;
    for (int_t item_id = 0; item_id < nb_; item_id++) {
      if (is_point_in_range(item_id, query_id)) {
        valid_items.push_back(item_id);
      }
    }
    *num_probe = nb_;
    return valid_items;
  }
};


template <int_t K, int_t Ks_PQ, int_t M_PQ>
class InvertMultiIndex : public ProductQuantization<K, M_IMI >{
 private:
  int_t                              nb_;
  ProductQuantization<Ks_PQ, M_PQ >* pq_ptr_;
  vector<int_t>                      buckets_[K][K];  // rank = M
 public:
  explicit InvertMultiIndex(int_t d, int_t nb,
                            const int_t* codes,
                            const float* centroids,
                            const float* xb)
                          : ProductQuantization<K, M_IMI >(
                              d, nb, codes, centroids, xb) {
    nb_ =nb;
    for (int i = 0; i < nb; ++i) {
      buckets_[codes[i * M_IMI]][codes[i * M_IMI + 1]].push_back(i);
    }
  }

  void set_pq(ProductQuantization<Ks_PQ, M_PQ >* pq_ptr) {
    pq_ptr_ = pq_ptr;
  }

  void stat() const {
    int max_ = 0;
    int min_ = nb_;
    int non_empty = 0;
    for (const auto& bs : buckets_) {
      for (const auto& b : bs) {
        if (b.size() > max_)
          max_ = b.size();
        if (b.size() > 0) {
          if (b.size() < min_) {
            min_ = b.size();
          }
          non_empty++;
        }
      }
    }
    std::cout << "max_      : " << max_ << "\n";
    std::cout << "min_      : " << min_ << "\n";
    std::cout << "non_empty : " << non_empty << "\n";
    std::cout << "avg_size  : " << nb_ / non_empty << "\n";
  }

  template<bool IMI=true, bool PQ=true >
  vector<int_t> exact_search(
      int_t *num_probe, int query_id,
      const FuncPointRange& is_point_in_range,
      const FuncIntervalOverlap& is_interval_overlap,
      const FuncWeightedDistance& distance) const {
    if constexpr (!IMI) {
      if constexpr (PQ) {
        return pq_ptr_->exact_pq_search(num_probe, query_id, is_point_in_range, is_interval_overlap);
      } else {
        return this->brute_force_search(num_probe, query_id, is_point_in_range);
      }
    }
    int_t num_probe_items_ = 0;
    vector<int_t > valid_items;
    
    const int_t max_d_ = this->dims_[0];
    array<vector<int_t >, M_IMI >  possible_buckets = {vector<int_t>(),
                                                       vector<int_t>()};
#pragma unroll
    for (int m = 0; m < M_IMI; ++m) {
      for (int i = 0; i < K; ++i) {
        if (this->exact_interval_check(m, i, query_id, is_interval_overlap)) {
          possible_buckets[m].push_back(i);
        }
      }
    }

    bool possible_PQ[M_PQ][Ks_PQ];
    if constexpr (PQ) {
#pragma unroll
      for (int m = 0; m < M_PQ;  ++m) {
        for (int i = 0 ; i < Ks_PQ; ++i) {
          possible_PQ[m][i] = pq_ptr_->exact_interval_check(m, i, query_id, is_interval_overlap);
        }
      }
    }

    for (int_t id_i : possible_buckets[0]) {
      for (int_t id_j : possible_buckets[1]) {
        const vector<int_t>& bucket = buckets_[id_i][id_j];
        num_probe_items_ += bucket.size();
        for (int_t item_id : bucket) {
          bool pass_verify = true;
          if constexpr (PQ) {
            const int_t* code_k = pq_ptr_->get_code(item_id);
            for (int m = 0; m < M_PQ; ++m) {
              auto k = code_k[m];
              if (!possible_PQ[m][k]) {
                pass_verify = false;
                break;
              }
            }
          }
          if (pass_verify && is_point_in_range(item_id, query_id)) {
            valid_items.push_back(item_id);
          }
        }
      }
    }

    *num_probe = num_probe_items_;
    return valid_items;
  }

  template<SearchMode mode, bool PQ=false, bool PQRank=true >
  vector<int_t> imi_probe(const float threshold,
                          int_t *num_probe, int query_id,
                          const FuncPointRange& is_point_in_range,
                          const FuncIntervalOverlap& is_interval_overlap,
                          const FuncWeightedDistance& p_weighted_distance) const {
    shared_ptr<float > dist = this->compute_dist(query_id, p_weighted_distance);
    shared_ptr<float > pq_dist = pq_ptr_->template compute_dist(query_id, p_weighted_distance);
    array<vector<int_t >, M_IMI> indices;
#pragma unroll
    for (int m = 0; m < M_IMI; ++m) {
      indices[m] = arg_sort(dist.get() + m * K, K);
    }

    auto dist_caller = [&](Coord coord) {
      float res = 0.0f;
#pragma unroll
      for (int m = 0; m < M_IMI; ++m) {
        res += dist.get()[m * K + indices[m][coord[m]]];
      }
      return res;
    };

    IMISequence<K> imi(dist_caller);
    vector<int_t > valid_items;
    int k = std::max(int(threshold / 10), 1);
    float * simi = nullptr;
    int64_t * idxq = nullptr;
    bool possible_PQ[M_PQ][Ks_PQ];
    if constexpr (PQ) {

      if constexpr (PQRank) {
        simi = new float[k];
        idxq = new int64_t[k];
        faiss::maxheap_heapify (k, simi, idxq);
      }
      else {
#pragma unroll
        for (int m = 0; m < M_PQ;  ++m) {
          for (int i = 0 ; i < Ks_PQ; ++i) {
            possible_PQ[m][i] = pq_ptr_->exact_interval_check(m, i, query_id, is_interval_overlap);
          }
        }
      }
    }
    
    int_t num_probe_items_ = 0;
    while (imi.has_next()) {
      pair<float, Coord > p = imi.next();
      if constexpr (TopK == mode) {
        if (num_probe_items_ >= threshold) {
          break;
        }
      } else if constexpr (Threshold == mode) {
        if (p.first > threshold) {
          break;
        }
      } else if constexpr (Precision == mode) {
        if (num_probe_items_ > 100) {
          if (1.0 * valid_items.size() / num_probe_items_ < threshold) {
            break;
          }
        }
      }
      
      auto first = indices[0][p.second[0]];
      auto second = indices[1][p.second[1]];
      const vector<int_t>& bucket = buckets_[first][second];
      num_probe_items_ += bucket.size();
      for (int_t item_id : bucket) {
        if constexpr (PQ && PQRank) {
          float distance = 0;
          const int_t* code_k = pq_ptr_->get_code(item_id);
          for (int m = 0 ; m < M_PQ; ++ m) {
            distance +=  pq_dist.get()[m * Ks_PQ + code_k[m]];
          }
          if (distance < simi[0]) {
            faiss::maxheap_pop (k, simi, idxq);
            faiss::maxheap_push (k, simi, idxq, distance, item_id);
          }
        }
        else if constexpr (PQ && !PQRank) {
          bool pass_verify = true;
          const int_t* code_k = pq_ptr_->get_code(item_id);
          for (int m = 0; m < M_PQ; ++m) {
            auto k = code_k[m];
            if (!possible_PQ[m][k]) {
              pass_verify = false;
              break;
            }
          }
          if (pass_verify &&
              is_point_in_range(item_id, query_id)) {
            valid_items.push_back(item_id);
          }
        }
        else {
          if (is_point_in_range(item_id, query_id)) {
            valid_items.push_back(item_id);
          }
        }
      }
    }

    if constexpr (PQ && PQRank) {
      for (int i = 0; i < std::min(num_probe_items_, k); ++i) {
        int64_t item_id = idxq[i];
        if (is_point_in_range(item_id, query_id)) {
          valid_items.push_back(item_id);
        }
      }
      delete[] simi;
      delete[] idxq;
    }

    *num_probe = num_probe_items_;
    return valid_items;
  }
};
#endif //SS_INTERVAL__IMI_H_
