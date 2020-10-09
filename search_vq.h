#include <boost/geometry/index/rtree.hpp>
#include "index.h"
#include "utils.h"

using namespace std;

extern int_t nb;
extern int_t nq;
extern int_t dimension;
extern float *xb;
extern float *xq;
extern float *rg;
extern vector<vector<int > > idx;
extern vector<vector<int > > gt;


/**
 * \brief read scale with shape [1 x dim] from f-vecs file
 * \param scale_path
 * \param d
 * \return
 */
float* load_scale(const char* scale_path, const int d) {
  float * scale;
  int_t dim, nx;
  load_data(scale, dim, nx, scale_path);
  if(dim != d || nx != 1) {
    throw std::runtime_error("codes size mis-matched!");
  }
  return scale;
}
/**
 * \brief read codes with shape [n x m] from i-vecs file
 * \param codes_path
 * \param n the number of items
 * \param m the number of code book
 * \return
 */
int_t* load_codes(const char* codes_path,
                  const int_t n, const int_t m) {
  int_t * codes;
  int_t dim, nx;
  load_data(codes, dim, nx, codes_path);
  if(dim != m || nx != n) {
    throw std::runtime_error("codes size mis-matched!");
  }
  return codes;
}

/**
 * \brief read centroids with shape [m x k x sub_d]
 * \param centroids_path
 * \param m the number of code book
 * \param k the number of centroids in each code book
 * \param d the dimensional of data
 * \return
 */
float* load_centroids(const char* centroids_path,
                      const int m, const int k, const int d) {
  float* centroids;
  int_t dim, nx;
  load_data(centroids,dim, nx, centroids_path);
  int sub_d = d / m + (d % m == 0 ? 0 : 1);
  if (sub_d != dim || nx != m * k) {
    std::cerr << "sub_d :" << sub_d << "\n"
              << "dim   :" << dim << "\n"
              << "nx    :" << nx << "\n"
              << "m     :" << m << "\n"
              << "k     :" << k << "\n";
    throw std::runtime_error("centroids size mis-matched!");
  }
  return centroids;
}


void scale_centroids(float* centroids, const char* scale_file,
                     const int m, const int k, const int d) {
  std::shared_ptr<float > scale(load_scale(scale_file, d));
  int sub_d = 0;
  const int max_d = d / m + (d % m == 0 ? 0 : 1);
  for (int mi = 0; mi < m; ++mi) {
    const int sub_d_mi = d / m + (mi < d % m ? 1 : 0);
    for (int ki = 0; ki < k; ++ki) {
      for (int di = 0; di < sub_d_mi; ++di) {
        centroids[mi * k * max_d + ki * max_d + di] *= scale.get()[sub_d + di];
      }
    }
    sub_d += sub_d_mi;
  }
}


void dimensional_scale(const char* scale_file) {
  std::shared_ptr<float > scale(load_scale(scale_file, dimension));
  auto scale_func = [](float* x, const float* s,
                       int_t nx, int16_t dim) {
    for (int_t i = 0; i < nx; ++i) {
      for (int_t d = 0; d < dim; ++d) {
        *(x++) /= s[d];
      }
    }
  };
  scale_func(xb, scale.get(), nb, dimension);
  scale_func(xq, scale.get(), nq, dimension);
  scale_func(rg, scale.get(), nq, dimension);
}

enum RangeType{
  StandardRange,
  SparseRange
};

bool is_point_in_range_standard (int_t item_id, int_t query_id) {
  return interval_check(xb + item_id * dimension,
                        xq + query_id * dimension,
                        rg + query_id * dimension, dimension);
};
bool is_interval_overlap_standard (const float* center,
                                   const float* width,
                                   int_t sub_d, int_t offset_pq, int_t query_id) {
  return interval_check(center, width,
                        xq + query_id * dimension + offset_pq,
                        rg + query_id * dimension + offset_pq, sub_d);
};
template <int_t p, bool WEIGHTED=true>
float weighted_distance_standard (const float* centroids ,
                                  int_t sub_d, int_t offset_pq, int_t query_id) {
  return weighted_dist<p, WEIGHTED >(xq + query_id * dimension + offset_pq,
                           rg + query_id * dimension + offset_pq,
                           centroids, sub_d);
};


bool is_point_in_range_sparse (int_t item_id, int_t query_id) {
  bool should_in = true;
  const float* xb_ = xb + item_id * dimension;
  const float* xq_ = xq + query_id * dimension;
  const float* rg_ = rg + query_id * dimension;
  for (int di : idx[query_id]) {
    float diff = xb_[di] - xq_[di];
    float range = rg_[di];
    if (diff > range || diff < -range) {
      should_in = false;
      break;
    }
  }
  return should_in;
};
bool is_interval_overlap_sparse (const float* center,
                                 const float* width,
                                 int_t sub_d, int_t offset_pq, int_t query_id) {
  bool should_in = true;
  const float* xb_ = center - offset_pq;
  const float* x_rg_ = width - offset_pq;
  const float* xq_ = xq + query_id * dimension;
  const float* q_rg_ = rg + query_id * dimension;
  for (int di : idx[query_id]) {
    if (di < offset_pq)
      continue;
    if (di >= offset_pq + sub_d)
      break;
    float diff = xb_[di] - xq_[di];
    float range = x_rg_[di] + q_rg_[di];
    if (diff > range || diff < -range) {
      should_in = false;
      break;
    }
  }
  return should_in;
};

template <int_t p, bool WEIGHTED=true>
float weighted_distance_sparse (const float* centroids ,
                                int_t sub_d, int_t offset_pq, int_t query_id) {
  float dist = 0.;
  const float* q = xq + query_id * dimension; 
  const float* w = rg + query_id * dimension;
  const float* x = centroids - offset_pq;

  for (int di : idx[query_id]) {
    if (di < offset_pq)
      continue;
    if (di >= offset_pq + sub_d)
      break;
    float t = (q[di] - x[di]);
    if constexpr (WEIGHTED) {
      t /= (DELTA + w[di]);
    }
    dist+=int_power<p>(t);
  }
  return dist;

};

template<int_t KsIMI, int_t KsPQ, int_t M_PQ, bool IMI, bool PQ, RangeType range_type, bool WEIGHTED>
void search_exactly(const InvertMultiIndex<KsIMI, KsPQ, M_PQ >& index) {
  {
    double recall = 0;
    double precision = 0;
    int_t non_empty = 0;
    long num_probe_items = 0;

    FuncPointRange is_point_in_range;
    FuncIntervalOverlap is_interval_overlap;
    FuncWeightedDistance p_weighted_distance;

    if constexpr (range_type == StandardRange) {
      is_point_in_range = is_point_in_range_standard;
      is_interval_overlap = is_interval_overlap_standard;
      p_weighted_distance = weighted_distance_standard<2, WEIGHTED>;
    }
    if constexpr (range_type == SparseRange) {
      is_point_in_range = is_point_in_range_sparse;
      is_interval_overlap = is_interval_overlap_sparse;
      p_weighted_distance = weighted_distance_sparse<2, WEIGHTED>;
    }

    StopW stop_w;
    for (int i = 0; i < nq; i++) {
      int_t num_probe_i = 0;
      vector<int_t>  ids = index.template exact_search<IMI, PQ>(&num_probe_i, i,
                                                                is_point_in_range,
                                                                is_interval_overlap,
                                                                p_weighted_distance);
      if (!gt[i].empty()) {
        non_empty++;
        recall += 1.0 * ids.size() / gt[i].size();
        precision += 1.0 * ids.size() / num_probe_i;
      }

      num_probe_items += num_probe_i;
    }
    std::cout << "exact \t"
              << "time  : " << stop_w.getElapsedTimeMicro(true) / nq << "\t"
              << "recall: " << recall / non_empty << "\t"
              << "probed: " << num_probe_items / nq << "\n";
  }
}


template<int_t KsIMI, int_t KsPQ, int_t M_PQ, SearchMode Mode, bool PQ, bool PQrank, RangeType range_type, bool WEIGHTED>
void search_approximately(const InvertMultiIndex<KsIMI, KsPQ, M_PQ >& index) {
  vector<float > thresholds ;
  if (Mode == TopK) {
    thresholds = {1, 10, 100, 200, 400, 600, 800,
                  1000, 2000, 4000, 6000, 8000,
                  10000, 20000, 40000, 80000,
                  100000, 200000, 
                  250000, 300000, 350000,
                  400000, 800000};
  } else if (Mode == Precision) {
    thresholds = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-10};
  } else if (Mode == Threshold) {
    thresholds = {1, 1000, 10000, 100000};
  }

  int_t gt_size = 0;
  for (auto& g: gt) {
    gt_size += g.size();
  }

  FuncPointRange is_point_in_range;
  FuncIntervalOverlap is_interval_overlap;
  FuncWeightedDistance p_weighted_distance;

  if constexpr (range_type == StandardRange) {
    is_point_in_range = is_point_in_range_standard;
    is_interval_overlap = is_interval_overlap_standard;
    p_weighted_distance = weighted_distance_standard<2, WEIGHTED>;
  }
  if constexpr (range_type == SparseRange) {
    is_point_in_range = is_point_in_range_sparse;
    is_interval_overlap = is_interval_overlap_sparse;
    p_weighted_distance = weighted_distance_sparse<2, WEIGHTED>;
  }

  std::cout << "threshold_imi" << "\t"
            << "time " << "\t"
            << "average recall "  << "\t"
            << "overall recall" << "\t"
            << "average precision" << "\t"
            << "overall precision" << "\t"
            << "average probe items" << "\n";


  for (auto threshold_imi: thresholds) {
    double recall = 0;
    double precision = 0;
    double return_size = 0;
    int_t non_empty = 0;
    int_t num_probe_items = 0;

    StopW stop_w;
    for (int i = 0; i < nq; i++) {
      int_t num_probe_i = 0;
      vector<int_t>  ids = index.template imi_probe<Mode, PQ, PQrank>(
          threshold_imi, &num_probe_i, i,
          is_point_in_range, is_interval_overlap, p_weighted_distance);

      if (!gt[i].empty()) {
        non_empty++;
        recall += 1.0 * ids.size() / gt[i].size();
        precision += 1.0 * ids.size() / num_probe_i;
      }
      return_size += ids.size();
      num_probe_items += num_probe_i;
    }
    std::cout << threshold_imi << "\t"
              << stop_w.getElapsedTimeMicro(true) / nq << "\t"
              << recall / non_empty << "\t"
              << return_size / gt_size << "\t"
              << precision / non_empty << "\t"
              << return_size / num_probe_items << "\t"
              << num_probe_items / nq << "\n";
    }
}

