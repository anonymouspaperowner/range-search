//
// main.cpp
//
#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <memory>

#include "utils.h"
#include "search_lsh.h"

using namespace std;

int_t nb, nq, dimension;
float *xb, *xq, *rg;
vector<vector<int > > idx;
vector<vector<int > > gt;

template <size_t DIM_, size_t IDX_>
void split_range(vector<vector<float > >* res, const vector<int_t>& idx, float* x, const float* rg) {
  if constexpr (DIM_ == IDX_) {
    res->emplace_back(x, x + dimension);
  } else {
    auto aid = idx[IDX_];
    auto bid = idx[DIM_];
    int num_split =  floor(rg[aid] / rg[bid] + 0.6); // rounding
    if (num_split < 1) {
      throw std::runtime_error("wrong split number < 1!!!");
    }
    float half_width = rg[aid] / num_split;
    float center = x[aid];
    for (int i = 0 ; i < num_split; i++) {
      x[aid] = center - rg[aid] + (2 * i + 1) * half_width;
      split_range<DIM_, IDX_ + 1>(res, idx, x, rg);
    }
    x[aid] = center;
  }
}

template <size_t DIM_SPLIT>
void lsh_query(LSH<E2LSH >& lsh, int l) {
  int_t gt_size = 0;
  for (auto& g: gt) {
    gt_size += g.size();
  }

  double recall = 0;
  double precision = 0;
  double return_size = 0;
  double num_probe_items = 0;
  int_t non_empty = 0;

  StopW stop_w;
  for (size_t i = 0 ; i < nq ; ++i) {
    std::set<int > probed;
    if constexpr (DIM_SPLIT == -1) {
      probed = lsh.range_search(xq + i * dimension, rg + i * dimension, l);
    } else if constexpr (DIM_SPLIT == 0){
      probed = lsh.search(xq + i * dimension, l);
    } else {
      vector<int_t> idx = arg_sort(rg + i * dimension, dimension);
      std::reverse(idx.begin(), idx.end());

      vector<vector<float > > split_xs;
     // split_xs.emplace_back(xq + i * dimension, xq + (i + 1) * dimension);
      split_range<DIM_SPLIT, 0>(&split_xs, idx, xq + i * dimension, rg + i * dimension);

      for (const vector<float >& split_x : split_xs) {
        auto s = lsh.search(split_x.data(), l);
        probed.insert(s.begin(), s.end());
      }
    }

    auto res = interval_prune(xb, &xq[i * dimension],
                              &rg[i * dimension],
                              probed, dimension);

    if (!gt[i].empty()) {
      non_empty++;
      recall += 1.0 * res.size() / gt[i].size();
      precision += 1.0 * res.size() / probed.size();
    }
    return_size += res.size();
    num_probe_items += probed.size();
  }

  std::cout << stop_w.getElapsedTimeMicro(true) / nq << " \t";
  std::cout << num_probe_items / nq << " \t";
  std::cout << recall / non_empty << " \t";
  std::cout << return_size / gt_size << " \t";
  std::cout << precision / non_empty << " \t";
  std::cout << return_size / num_probe_items << " \n";
}

template <size_t DIM_SPLIT>
void search_lshs(string data_dir, string dataset, string seed,
                 string rg_size, string rg_type, int l, int k, float r, int dense_dim) {
  string fb = data_dir + dataset + "/" + dataset + "_base.fvecs";
  string fq = data_dir + dataset + "/" + dataset + "_query.fvecs";

  load_data(xb, dimension, nb, fb.c_str());
  load_data(xq, dimension, nq, fq.c_str());

  std::cout << "# nb " << nb << std::endl;
  std::cout << "# nq " << nq << std::endl;
  std::cout << "# xd " << dimension << std::endl;
  std::cout << "# dataset " << dataset <<std::endl;
  std::cout << "# L " << l << std::endl;
  std::cout << "# K " << k << std::endl;
  std::cout << "# R " << r << std::endl;

  /*
  vector<float > norms(nb);
  for (int i = 0 ; i < nb; ++i) {
    float norm_sqr = std::inner_product(xb + i * dimension,
                                        xb + (i + 1) * dimension,
                                        xb + i * dimension, 0.0);
    norms[i] = std::sqrt(norm_sqr);
  }
  std::sort(norms.begin(), norms.end());
  const float median = norms[nb / 2];
*/

  LSH<E2LSH > lsh(l, k, dimension, r);
  // Insert values
  StopW stop_w;
  std::cout << "# add" << std::endl;
  float avg_size = lsh.add(xb, nb);
  std::cout << "# bucket " << avg_size << std::endl;
  std::cout << "# memory : "
            << 1.0 * getCurrentRSS() / 1000000000
            << "Gb" << std::endl;
  std::cout << "# average insert time : "
            << stop_w.getElapsedTimeMicro(true) / nb
            << std::endl;

  string sparse_or_dense = "";
  if (dense_dim > 0) {
    sparse_or_dense += "_sparse";
    sparse_or_dense += std::to_string(dense_dim);
  }

  std::cout << "# rg size " << rg_size << "\n";
  std::cout << "# rg type " << rg_type << "\n";
  string fr = data_dir + dataset + "/" + dataset + "_s" + seed +
                   "_sz" + rg_size + "_" +  rg_type + sparse_or_dense + "_rg.fvecs";
  string fg = data_dir + dataset + "/" + dataset + "_s" + seed +
                   "_sz" + rg_size + "_" +  rg_type + sparse_or_dense + "_gt.txt";
  string fi = data_dir + dataset + "/" + dataset + "_s" + seed +
      "_sz" + rg_size + "_" +  rg_type + sparse_or_dense + "_idx.txt";
  load_data(rg, dimension, nq, fr.c_str());

  gt = load_gt<false>(fg.c_str(), xb, xq, rg, nb, nq, dimension);

  if (dense_dim > 0) {
    idx = read_list(fi.c_str());
    vector<float> max_ele(xb, xb + dimension);
    vector<float> min_ele(xb, xb + dimension);
    for (int i = 1; i < nb; ++i) {
      for (int di = 0; di < dimension; ++di) {
        float x = xb[i *  dimension + di];
        if (x > max_ele[di]) {
          max_ele[di] = x;
        }
        if (x < min_ele[di]) {
          min_ele[di] = x;
        }
      }
    }
    vector<float> center_ele(dimension, 0);
    vector<float> width_ele(dimension, 0);
    for (int di = 0; di < dimension; ++di) {
      center_ele[di] = (max_ele[di] + min_ele[di]) / 2.0;
      width_ele[di] = (max_ele[di] - min_ele[di]) / 2.0;
    }

    vector<float> xq_cp(xq, xq + nq * dimension);
    vector<float> rg_cp(rg, rg + nq * dimension);
    for (int i = 0; i < nq; ++i) {
      std::memcpy(xq + i * dimension, center_ele.data(), dimension * sizeof(float));
      std::memcpy(rg + i * dimension, width_ele.data(), dimension * sizeof(float));
    }
    for (int i = 0; i < nq; ++i) {
      for (int di : idx[i]) {
        auto shift = i * dimension + di;
        xq[shift] = xq_cp[shift];
        rg[shift] = rg_cp[shift];
      }
    }
  }

  std::cout << "time \t"
            << "item \t"
            << "avg recall \t"
            << "overall recall \t"
            << "avg precision \t"
            << "overall precision \n";


  for (int l_ = 1; l_ <= l; l_ *= 2) {
    lsh_query<DIM_SPLIT >(lsh,  l_);
  }

  std::cout << "\n\n\n\n";

  delete [] rg;
  delete [] xb;
  delete [] xq;
}

int main(int argc, const char** argv) {
  if (argc < 8) {
      std::cerr << " ./runable datset seed rg_size rg_type k l r [dim]\n";
      exit(1);
  }

  const char* dataset = argv[1];
  const char* seed = argv[2];
  const char* rg_size = argv[3];
  const char* rg_type = argv[4];
  int k = std::atoi(argv[5]);
  int l = std::atoi(argv[6]);
  float r = std::atof(argv[7]);
  int dense_dim = -1;
  if (argc > 8)
    dense_dim = std::atoi(argv[8]);
  string data_dir = "/data/xydai/data/";

  const size_t DIM_SPLIT = 1;
  std::cout << "spliting dim " << DIM_SPLIT << "\N";
  search_lshs<DIM_SPLIT >(data_dir, dataset, seed, rg_size, rg_type, l, k, r, dense_dim);
  return 0;
}
