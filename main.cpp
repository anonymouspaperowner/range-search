//
// main.cpp
//
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "utils.h"
#include "search_vq.h"
#include "search_lsh.h"
#include "search_kdt.h"
#include "search_rtree.h"

using namespace std;

int_t nb, nq, dimension;
float *xb, *xq, *rg;
vector<vector<int > > gt;



int main(int argc, const char** argv)
{
  const char* fb = "/home/xydai/program/data/sift-128/"
                   "sift-128_base.fvecs";
  const char* fq = "/home/xydai/program/data/sift-128/"
                   "sift-128_query.fvecs";
  const char* fr = "/home/xydai/program/data/sift-128/"
                   "sift-128_s808_p50_zipf_rg.fvecs";
  const char* fg = "/home/xydai/program/data/sift-128/"
                   "sift-128_s808_p50_zipf_gt.txt";
  if (argc > 3) {
    fb = argv[1];
    fq = argv[2];
    fr = argv[3];
    fg = argv[4];
  }

  load_data(xb, dimension, nb, fb);
  load_data(xq, dimension, nq, fq);
  load_data(rg, dimension, nq, fr);
#ifdef verbose
  std::cout << "# Using DIM = " << DIM << std::endl;
  std::cout << "# nb " << nb << std::endl;
  std::cout << "# nq " << nq << std::endl;
  std::cout << "# xd " << dimension << std::endl;
#endif

  StopW stop_w;
  gt = load_gt(fg, xb, xq, rg, nb, nq, dimension);
  std::cout << "# average time " << stop_w.getElapsedTimeMicro(true) / nq << std::endl;
#ifdef verbose
  int count = 0;
  for (const auto& g : gt) {
    count += g.size();
  }
  std::cout << "# average size " << 1. * count / nq << std::endl;
#endif

  const char* imi_centroids_file = "/home/xydai/program/data/sift-128/"
                               "sift-128_s808_pq2_ks256_centroids.fvecs";
  const char* imi_codes_file = "/home/xydai/program/data/sift-128/"
                           "sift-128_s808_pq2_ks256_codes.fvecs";

  const char* pq_centroids_file = "/home/xydai/program/data/sift-128/"
                               "sift-128_s808_pq16_ks256_centroids.fvecs";
  const char* pq_codes_file = "/home/xydai/program/data/sift-128/"
                           "sift-128_s808_pq16_ks256_codes.fvecs";

  const char* scale_file = "/home/xydai/program/data/sift-128/"
                           "sift-128_s808_p75_scale.fvecs";
  if (argc > 7) {
    scale_file = argv[5];
    imi_centroids_file = argv[6];
    imi_codes_file = argv[7];
  }

  if (argc > 9) {
    pq_centroids_file = argv[8];
    pq_codes_file = argv[9];
  }

//  lsh_search();
//  search_kd_tree<91>();

  imi_flat_search<256, 256, 16, TopK>(imi_codes_file, imi_centroids_file, pq_codes_file, pq_centroids_file, scale_file);
//  imi_flat_search<256, Threshold>(codes_file, centroids_file, scale_file);
//  imi_flat_search<256, Precision>(codes_file, centroids_file, scale_file);

//  rtree_search();

  delete [] xb;
  delete [] xq;
  delete [] rg;
  return 0;
}
