//
// main.cpp
//
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "utils.h"
#include "search_vq.h"

using namespace std;
using std::to_string;

int_t nb, nq, dimension;
float *xb, *xq, *rg;
vector<vector<int > > idx;
vector<vector<int > > gt;

typedef struct {
  string data_dir;
  string dataset;
  string trained_on;
  string seed;
  string rg_size;
  string rg_type;
  int Ks;
  int M;
  int dense_dim;
  int weighted;
} vq_parameters;

template<int_t KsIMI, int_t KsPQ, int_t M_PQ, RangeType range_type, bool WEIGHTED>
void search_vqs(const vq_parameters& para) {
  int_t dense_dim = para.dense_dim;
  string data_dir = para.data_dir;
  string dataset = para.dataset;
  string seed = para.seed;
  string rg_size = para.rg_size;
  string rg_type = para.rg_type;

  string fb = data_dir + dataset + "/" + dataset + "_base.fvecs";
  string fq = data_dir + dataset + "/" + dataset + "_query.fvecs";


  string imi_centroids_file = data_dir + dataset + "/" + dataset +
                               "_s808_pq2_ks" + to_string(KsIMI) + "_centroids.fvecs";
  string imi_codes_file = data_dir + dataset + "/" + dataset +
                           "_s808_pq2_ks" + to_string(KsIMI) + "_codes.fvecs";
  string pq_centroids_file = data_dir + dataset + "/" + dataset +
                              "_s808_pq" + to_string(M_PQ) +
                              "_ks" + to_string(KsPQ) + "_centroids.fvecs";
  string pq_codes_file = data_dir + dataset + "/" + dataset +
                          "_s808_pq" + to_string(M_PQ) +
                          "_ks" + to_string(KsPQ) + "_codes.fvecs";
  string scale_file =  data_dir + dataset + "/" + dataset + "_s808_p75_scale.fvecs";

  if (para.trained_on != dataset) {
    imi_centroids_file = data_dir + para.trained_on + "/" + para.trained_on +
        "_s808_pq2_ks" + to_string(KsIMI) + "_centroids.fvecs";
    imi_codes_file = data_dir + dataset + "/" + dataset + "_basedon_" + para.trained_on +
        "_s808_pq2_ks" + to_string(KsIMI) + "_codes.fvecs";
    pq_centroids_file = data_dir + para.trained_on + "/" + para.trained_on +
        "_s808_pq" + to_string(M_PQ) +
        "_ks" + to_string(KsPQ) + "_centroids.fvecs";
    pq_codes_file = data_dir + dataset + "/" + dataset + "_basedon_" + para.trained_on +
        "_s808_pq" + to_string(M_PQ) +
        "_ks" + to_string(KsPQ) + "_codes.fvecs";
    scale_file =  data_dir + para.trained_on + "/" + para.trained_on + "_s808_p75_scale.fvecs";
  }

  std::cout << "# loading data from" << fb << "\n";
  load_data(xb, dimension, nb, fb.c_str());
  std::cout << "# loading data from" << fq << "\n";
  load_data(xq, dimension, nq, fq.c_str());


  std::cout << "# nb " << nb << std::endl;
  std::cout << "# nq " << nq << std::endl;
  std::cout << "# xd " << dimension << std::endl;

  std::cout << "# dataset " << dataset <<std::endl;
  std::cout << "# KsIMI " << KsIMI <<std::endl;
  std::cout << "# KsPQ " << KsPQ <<std::endl;
  std::cout << "# M_PQ " << M_PQ <<std::endl;

  std::shared_ptr<int_t> pq_codes(load_codes(pq_codes_file.c_str(), nb, M_PQ));
  std::shared_ptr<float> pq_centroids(load_centroids(pq_centroids_file.c_str(), M_PQ, KsPQ, dimension));
  scale_centroids(pq_centroids.get(), scale_file.c_str(), M_PQ, KsPQ, dimension);
  auto pq_ptr = new ProductQuantization<KsPQ, M_PQ >(dimension, nb, pq_codes.get(), pq_centroids.get(), xb);
  shared_ptr<ProductQuantization<KsPQ, M_PQ > > pq_shared_ptr(pq_ptr);

  std::shared_ptr<int_t> codes(load_codes(imi_codes_file.c_str(), nb, M_IMI));
  std::shared_ptr<float> centroids(load_centroids(imi_centroids_file.c_str(), M_IMI, KsIMI, dimension));
  scale_centroids(centroids.get(), scale_file.c_str(), M_IMI, KsIMI, dimension);
  // new index from heap since stack size is limited
  auto index_ptr = new InvertMultiIndex<KsIMI, KsPQ, M_PQ >(dimension, nb, codes.get(), centroids.get(), xb);
  index_ptr->set_pq(pq_ptr);
  shared_ptr<InvertMultiIndex<KsIMI, KsPQ, M_PQ > > index_shared_ptr(index_ptr);
  InvertMultiIndex<KsIMI, KsPQ, M_PQ >& index = *index_ptr;

  index.stat();

  std::cout << "# rg size " << rg_size << "\n";
  std::cout << "# rg type " << rg_type << "\n";
  string sparse_or_dense = "";
  if (range_type == SparseRange) {
    sparse_or_dense += "_sparse";
    sparse_or_dense += std::to_string(dense_dim);
  }
  string fr = data_dir + dataset + "/" + dataset + "_s" + seed +
              "_sz" + rg_size + "_" +  rg_type + sparse_or_dense + "_rg.fvecs";
  string fg = data_dir + dataset + "/" + dataset + "_s" + seed +
              "_sz" + rg_size + "_" +  rg_type + sparse_or_dense + "_gt.txt";
  string fi = data_dir + dataset + "/" + dataset + "_s" + seed +
              "_sz" + rg_size + "_" +  rg_type + sparse_or_dense + "_idx.txt";
  if (range_type == SparseRange) {
    idx = read_list(fi.c_str());
  }
  load_data(rg, dimension, nq, fr.c_str());
  gt = load_gt(fg.c_str(), xb, xq, rg, nb, nq, dimension);

//  std::cout << "# PQx only\n";
//  search_exactly<KsIMI, KsPQ, M_PQ, false, true, range_type, WEIGHTED>(index);
  std::cout << "# IMI only\n";
  search_approximately<KsIMI, KsPQ, M_PQ, TopK, false, false, range_type, WEIGHTED>(index);
  std::cout << "# IMI + PQ\n";
  search_approximately<KsIMI, KsPQ, M_PQ, TopK, true, false, range_type, WEIGHTED>(index);

  delete [] rg;
  delete [] xb;
  delete [] xq;
}

template<int_t KsIMI, int_t KsPQ, int_t M_PQ, RangeType range_type>
void search_vqs_4t(const vq_parameters& para) {
  if (para.weighted > 0)
    search_vqs<KsIMI, KsPQ, M_PQ, range_type, true>(para);
  else
    search_vqs<KsIMI, KsPQ, M_PQ, range_type, false>(para);
}

template<int_t KsIMI, int_t KsPQ, int_t M_PQ>
void search_vqs_3t(const vq_parameters& para) {
  if (para.dense_dim > 0)
    search_vqs_4t<KsIMI, KsPQ, M_PQ, SparseRange>(para);
  else
    search_vqs_4t<KsIMI, KsPQ, M_PQ, StandardRange>(para);
}

template<int_t KsIMI, int_t KsPQ>
void search_vqs_2t(const vq_parameters& para) {
  int_t M_PQ = para.M;
  if (M_PQ == 2)
    search_vqs_3t<KsIMI, KsPQ, 2>(para);
  else if (M_PQ == 4)
    search_vqs_3t<KsIMI, KsPQ, 4>(para);
  else if (M_PQ == 8)
    search_vqs_3t<KsIMI, KsPQ, 8>(para);
  else if (M_PQ == 16)
    search_vqs_3t<KsIMI, KsPQ, 16>(para);
  else if (M_PQ == 20)
    search_vqs_3t<KsIMI, KsPQ, 20>( para);
  else if (M_PQ == 25)
    search_vqs_3t<KsIMI, KsPQ, 25>( para);
  else if (M_PQ == 32)
    search_vqs_3t<KsIMI, KsPQ, 32>( para);
  else if (M_PQ == 48)
    search_vqs_3t<KsIMI, KsPQ, 48>( para);
  else if (M_PQ == 50)
    search_vqs_3t<KsIMI, KsPQ, 50>( para);
  else if (M_PQ == 64)
    search_vqs_3t<KsIMI, KsPQ, 64>( para);
  else if (M_PQ == 120)
    search_vqs_3t<KsIMI, KsPQ, 120>( para);
  else if (M_PQ == 128)
    search_vqs_3t<KsIMI, KsPQ, 128>( para);
  else if (M_PQ == 240)
    search_vqs_3t<KsIMI, KsPQ, 240>( para);
  else if (M_PQ == 256)
    search_vqs_3t<KsIMI, KsPQ, 256>( para);
  else
    throw std::runtime_error("no matched M");
}


void search_vqs_0t(const vq_parameters& para) {
  int_t KsIMI = para.Ks;
  if (KsIMI == 256)
    search_vqs_2t<256, 256 >(para);
  else if (KsIMI == 512)
    search_vqs_2t<512, 512 >(para);
  else if (KsIMI == 1024)
    search_vqs_2t<1024, 1024 >(para);
  else
    throw std::runtime_error("no matched KsIMI");
}



int main(int argc, const char** argv) {
  if (argc <= 8) {
    std::cerr << "./runable dataset seed rg_size rg_type K M weighted dense_dim\n";
    exit(0);
  }
  vq_parameters para;
  para.dataset = argv[1];
  para.seed = argv[2];
  para.rg_size = argv[3];
  para.rg_type = argv[4];
  para.Ks = std::atoi(argv[5]);
  para.M = std::atoi(argv[6]);
  para.weighted = std::atoi(argv[7]);
  para.dense_dim = std::atoi(argv[8]);
  para.data_dir = "/data/xydai/data/";

  para.trained_on = para.dataset;
  if (argc > 9) {
    para.trained_on = argv[9];
  }

  search_vqs_0t(para);

  return 0;
}
