//
// main.cpp
//
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "utils.h"

using namespace std;
using std::to_string;

int_t nb, nq, dimension;
float *xb, *xq, *rg;
vector<vector<int > > gt;


void search_bf(string data_dir, string dataset, string seed, string rg_size, string rg_type) {
  string fb = data_dir + dataset + "/" + dataset + "_base.fvecs";
  string fq = data_dir + dataset + "/" + dataset + "_query.fvecs";

  std::cout << "# loading data from" << fb << "\n";
  load_data(xb, dimension, nb, fb.c_str());
  std::cout << "# loading data from" << fq << "\n";
  load_data(xq, dimension, nq, fq.c_str());

#ifdef verbose
  std::cout << "# nb " << nb << std::endl;
  std::cout << "# nq " << nq << std::endl;
  std::cout << "# xd " << dimension << std::endl;
#endif
  std::cout << "# dataset " << dataset <<std::endl;


          std::cout << "# rg size " << rg_size << "\n";
          std::cout << "# rg type " << rg_type << "\n";
          string fr = data_dir + dataset + "/" + dataset + "_s" + seed +
                           "_sz" + rg_size + "_" +  rg_type + "_rg.fvecs";
          string fg = data_dir + dataset + "/" + dataset + "_s" + seed +
                           "_sz" + rg_size + "_" +  rg_type + "_gt.txt";

          load_data(rg, dimension, nq, fr.c_str());
        
          StopW stop_w;
          gt = load_gt<true>(fg.c_str(), xb, xq, rg, nb, nq, dimension);
          std::cout << "# brute force time : "
                    << stop_w.getElapsedTimeMicro(true) / nq << "\n";
       
         delete [] rg;

  delete [] xb;
  delete [] xq;
}

int main(int argc, const char** argv) {
  const char* dataset = argv[1];
  const char* seed = argv[2];
  const char* rg_size = argv[3];
  const char* rg_type = argv[4];
  string data_dir = "/data/xydai/data/";

  search_bf(data_dir, dataset, seed, rg_size, rg_type);
  return 0;
}
