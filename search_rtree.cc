//
// main.cpp
//
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>

#include "utils.h"
#include "search_rtree.h"

using namespace std;

int_t nb, nq, dimension;
float *xb, *xq, *rg;
vector<vector<int > > gt;

void search_trees(string fb, string fq, string fr, string fg, int d) {
  load_data(xb, dimension, nb, fb.c_str());
  load_data(xq, dimension, nq, fq.c_str());
  load_data(rg, dimension, nq, fr.c_str());
  gt = load_gt<false >(fg.c_str(), xb, xq, rg, nb, nq, dimension);

  if (d <= 0)
    d = dimension;

  std::cout << "# RD " << d << std::endl;
  std::cout << "# nb " << nb << std::endl;
  std::cout << "# nq " << nq << std::endl;
  std::cout << "# xd " << dimension << std::endl;

  if (d == 2)
    rtree_search<2 >();
  if (d == 6)
    rtree_search<6 >();
  if (d == 8)
    rtree_search<8 >();
  if (d == 12)
    rtree_search<12 >();
  if (d == 16)
    rtree_search<16 >();
  if (d == 32)
    rtree_search<32 >();
  if (d == 64)
    rtree_search<64 >();
  if (d == 91)
    rtree_search<91 >();
  if (d == 128)
    rtree_search<128 >();
  if (d == 200)
    rtree_search<200 >();
  if (d == 256)
    rtree_search<256 >();
  if (d == 512)
    rtree_search<512 >();
  if (d == 960)
    rtree_search<960 >();

  delete [] rg;
  delete [] xb;
  delete [] xq;
}

int main(int argc, const char** argv) {
  if (argc <= 4) {
    std:cerr << "./kdt fb fq fr fg [d]\n";
  }
  string fb = argv[1];
  string fq = argv[2];
  string fr = argv[3];
  string fg = argv[4];
  int d = -1;
  if (argc > 5)
    d = std::atoi(argv[5]);

  search_trees(fb, fq, fr, fg, d);

  return 0;
}
