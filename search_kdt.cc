//
// main.cpp
//
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>

#include "utils.h"
#include "search_kdt.h"

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

  search_kd_tree(d);

  delete [] xb;
  delete [] xq;
  delete [] rg;
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
