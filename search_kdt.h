//
// Created by xydai on 2020/7/9.
//

#ifndef SS_INTERVAL__SEARCH_KDT_H_
#define SS_INTERVAL__SEARCH_KDT_H_
#include "kdt/kdtree.h"
#include "utils.h"
extern int_t nb;
extern int_t nq;
extern int_t dimension;
extern float *xb;
extern float *xq;
extern float *rg;
extern vector<vector<int > > gt;


template<size_t D>
class Point {
 private:
  std::array<T, D> ptr_;
 public:
  explicit Point(const std::array<T, D>& p): ptr_(p) {}
  explicit Point(const T* p) {
#pragma unroll
    for (int i = 0; i < D; ++i) {
      ptr_[i] = p[i];
    }
  }

  constexpr static int dimension() {
    return D;
  }

  T operator() (int dim) const {
    return ptr_[dim];
  }
  const T* ptr() const {
    return ptr_.data();
  }
};

template<std::size_t D>
std::ostream& operator <<(std::ostream& outs, const Point<D >& p) {
  outs << p(0);
  for (int i = 1; i< p.dimension(); i++) {
    outs << ' ' << p(i);
  }
  return outs;
}


template<size_t D>
void _search_kd_tree() {
  std::cout << "# building..." << std::endl;
  std::cout << "# D " << D << std::endl;
  KDTree<Point<D > > tree;

  std::vector<Point<D > > points;
  points.reserve(nb);
  for (int i = 0; i < nb; i++) {
    points.emplace_back(xb + i * dimension);
  }

  tree.build_tree(points);

  std::cout << "# searching..." << std::endl;
  double ret_size = 0;
  double probe_time = 0;
  double prune_time = 0;
  StopW stop_w;
  for (int i = 0; i < nq; ++i) {
    Point<D> q(xq + i * dimension);
    Point<D> r(rg + i * dimension);
    auto ret = tree.range_search(q, r);
    ret_size += ret.size();
    probe_time += stop_w.getElapsedTimeMicro(true);

    interval_prune(xb, &xq[i * dimension], &rg[i * dimension], ret, dimension);
    prune_time += stop_w.getElapsedTimeMicro(true);
  }

  std::cout << "average probe time : "
            << probe_time / nq
            << std::endl;
  std::cout << "average prune time : "
            << prune_time / nq
            << std::endl;
  std::cout << "average total time : "
            << (probe_time + prune_time) / nq
            << std::endl;
  std::cout << "average number of probed item : "
            << ret_size / nq
            << std::endl;
}


void search_kd_tree(int d) {
  if (d==2) {
    _search_kd_tree<2>();
  }
  else if (d==3) {
    _search_kd_tree<3>();
  }
  else if (d==4) {
    _search_kd_tree<4>();
  }
  else if (d==5) {
    _search_kd_tree<5>();
  }
  else if (d==6) {
    _search_kd_tree<6>();
  }
  else if (d==8) {
    _search_kd_tree<8>();
  }
  else if (d==10) {
    _search_kd_tree<10>();
  }
  else if (d==12) {
    _search_kd_tree<12>();
  }
  else if (d==16) {
    _search_kd_tree<16>();
  }
  else if (d==24) {
    _search_kd_tree<24>();
  }
  else if (d==32) {
    _search_kd_tree<32>();
  }
  else if (d==64) {
    _search_kd_tree<64>();
  }
  else if (d==91) {
    _search_kd_tree<91>();
  }
  else if (d==128) {
    _search_kd_tree<128>();
  }
  else if (d==200) {
    _search_kd_tree<200>();
  }
  else if (d==256) {
    _search_kd_tree<256>();
  }
  else if (d==512) {
    _search_kd_tree<512>();
  }
  else if (d==960) {
    _search_kd_tree<960>();
  }
  else {
    throw std::runtime_error("not supported dimension");
  }
}
#endif //SS_INTERVAL__SEARCH_KDT_H_
