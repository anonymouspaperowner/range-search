#include <boost/geometry/index/rtree.hpp>
#include "utils.h"

using namespace std;

extern int_t nb;
extern int_t nq;
extern int_t dimension;
extern float *xb;
extern float *xq;
extern float *rg;
extern vector<vector<int > > gt;

/*
#ifndef DIM
#define DIM 2
#endif

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::point<ValueType , DIM, bg::cs::cartesian> point_t;
typedef bg::model::box<point_t> box_t;
typedef std::pair<box_t, uint64_t> value_t;
*/

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
template <size_t RD >
using point_t = bg::model::point<ValueType, RD, bg::cs::cartesian>;
template <size_t RD >
using box_t = bg::model::box<point_t<RD > >;
template <size_t RD >
using value_t = std::pair<box_t<RD >, uint64_t>;

template <size_t N, size_t I>
static void assign_point(point_t<N >& p, const float * x) {
//  p.set<I>(x[I]);
  bg::set<I>(p, x[I]);
  if constexpr (I + 1 < N) {
    assign_point<N, I+1 >(p, x);
  }
}

template <size_t RD>
void rtree_search() {
  bgi::rtree<value_t<RD >, bgi::quadratic<8, 4> > bg_tree;
  
  // Insert values
  StopW stop_w;
  for(size_t i = 0; i < nb; i++)
  {
    auto p = point_t<RD >();
    assign_point<RD, 0>(p, &xb[i * dimension]);

    box_t<RD > b(p, p);
    bg_tree.insert(value_t<RD >(b, i));
  }
  std::cout << "average insert time : "
            << stop_w.getElapsedTimeMicro(true) / nb << std::endl;
  // test BG Rtree
  float *xq_min = new float[dimension * nq];
  float *xq_max = new float[dimension * nq];

  for (int_t i = 0; i < nq; ++i) {
    for (int_t dim = 0; dim < dimension; ++dim) {
      int_t pos = i * dimension + dim;
      xq_min[pos] = xq[pos] - rg[pos];
      xq_max[pos] = xq[pos] + rg[pos];
    }
  }
  {
    vector<float > n_hits;
    vector<vector<int > > res(nq);
    
    long probe_time = 0;
    long prune_time = 0;

#ifdef verbose
    ProgressBar search_progress(nq, "searching");
#endif

    for (size_t i = 0 ; i < nq ; ++i)
    {
      stop_w.reset();
      auto p_min = point_t<RD >();
      auto p_max = point_t<RD >();
      assign_point<RD, 0>(p_min, &xq_min[i * dimension]);
      assign_point<RD, 0>(p_max, &xq_max[i * dimension]);
      box_t<RD > search_box(p_min, p_max);
      std::vector<value_t<RD > > re;
      bg_tree.query(bgi::intersects(search_box), std::back_inserter(re));
      probe_time += stop_w.getElapsedTimeMicro(false);

      vector<int_t > ids;
      ids.reserve(re.size());
      for (auto p: re) {
        ids.push_back(p.second);
      }
      n_hits.push_back(re.size());

      stop_w.reset();
      res[i] = interval_prune(xb, &xq[i * dimension], &rg[i * dimension], ids, dimension);
      prune_time += stop_w.getElapsedTimeMicro(false);

#ifdef verbose
      ++search_progress;
#endif
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
    double sum = std::accumulate(n_hits.begin(), n_hits.end(), 0.0);
    std::cout << "average number of probed item : "
              << sum / nq 
              << std::endl;
  }
  delete[] xq_min;
  delete[] xq_max;
}
