#include "kdtree.h"




template<size_t D>
class Point {
 private:
  std::array<T, D> ptr_;
 public:
  explicit Point(const std::array<T, D>& p): ptr_(p) {}
  explicit Point(const T p[D]) {
#pragma unroll
    for (int i = 0; i < D; ++i) {
      ptr_[i] = p[i];
    }
  }

  Point& operator=(const Point<D>& c) {
    ptr_ = c.ptr_;
    return *this;
  }

  constexpr static int dimension() { return D; }
  float operator() (int dim) const{
    return ptr_[dim];
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

int main(int argc, char** argv) {
  constexpr size_t D = 3;
  KDTree<Point<D > > tree;
  std::vector<std::array<T, D> > pts_{
    {0,0,0},
    {1,1,1},
    {-1, 3, 4},
    {5, 6, 7},
    {2, -6, 8},
    {4, 5, -4},
    {2, 3, 4},
    {2, 5, 6}
    };
  std::vector<Point<D > > pts;
  for (auto& pt : pts_) {
    pts.emplace_back(pt);
  }
  tree.build_tree(pts);
//  tree.dump_tree_inorder();
  
  std::cout << "searching near 0,0,0.1" << std::endl;
  Point<D> q({0, 0, 0.1});
  auto closeNodes = tree.get_points_within_cube({q}, 0.2);
  std::cout << "found" << std::endl;
  for(auto n : closeNodes) {
    tree.dump_node(n);
  }

  std::cout << "searching near 0.5,0.5,0.5" << std::endl;
  Point<D> rg({0.5, 0.5, 0.5});
  closeNodes = tree.get_points_within_cube(rg, 0.7);
  std::cout << "found" << std::endl;
  for(auto n : closeNodes) {
    tree.dump_node(n);
  }

}
