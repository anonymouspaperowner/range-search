#ifndef _BJ_KD_TREE_H
#define _BJ_KD_TREE_H

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <iostream>

using std::vector;
using std::unique_ptr;

using T = float;


template <typename PointType, int SplitDimension>
class KDNode {
 private:
  constexpr static size_t NextDimension() {
    return (SplitDimension +1) % PointType::dimension();
  }
 public:
  explicit KDNode(size_t ind) : tree_index(ind) {}

  size_t tree_index; //  particle's position in tree's list
  unique_ptr<KDNode<PointType, NextDimension() > > left_child, right_child;
};


template <typename PointType, typename PointArray=vector<PointType > >
class KDTree {
 public:
  KDTree() = default;
  explicit KDTree(const PointArray& pointsIn) {
    build_tree(pointsIn);
  }

  void build_tree(const PointArray& points_in);
  void dump_tree_inorder();

  template <typename F>
  void inorder_traversal(F func);

  vector<size_t> get_points_within_cube(PointType test_point, T radius);
  vector<size_t> range_search(PointType query, PointType range);

  size_t find_min(int dimension);

  void insert_point(const PointType& p);
  void delete_point(size_t node_index);
  PointType get_point(size_t node_index) {
    return points_[node_index];
  }
  void dump_node(size_t i) {
    std::cout << points_[i] << std::endl;
  }

  //  END PUBLIC API

 private:
  unique_ptr<KDNode<PointType, 0> > root_;
  PointArray points_;
  vector<size_t> point_indices_;

  template <int SplitDimension>
  unique_ptr<KDNode<PointType, SplitDimension> >
  build_subtree( vector<size_t>::iterator begin, vector<size_t>::iterator end);




  template<int SplitDimension>
  void dump_subtree(unique_ptr<KDNode<PointType, SplitDimension> >& node);

  template<int SplitDimension>
  void get_points_within_cube_subtree(T query_range[2 * PointType::dimension()],
                                      unique_ptr<KDNode<PointType, SplitDimension>>& node,
                                      vector<size_t>& ret);

  static bool point_in_range(PointType test_point, const T *query_range) {
    for(int i = 0; i < PointType::dimension(); ++i) {
      if (test_point(i) < query_range[2 * i] ||
          test_point(i) > query_range[2 * i + 1]) {
        return false;
      }
    }
    return true;
  }

  static bool interval_check(PointType xb_, PointType xq_, PointType rg_) {
    bool should_in = true;
    for (int di = 0; di < PointType::dimension(); ++di) {
      T diff = xb_(di) - xq_(di);
      T range = rg_(di);
      if (diff > range || diff < -range) {
        should_in = false;
        break;
      }
    }
    return should_in;
  }

  template<int SplitDimension>
  size_t find_min_subtree(int dimension,
                          unique_ptr<KDNode<PointType, SplitDimension> >& node);

  template<int SplitDimension>
  unique_ptr<KDNode<PointType, SplitDimension> >
  delete_from_subtree(size_t nodeIndex,
                      unique_ptr<KDNode<PointType, SplitDimension> >& node);

  template<int SplitDimension, typename F>
  void inorder_traversal_subtree(F func,
                                 unique_ptr<KDNode<PointType, SplitDimension> >& node);


  template<int SplitDimension>
  unique_ptr<KDNode<PointType, SplitDimension> >
  insert_point_subtree(unique_ptr<KDNode<PointType, SplitDimension > >& node,
                       size_t point_index);
};


template<typename PointType, typename PointArray>
void KDTree<PointType, PointArray>::build_tree(const PointArray& points_in) {
  points_ = points_in;
  point_indices_.resize(points_.size());
  std::iota(begin(point_indices_), end(point_indices_), 0);
  root_ = build_subtree<0>(begin(point_indices_), end(point_indices_));
}

template<typename PointType, typename PointArray>
template<int SplitDimension>
unique_ptr<KDNode<PointType, SplitDimension> >
KDTree<PointType, PointArray>::build_subtree(vector<size_t>::iterator begin,
                                             vector<size_t>::iterator end) {
  constexpr size_t NextDimension = (SplitDimension + 1) % PointType::dimension();
  auto range_size = std::distance(begin, end);

  if (range_size == 0) {
    return unique_ptr<KDNode<PointType, SplitDimension> >(nullptr);
  } else {
    std::sort(begin, end,
              [this](size_t a, size_t b) {
                return points_[a](SplitDimension) <
                       points_[b](SplitDimension);
              });
    auto median = begin + range_size / 2;
    while(median != begin  &&
          points_[*(median)](SplitDimension) ==
          points_[*(median - 1)](SplitDimension)) {
      --median; // put all the nodes with equal coord value in the right subtree
    }
    auto ret = unique_ptr<KDNode<PointType, SplitDimension> >
        ( new KDNode<PointType, SplitDimension>(*median));

    ret->left_child  = build_subtree<NextDimension >(begin, median);
    ret->right_child = build_subtree<NextDimension >(median + 1, end);
    return ret;
  }
}


template<typename PointType, typename PointArray>
void KDTree<PointType, PointArray>::dump_tree_inorder() {
  dump_subtree<0>(root_);
}

template<typename PointType, typename PointArray>
template<int SplitDimension>
void KDTree<PointType, PointArray>::dump_subtree(unique_ptr<KDNode<PointType, SplitDimension> >& node) {
  if (node->left_child) {
    std::cout << "dumping left: " << std::endl;
    dump_subtree<(SplitDimension +1)%PointType::dimension() %3>(node->left_child);
  }
  std::cout << "dumping this: " << std::endl;
  std::cout << node->tree_index << ": " << points_[node->tree_index] << std::endl;
  if (node->right_child) {
    std::cout << "dumping right: " << std::endl;
    dump_subtree<(SplitDimension +1)%PointType::dimension() %3>(node->right_child);
  }
}

template<typename PointType, typename PointArray>
vector<size_t> KDTree<PointType, PointArray>::get_points_within_cube(PointType test_point, T radius) {

  T query_range[2 * PointType::dimension()];
  for(auto i = 0; i < PointType::dimension(); ++i) {
    query_range[2*i] = test_point(i) - radius;
    query_range[2*i +1] = test_point(i) + radius;
  }

  vector<size_t> ret;
  get_points_within_cube_subtree<0>(query_range, root_, ret);
  return ret;
}

template<typename PointType, typename PointArray>
vector<size_t> KDTree<PointType, PointArray>::range_search(PointType query, PointType range) {

  T query_range[2 * PointType::dimension()];
  for(auto i = 0; i < PointType::dimension(); ++i) {
    query_range[2 * i] = query(i) - range(i);
    query_range[2 * i + 1] = query(i) + range(i);
  }

  vector<size_t> ret;
  get_points_within_cube_subtree<0>(query_range, root_, ret);
  return ret;
}

template<typename PointType, typename PointArray>
template<int SplitDimension>
void KDTree<PointType, PointArray>::get_points_within_cube_subtree(T query_range[2 * PointType::dimension()],
                                                                   unique_ptr<KDNode<PointType,
                                                                                     SplitDimension> >& node,
                                                                   vector<size_t>& ret) {

  if (node == nullptr) {
    return;
  }

  constexpr int NextDimension = (SplitDimension +1) % PointType::dimension();
  auto node_point = points_[node->tree_index];
  if (point_in_range(node_point, query_range)) {
    ret.push_back(node->tree_index);
  }
  if (node_point(SplitDimension) >= query_range[2 * SplitDimension]) {
    //query range goes into the left subtree
    //std::cout << "recurse left" << std::endl;
    get_points_within_cube_subtree<NextDimension >(query_range,
                                                   node->left_child,
                                                   ret);
  }

  if (node_point(SplitDimension) <= query_range[2 * SplitDimension + 1]) {
    //query range goes into the right subtree
    //std::cout << "recurse right" << std::endl;
    get_points_within_cube_subtree<NextDimension >(query_range,
                                                   node->right_child,
                                                   ret);
  }

}


template<typename PointType, typename PointArray>
size_t KDTree<PointType, PointArray>::find_min(int dimension) {
  return find_min_subtree<0>(dimension, root_);
}


template<typename PointType, typename PointArray>
template<int SplitDimension>
size_t KDTree<PointType, PointArray>::find_min_subtree(int dimension,
                                                       unique_ptr<KDNode<PointType,
                                                                         SplitDimension> >& node) {
  if (SplitDimension == dimension) {
    if (node->left_child == nullptr) {
      return node->tree_index;
    } else {
      return find_min_subtree<(SplitDimension+1) % PointType::dimension()>(dimension,
                                                                       node->left_child);
    }
  } else {
    size_t left_min = 123456, right_min= 123456;
    if (node->left_child) {
      left_min = find_min_subtree<(SplitDimension+1) % PointType::dimension()>(dimension,
                                                                           node->left_child);
    }
    if (node->right_child) {
      right_min = find_min_subtree<(SplitDimension+1) % PointType::dimension()>(dimension,
                                                                            node->right_child);
    }

    auto nodeValue = points_[node->tree_index](dimension);
    if (node->left_child &&
        points_[left_min](dimension) <
            nodeValue) {

      if (node->right_child) {
        return (points_[left_min](dimension) <
            points_[right_min](dimension)) ? left_min : right_min;

      } else {
        return left_min;
      }
    } else if (node->right_child &&
        points_[right_min](dimension) <
            nodeValue) {
      return right_min;
    } else {
      return node->tree_index;
    }
  }
}

template<typename PointType, typename PointArray>
void KDTree<PointType, PointArray>::delete_point(size_t node_index) {

  root_ = delete_from_subtree<0>(node_index, root_);
}

template<typename PointType, typename PointArray>
template<int SplitDimension>
unique_ptr<KDNode<PointType, SplitDimension> >
KDTree<PointType, PointArray>::delete_from_subtree(size_t nodeIndex,
                                                   unique_ptr<KDNode<PointType, SplitDimension> >& node) {

  constexpr size_t NextDimension = (SplitDimension +1) % PointType::dimension();

  if (node->tree_index == nodeIndex) {
    if (node->right_child) {
      auto rightMin = find_min_subtree<NextDimension>(SplitDimension, node->right_child);
      node->tree_index = rightMin;
      node->right_child = delete_from_subtree<NextDimension>(rightMin,
                                                             node->right_child);
    } else if (node->left_child) {
      auto leftMin = find_min_subtree<NextDimension>(SplitDimension, node->left_child);
      node->tree_index = leftMin;
      node->right_child = delete_from_subtree<NextDimension>(leftMin,
                                                             node->left_child);
      node->left_child = nullptr;
    } else {
      return nullptr;
    }
  } else if (points_[nodeIndex](SplitDimension) <
      points_[node->tree_index](SplitDimension)) {

    node->left_child = delete_from_subtree<NextDimension>(nodeIndex,
                                                          node->left_child);
  } else {
    node->right_child = delete_from_subtree<NextDimension>(nodeIndex,
                                                           node->right_child);
  }
  return std::move(node);

}

template<typename PointType, typename PointArray>
template <typename F>
void KDTree<PointType, PointArray>::inorder_traversal(F func) {

  inorder_traversal_subtree<0, F>(func, root_);

}

template<typename PointType, typename PointArray>
template<int SplitDimension, typename F>
void KDTree<PointType, PointArray>::inorder_traversal_subtree(F func,
                                                              unique_ptr<KDNode<PointType,
                                                                                SplitDimension> >& node) {
  auto constexpr nextDimension = (SplitDimension +1)%PointType::dimension();
  if (node->left_child) {
    inorder_traversal_subtree<nextDimension, F>(func, node->left_child);
  }
  func(points_[node->tree_index]);
  if (node->right_child) {
    inorder_traversal_subtree<nextDimension, F>(func, node->right_child);
  }

}

template<typename PointType, typename PointArray>
void KDTree<PointType, PointArray>::insert_point(const PointType& point) {
  points_.push_back(point);
  root_ = insert_point_subtree<0>(root_, points_.size() -1);
}

template<typename PointType, typename PointArray>
template<int SplitDimension>
unique_ptr<KDNode<PointType, SplitDimension> >
KDTree<PointType, PointArray>::insert_point_subtree(unique_ptr<KDNode<PointType, SplitDimension> >& node,
                                                    size_t point_index) {

  auto constexpr nextDimension = (SplitDimension +1)%PointType::dimension();

  if (node == nullptr) {
    std::cout << "new node" << std::endl;
    return unique_ptr<KDNode<PointType, SplitDimension> > (new KDNode<PointType, SplitDimension>(point_index));
  } else if (points_[point_index](SplitDimension) <
      points_[node->tree_index](SplitDimension)) {
    std::cout << "adding left" << std::endl;
    node->left_child = insert_point_subtree<nextDimension>(node->left_child, point_index);
    std::cout << "added left " << std::endl;
  } else {
    std::cout << "adding right " << std::endl;
    node->right_child = insert_point_subtree<nextDimension>(node->right_child, point_index);
    std::cout << "added right" << std::endl;
  }
  return std::move(node);

}
#endif //_BJ_KD_TREE_H
