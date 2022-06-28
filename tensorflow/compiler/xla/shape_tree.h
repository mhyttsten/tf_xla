/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
#define TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace internal {

class IndexTable {
 public:
  // Use indices, rather than pointers, so index table can be copied between
  // ShapeTrees.
  struct Entry {
    // Index of the node in the nodes vector.
    size_t node_id;
    // Index of the first child of this node in the index table (-1 for leaves).
    std::make_signed_t<size_t> children_start_id = -1;
  };

  IndexTable() = default;
  explicit IndexTable(const Shape& shape);

  bool empty() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/shape_tree.h", "empty");
 return entries_.empty(); }

  const Entry& operator[](ShapeIndexView index) const;

 private:
  void CreateEntry(Entry& entry, const Shape& shape, size_t& next_node_id);

  absl::InlinedVector<Entry, 1> entries_;
};

}  // namespace internal

// A ShapeTree<T> is a recursive data structure which mirrors the structure of a
// XLA shape and holds a value of type T for each subshape (i.e. tuple or array)
// in the shape. For array shapes, a ShapeTree trivially holds a single value of
// type T.
//
// For tuple shapes which can be an arbitrary tree with arrays at the leaves, a
// ShapeTree is an identically structured tree with data elements of type T at
// every node. I.e. the root is a tuple by definition, all interior nodes are
// also tuples, and all leaves are arrays.
//
// Like the Shape data structure, this is a tree and tuple elements cannot be
// duplicated. That is, every distinct ShapeIndex in the Shape has a unique T
// object.
//
// Normally a ShapeTree owns its Shape, but for efficiency reasons, sometimes
// it's helpful not to copy a Shape just to make a ShapeTree.  In these cases,
// you can pass a Shape* instead of a Shape to the ShapeTree constructor.  It's
// then up to you to ensure that the pointed-to Shape isn't freed, moved or
// modified before its ShapeTree goes away.
template <typename T>
class ShapeTree {
  template <typename U>
  friend class ShapeTree;

 public:
  // TODO(cjfj): Don't store ShapeIndex with data. Generate it or cache it?
  using Node = std::pair<ShapeIndex, T>;
  using Nodes = absl::InlinedVector<Node, 1>;
  using IndexTable = internal::IndexTable;

  template <typename Iterator, typename ValueType>
  class LeafIterator;

  // Default constructor creates a tree with a nil shape (i.e. an empty tuple).
  ShapeTree() : ShapeTree(ShapeUtil::MakeNil()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_1(mht_1_v, 273, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");
}

  // Create ShapeTree with the given shape, and default-constructed T values for
  // all nodes.
  //
  // The version that takes a pointer may be cheaper because it doesn't require
  // any Shape copies, but then it's up to you to ensure that the pointer stays
  // alive longer than this ShapeTree.
  explicit ShapeTree(Shape shape)
      : ShapeTree(std::make_shared<Shape>(std::move(shape))) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_2(mht_2_v, 285, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");
}

  explicit ShapeTree(const Shape* shape)
      : ShapeTree(shape, CreateNodes(*shape)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_3(mht_3_v, 291, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");
}

  // Create ShapeTree with the given shape, and init_value for all nodes.
  ShapeTree(Shape shape, const T& init_value)
      : ShapeTree(std::make_shared<Shape>(std::move(shape)), init_value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_4(mht_4_v, 298, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");
}

  ShapeTree(const Shape* shape, const T& init_value)
      : ShapeTree(shape, CreateNodes(*shape, init_value)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_5(mht_5_v, 304, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");
}

  // Returns the data element associated with the array in the shape at the
  // given index (see ShapeUtil::GetSubshape for how indexes are defined).
  const T& element(ShapeIndexView index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_6(mht_6_v, 311, "", "./tensorflow/compiler/xla/shape_tree.h", "element");
 return find(index)->second; }
  T* mutable_element(ShapeIndexView index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_7(mht_7_v, 315, "", "./tensorflow/compiler/xla/shape_tree.h", "mutable_element");
 return &find(index)->second; }

  // Return the shape represented with this ShapeTree.
  const Shape& shape() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_8(mht_8_v, 321, "", "./tensorflow/compiler/xla/shape_tree.h", "shape");
 return *shape_; }

  // A ShapeTree object can own the underlying Shape pointer (via the
  // shape_storage_ member), or can point to a Shape object owned by the caller.
  // This API replaces the underlying Shape object to the one supplied by the
  // caller, whom must ensure the object remain valid for the whole lifetime of
  // this ShapeTree object, and also that the Shape is consistent with it.
  void replace_shape_ptr(const Shape& shape) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_9(mht_9_v, 331, "", "./tensorflow/compiler/xla/shape_tree.h", "replace_shape_ptr");

    if (shape_storage_ != nullptr) {
      DCHECK_EQ(shape, *shape_storage_);
      shape_storage_ = nullptr;
    }
    shape_ = &shape;
  }

  // Returns true if the node at the given index is a leaf node (an array
  // shape).
  bool IsLeaf(ShapeIndexView index) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_10(mht_10_v, 344, "", "./tensorflow/compiler/xla/shape_tree.h", "IsLeaf");

    return index_table_[index].children_start_id == -1;
  }

  using iterator = typename Nodes::iterator;
  using const_iterator = typename Nodes::const_iterator;
  using reverse_iterator = typename Nodes::reverse_iterator;
  using const_reverse_iterator = typename Nodes::const_reverse_iterator;

  using leaf_iterator = LeafIterator<iterator, Node>;
  using const_leaf_iterator = LeafIterator<const_iterator, const Node>;
  using reverse_leaf_iterator = std::reverse_iterator<leaf_iterator>;
  using const_reverse_leaf_iterator =
      std::reverse_iterator<const_leaf_iterator>;

  iterator begin() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_11(mht_11_v, 362, "", "./tensorflow/compiler/xla/shape_tree.h", "begin");
 return nodes_.begin(); }
  iterator end() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_12(mht_12_v, 366, "", "./tensorflow/compiler/xla/shape_tree.h", "end");
 return nodes_.end(); }
  const_iterator begin() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_13(mht_13_v, 370, "", "./tensorflow/compiler/xla/shape_tree.h", "begin");
 return nodes_.begin(); }
  const_iterator end() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_14(mht_14_v, 374, "", "./tensorflow/compiler/xla/shape_tree.h", "end");
 return nodes_.end(); }

  reverse_iterator rbegin() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_15(mht_15_v, 379, "", "./tensorflow/compiler/xla/shape_tree.h", "rbegin");
 return nodes_.rbegin(); }
  reverse_iterator rend() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_16(mht_16_v, 383, "", "./tensorflow/compiler/xla/shape_tree.h", "rend");
 return nodes_.rend(); }
  const_reverse_iterator rbegin() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_17(mht_17_v, 387, "", "./tensorflow/compiler/xla/shape_tree.h", "rbegin");
 return nodes_.rbegin(); }
  const_reverse_iterator rend() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_18(mht_18_v, 391, "", "./tensorflow/compiler/xla/shape_tree.h", "rend");
 return nodes_.rend(); }

  // leaf_begin()/leaf_end() iterates over all leaf nodes (nodes with no
  // children).
  leaf_iterator leaf_begin() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_19(mht_19_v, 398, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_begin");
 return leaf_iterator(*this, nodes_.begin()); }
  leaf_iterator leaf_end() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_20(mht_20_v, 402, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_end");
 return leaf_iterator(*this, nodes_.end()); }
  const_leaf_iterator leaf_begin() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_21(mht_21_v, 406, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_begin");

    return const_leaf_iterator(*this, nodes_.begin());
  }
  const_leaf_iterator leaf_end() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_22(mht_22_v, 412, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_end");

    return const_leaf_iterator(*this, nodes_.end());
  }
  // range-based iterator for leaf_begin()/leaf_end().
  tensorflow::gtl::iterator_range<leaf_iterator> leaves() {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }
  tensorflow::gtl::iterator_range<const_leaf_iterator> leaves() const {
    return tensorflow::gtl::make_range(leaf_begin(), leaf_end());
  }

  reverse_leaf_iterator leaf_rbegin() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_23(mht_23_v, 426, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_rbegin");

    return reverse_leaf_iterator(leaf_end());
  }
  reverse_leaf_iterator leaf_rend() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_24(mht_24_v, 432, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_rend");

    return reverse_leaf_iterator(leaf_begin());
  }
  const_reverse_leaf_iterator leaf_rbegin() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_25(mht_25_v, 438, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_rbegin");

    return const_reverse_leaf_iterator(leaf_end());
  }
  const_reverse_leaf_iterator leaf_rend() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_26(mht_26_v, 444, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_rend");

    return const_reverse_leaf_iterator(leaf_begin());
  }

  // Returns an iterator pointing to the given ShapeIndex.
  // REQUIRES: index must exist in the ShapeTree.
  iterator find(ShapeIndexView index) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_27(mht_27_v, 453, "", "./tensorflow/compiler/xla/shape_tree.h", "find");

    return nodes_.begin() + index_table_[index].node_id;
  }
  const_iterator find(ShapeIndexView index) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_28(mht_28_v, 459, "", "./tensorflow/compiler/xla/shape_tree.h", "find");

    return nodes_.begin() + index_table_[index].node_id;
  }

  // Returns the number of leaf nodes in the tree.
  int64_t leaf_count() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_29(mht_29_v, 467, "", "./tensorflow/compiler/xla/shape_tree.h", "leaf_count");
 return std::distance(leaf_begin(), leaf_end()); }

  // TODO(cjfj): Remove the `ForEach...` methods. They are redundant.
  // Recursively traverses the shape and calls the given function at each
  // element.
  void ForEachElement(
      absl::FunctionRef<void(const ShapeIndex&, const T&)> func) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_30(mht_30_v, 476, "", "./tensorflow/compiler/xla/shape_tree.h", "ForEachElement");

    for (const Node& node : nodes_) {
      func(node.first, node.second);
    }
  }

  void ForEachMutableElement(
      absl::FunctionRef<void(const ShapeIndex&, T*)> func) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_31(mht_31_v, 486, "", "./tensorflow/compiler/xla/shape_tree.h", "ForEachMutableElement");

    for (Node& node : nodes_) {
      func(node.first, &node.second);
    }
  }

  // Like ForEach(Mutable)Element, but the callable returns a Status instead of
  // void.  The first non-OK return value is returned by the ForEach* function.
  Status ForEachElementWithStatus(
      absl::FunctionRef<Status(const ShapeIndex&, const T&)> func) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_32(mht_32_v, 498, "", "./tensorflow/compiler/xla/shape_tree.h", "ForEachElementWithStatus");

    for (const Node& node : nodes_) {
      TF_RETURN_IF_ERROR(func(node.first, node.second));
    }
    return Status::OK();
  }

  Status ForEachMutableElementWithStatus(
      absl::FunctionRef<Status(const ShapeIndex&, T*)> func) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_33(mht_33_v, 509, "", "./tensorflow/compiler/xla/shape_tree.h", "ForEachMutableElementWithStatus");

    for (Node& node : nodes_) {
      TF_RETURN_IF_ERROR(func(node.first, &node.second));
    }
    return Status::OK();
  }

  // Maps each element to generate a new tree with the same shape.
  template <typename U>
  ShapeTree<U> Map(absl::FunctionRef<U(const T&)> func) {
    typename ShapeTree<U>::Nodes result_nodes;
    result_nodes.reserve(nodes_.size());
    for (const Node& node : nodes_) {
      result_nodes.push_back({node.first, func(node.second)});
    }

    ShapeTree<U> result(shape_, std::move(result_nodes));
    result.index_table_ = index_table_;
    result.shape_storage_ = shape_storage_;
    return result;
  }

  // Copy the subtree of values from 'other' rooted at ShapeIndex 'src_index'
  // into the subtree of value in this ShapeTree rooted at 'dst_index'.
  //
  // Precondition: The subshape of other.shape() at index src_index must be
  // compatible with the subshape of shape() at index dst_index.
  void CopySubtreeFrom(const ShapeTree<T>& other, const ShapeIndex& src_index,
                       const ShapeIndex& dst_index) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_34(mht_34_v, 540, "", "./tensorflow/compiler/xla/shape_tree.h", "CopySubtreeFrom");

    const Shape& src_shape = ShapeUtil::GetSubshape(other.shape(), src_index);
    const Shape& dst_shape = ShapeUtil::GetSubshape(shape(), dst_index);
    CHECK(ShapeUtil::Compatible(src_shape, dst_shape))
        << src_shape << ", " << dst_shape;

    // Replace the prefix `src_index` with `dst_index`.
    auto replace_shape_index_prefix = [&](const ShapeIndex& index) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_35(mht_35_v, 550, "", "./tensorflow/compiler/xla/shape_tree.h", "lambda");

      auto without_prefix = ShapeIndexView(index).subspan(src_index.size());
      ShapeIndex result;
      result.reserve(dst_index.size() + without_prefix.size());
      result.insert(result.end(), dst_index.begin(), dst_index.end());
      result.insert(result.end(), without_prefix.begin(), without_prefix.end());
      return result;
    };

    auto first = other.find(src_index);
    auto last = first + ShapeUtil::SubshapeCount(src_shape);

    std::transform(first, last, find(dst_index), [&](const Node& node) -> Node {
      return {replace_shape_index_prefix(node.first), node.second};
    });
  }

  StatusOr<ShapeTree<T>> SubShapeTree(const ShapeIndex& index) const {
    TF_ASSIGN_OR_RETURN(const Shape* sub_shape,
                        ShapeUtil::TryGetSubshape(shape(), index));
    size_t count = ShapeUtil::SubshapeCount(*sub_shape);
    Nodes sub_tree_nodes;
    sub_tree_nodes.reserve(count);
    for (auto it = find(index), end = it + count; it != end; ++it) {
      // For each shape index, remove the prefix `index`.
      auto without_prefix = ShapeIndexView(it->first).subspan(index.size());
      sub_tree_nodes.push_back(Node{without_prefix, it->second});
    }
    return ShapeTree(sub_shape, std::move(sub_tree_nodes));
  }

  bool operator==(const ShapeTree<T>& other) const {
    return nodes_ == other.nodes_;
  }
  bool operator!=(const ShapeTree<T>& other) const { return !(*this == other); }

 private:
  explicit ShapeTree(std::shared_ptr<Shape> shape) : ShapeTree(shape.get()) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_36(mht_36_v, 590, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");

    shape_storage_.swap(shape);
  }

  ShapeTree(std::shared_ptr<Shape> shape, const T& init_value)
      : ShapeTree(shape.get(), init_value) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_37(mht_37_v, 598, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");

    shape_storage_.swap(shape);
  }

  ShapeTree(const Shape* shape, Nodes nodes)
      : nodes_(std::move(nodes)), index_table_(*shape), shape_(shape) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_38(mht_38_v, 606, "", "./tensorflow/compiler/xla/shape_tree.h", "ShapeTree");

    DCHECK_EQ(nodes_.size(), ShapeUtil::SubshapeCount(*shape));
  }

  template <typename... Ts>
  static Nodes CreateNodes(const Shape& shape, Ts&&... args) {
    Nodes nodes;
    ShapeUtil::ForEachSubshape(
        shape, [&](const Shape&, const ShapeIndex& index) {
          nodes.push_back({index, T(std::forward<Ts>(args)...)});
        });
    return nodes;
  }

  // The nodes in this shape tree.
  Nodes nodes_;

  // Index table for node lookups. Each entry contains the index of the first
  // child of the node at that index, or -1 for leaf nodes. Evaluated lazily.
  IndexTable index_table_;

  // If we own our Shape, this field contains it, and shape_ is a pointer into
  // here.  Otherwise if we don't own our shape, this is nullptr.
  std::shared_ptr<Shape> shape_storage_;

  // The XLA shape mirrored in this ShapeTree.  This is either
  // shape_storage_.get() or the Shape pointer passed to our constructor.
  const Shape* shape_;
};

// Internal iterator that performs a pre-order walk of the leaves. This is cheap
// to copy. The iterator value_type is equivalent to a std::pair<ShapeIndex,T>&,
// similar to std::map.
template <typename T>
template <typename Iterator, typename ValueType>
class ShapeTree<T>::LeafIterator
    : public std::iterator<std::bidirectional_iterator_tag, ValueType> {
 public:
  LeafIterator(const ShapeTree& tree, Iterator it) : tree_(tree), it_(it) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_39(mht_39_v, 647, "", "./tensorflow/compiler/xla/shape_tree.h", "LeafIterator");

    while ((it_ != tree_.nodes_.end()) && !IsLeaf()) ++it_;
  }

  LeafIterator& operator++() {
    do {
      ++it_;
    } while ((it_ != tree_.nodes_.end()) && !IsLeaf());
    return *this;
  }

  LeafIterator operator++(int) {
    auto prev = *this;
    ++(*this);
    return prev;
  }

  LeafIterator& operator--() {
    do {
      --it_;
    } while ((it_ != tree_.nodes_.begin()) && !IsLeaf());
    return *this;
  }

  LeafIterator operator--(int) {
    auto prev = *this;
    --(*this);
    return prev;
  }

  bool operator==(const LeafIterator& other) const { return it_ == other.it_; }
  bool operator!=(const LeafIterator& other) const { return !(*this == other); }
  ValueType& operator*() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_40(mht_40_v, 682, "", "./tensorflow/compiler/xla/shape_tree.h", "*");
 return *it_; }
  ValueType* operator->() const { return &*it_; }

 private:
  bool IsLeaf() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_treeDTh mht_41(mht_41_v, 689, "", "./tensorflow/compiler/xla/shape_tree.h", "IsLeaf");
 return tree_.IsLeaf(it_->first); }

  const ShapeTree<T>& tree_;
  Iterator it_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_TREE_H_
