/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh() {
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


#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

// ClusterBatchSize is a data structure to record the batch size we have seen
// for a cluster during segmentation.
//
// With the help of shape inference, all the dynamic batch sizes are converted
// to a negative integer number.
// If the number is -1, then nothing is known about the dynamic batch size.
// Ideally, we should not put nodes with -1 batch size into the same cluster,
// as they will likely have different batch sizes at runtime. However, we
// currently treat -1 as an equivalent class for simple implementation. We may
// need to revise this if it causes performance issues.
// If the number is strictly less than -1, then it represents a equivalent
// class. It is infered that all the nodes with the same equivalent class
// (strictly less than -1) shall have the same batch size at runtime.
//
// When constructing clusters for implicit batch mode, we support both
// dynamic batch sizes and static batch sizes. As all the nodes inside the same
// cluster shall have the same batch size at runtime, we restrict nodes inside a
// cluster to either have the same dynamic batch size equivalent class or the
// same static batch size value.
//
// Besides, all the nodes with an annotated max batch size inside the same
// cluster shall have the same annotated max batch size. (It is allowed if
// part or all the nodes inside the cluster doesn't have annotated max batch
// size). Static batch sizes are treated as max batch size annotations. The
// converter max batch size is used for an OP with a dynamic batch size and no
// annotated max batch size.
//
// cluster:  a = a1[1,3] + a1[1,3]
// ClusterBatchSize: batch_size_ = 1
//                   max_batch_size_ = 1
//
// cluster:  b = b1[-1,3] + b2[-1, 3]
// ClusterBatchSize: batch_size_ = -1
//                   max_batch_size_ = null
//
// cluster:  c = c1[-2,3] + c2[-2, 3](max_batch_size=100)
// ClusterBatchSize: batch_size_ = -2
//                   max_batch_size_ = 100
//
// When constructing cluster for explicit batch mode, all ClusterBatchSize is
// irrelevant.
//

class ClusterBatchSize {
 public:
  ClusterBatchSize();

  bool operator==(const ClusterBatchSize& other);
  bool operator!=(const ClusterBatchSize& other) { return !(*this == other); }

  // Sets the batch size assuming that the object doesn't have a batch size yet:
  //   A non-negative input representing a static batch size value.
  //   A negative input representing a dynamic batch size equivalent class.
  ClusterBatchSize& SetBatchSize(int batch_size);
  bool HasBatchSize() const;
  int GetBatchSize() const;

  // Sets the max batch size assuming that the object doesn't have a max batch
  // size yet.
  ClusterBatchSize& SetMaxBatchSize(int max_batch_size);
  absl::optional<int> GetOptionalMaxBatchSize() const;

  // Merge `other` into the current ClusterBatchSize if the two are not
  // conflicting. Two ClusterBatchSizes are conflicting iff they both have a
  // value and their values are different.
  bool MergeIfCompatible(const ClusterBatchSize& other);

  // Returns a string for the batch size and the annotated max batch size.
  // For the batch size:
  //   If the object has a static batch size, return a string representing a
  //     non-negative integer.
  //   If the object has a dynamic batch size, return a string representing a
  //     negative integer as an equivalent class.
  //   If the object doesn't have a batch size yet, return "?".
  // For the annotated max batch size:
  //   If the cluster has annotated max batch size in at least one of the nodes,
  //     return a string representing the annotated max batch size. Otherwise,
  //     return "?".
  std::string ToString() const;

 private:
  ClusterBatchSize& SetBatchSize(const absl::optional<int>& batch_size);
  ClusterBatchSize& SetMaxBatchSize(const absl::optional<int>& batch_size);

  absl::optional<int> batch_size_;
  absl::optional<int> max_batch_size_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ClusterBatchSize& batch_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_0(mht_0_v, 287, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "operator<<");

  return os << batch_size.ToString();
}

// Represents the accumulated properties of a cluster during segmentation,
// including information about batch size and device assignment. Clusters shall
// have compatible properties in order to be merged together.
class ClusterProperty {
 public:
  ClusterProperty() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_1(mht_1_v, 299, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "ClusterProperty");
}
  ClusterProperty(const ClusterBatchSize& batch_size,
                  const DeviceNameUtils::ParsedName& device_name);

  // Returns the batch size of the cluster and compresses the path from this
  // object to the root object.
  const ClusterBatchSize& BatchSize() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_2(mht_2_v, 308, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "BatchSize");
 return batch_size_; }

  // Returns the device name of the cluster and compresses the path from this
  // object to the root object.
  const DeviceNameUtils::ParsedName& DeviceName() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_3(mht_3_v, 315, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "DeviceName");
 return device_name_; }

  Status Merge(const ClusterProperty& other);

 private:
  ClusterBatchSize batch_size_;
  DeviceNameUtils::ParsedName device_name_;
};

// Represents a disjoint set of copyable value with type T and accumulated
// property of the values with type P. Most of the methods in this class are
// side-effecting as they also compress the path from the object to the parent
// of its containing set.
template <typename T, typename P = ClusterProperty>
class UnionFind {
 public:
  UnionFind() : size_(1), parent_(nullptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_4(mht_4_v, 334, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "UnionFind");
}
  UnionFind(const T& v, const P& p)
      : size_(1), parent_(nullptr), value_(v), property_(p) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_5(mht_5_v, 339, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "UnionFind");
}
  UnionFind(const T& v, P&& p)
      : size_(1), parent_(nullptr), value_(v), property_(p) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_6(mht_6_v, 344, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "UnionFind");
}

  // Returns the number of elements in the set and compresses the path from
  // this object to the root of the set.
  int Size() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_7(mht_7_v, 351, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "Size");
 return FindRoot()->size_; }

  // Returns the accumulated property of all the elements in the set and
  // compresses the path from this object to the root of the set.
  const P& Property() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_8(mht_8_v, 358, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "Property");
 return FindRoot()->property_; }

  // Merges this set with 'other'. This updates the size_ and property_ of the
  // set. The size_ and property_ of 'other' becomes inaccessible as only the
  // size_ and property_ of the root of the set is accessible.
  Status Merge(UnionFind* other);

  // Retrieves the value for the root of the set.
  const T& ParentValue() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_9(mht_9_v, 369, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "ParentValue");
 return FindRoot()->value_; }

  // Returns the value for the object.
  const T& Value() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSsegmentPSunion_findDTh mht_10(mht_10_v, 375, "", "./tensorflow/compiler/tf2tensorrt/segment/union_find.h", "Value");
 return value_; }

 private:
  // Returns the root object for the set and compresses the path from this
  // object to the root object.
  UnionFind* FindRoot();

  int size_;
  UnionFind* parent_;
  T value_;
  P property_;
};

template <typename T, typename P>
Status UnionFind<T, P>::Merge(UnionFind* other) {
  UnionFind<T>* a = FindRoot();
  UnionFind<T>* b = other->FindRoot();
  if (a == b) return Status::OK();

  P merged_property(a->property_);
  TF_RETURN_IF_ERROR(merged_property.Merge(b->property_));
  b->parent_ = a;
  a->size_ += b->size_;
  a->property_ = std::move(merged_property);
  return Status::OK();
}

template <typename T, typename P>
UnionFind<T, P>* UnionFind<T, P>::FindRoot() {
  if (!parent_) return this;
  // Path compression: update intermediate nodes to point to the root of the
  // equivalence class.
  parent_ = parent_->FindRoot();
  return parent_;
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_SEGMENT_UNION_FIND_H_
