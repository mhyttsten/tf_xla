/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_SLICE_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_SLICE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh() {
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


#include <string>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// A tensor slice represents a slice of a given tensor. It is represented by a
// list of (start, length) pairs, where the size of the list is the rank of the
// tensor.

class TensorSlice {
 public:
  // Construct a tensor slice: you have a number of ways:
  // -- creating an empty slice
  // -- from just a dimension (in this case it will create a full slice)
  // -- from an array of pairs of integers.
  // -- from a TensorSliceProto protocol buffer
  // -- from a string format of "start,length:start,length..." where each
  //    "start,length" pair represents the slice on one dimension. We allow a
  //    special "-" that means "everything for this dimension". One such example
  //    is:  0,10:-:14,1:-:-
  TensorSlice() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/framework/tensor_slice.h", "TensorSlice");
}
  explicit TensorSlice(int dim);
  explicit TensorSlice(const TensorSliceProto& proto);
  explicit TensorSlice(
      std::initializer_list<std::pair<int64_t, int64_t>> extents);

  // This factory methods should be used instead of the constructor that takes a
  // `TensorSliceProto` if calling code cannot validate that the sizes specify a
  // valid `TensorSlice`.
  static Status BuildTensorSlice(const TensorSliceProto& proto,
                                 TensorSlice* output);

  static Status Parse(const string& str, TensorSlice* output);
  static TensorSlice ParseOrDie(const string& str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/framework/tensor_slice.h", "ParseOrDie");

    TensorSlice ret;
    Status s = Parse(str, &ret);
    if (!s.ok()) {
      LOG(FATAL) << "Could not parse TensorSlice";
    }
    return ret;
  }

  void Clear();

  // Accessors
  int dims() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/framework/tensor_slice.h", "dims");
 return starts_.size(); }

  int64_t start(int d) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/framework/tensor_slice.h", "start");

    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return starts_[d];
  }

  int64_t length(int d) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_4(mht_4_v, 260, "", "./tensorflow/core/framework/tensor_slice.h", "length");

    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return lengths_[d];
  }

  int64_t end(int d) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_5(mht_5_v, 269, "", "./tensorflow/core/framework/tensor_slice.h", "end");

    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    return start(d) + length(d);
  }

  void set_start(int d, int64_t x) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_6(mht_6_v, 278, "", "./tensorflow/core/framework/tensor_slice.h", "set_start");

    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    DCHECK_GE(x, 0);
    starts_[d] = x;
  }

  void set_length(int d, int64_t x) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_7(mht_7_v, 288, "", "./tensorflow/core/framework/tensor_slice.h", "set_length");

    DCHECK_GE(d, 0);
    DCHECK_LT(d, dims());
    lengths_[d] = x;
  }

  // If we have a full slice along dimension "d".
  bool IsFullAt(int d) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_8(mht_8_v, 298, "", "./tensorflow/core/framework/tensor_slice.h", "IsFullAt");

    return lengths_[d] == kFullExtent && starts_[d] == 0;
  }

  // If this is a full slice, i.e. IsFullAt(d) for every d.
  bool IsFull() const;

  // Set the slice to be a full slice of "dim" dimensions
  void SetFullSlice(int dim);

  // Extend a slice to "dim" dimensions: all the added dimensions are full.
  // Requires: dim >= dims().
  void Extend(int dim);

  // Conversion of a TensorSlice to other formats
  void AsProto(TensorSliceProto* proto) const;
  string DebugString() const;

  // Fill *indices and *sizes from *this (so that we can use the slice()
  // function in eigen tensor). We need a tensor shape in case some of the
  // slices are full slices.
  // We allow NDIMS to be greater than dims(), in which case we will pad the
  // higher dimensions with trivial dimensions.
  template <int NDIMS>
  void FillIndicesAndSizes(
      const TensorShape& shape,
      Eigen::DSizes<Eigen::DenseIndex, NDIMS>* indices,
      Eigen::DSizes<Eigen::DenseIndex, NDIMS>* sizes) const;

  // Interaction with other TensorSlices.

  // Compute the intersection with another slice and if "result" is not
  // nullptr, store the results in *result; returns true if there is any real
  // intersection.
  bool Intersect(const TensorSlice& other, TensorSlice* result) const;
  // A short hand.
  bool Overlaps(const TensorSlice& other) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_9(mht_9_v, 337, "", "./tensorflow/core/framework/tensor_slice.h", "Overlaps");

    return Intersect(other, nullptr);
  }

  // Equals iff "*this" and "other" are logically equivalent.
  bool operator==(const TensorSlice& other) const;
  bool operator!=(const TensorSlice& other) const { return !(*this == other); }

  // Interaction with TensorShape.

  // Slices a shape and stores the result into *result_shape.
  // Requires that the shape and *this have the same rank.
  // For example, given a tensor shape of {3, 4, 5}, and a slice of
  // 1,2:-:0,2, the result shape is {2, 4, 2}.
  Status SliceTensorShape(const TensorShape& shape,
                          TensorShape* result_shape) const;

  // Given slice "sub" where "sub" is fully contained in *this,
  // (meaning that the intersection of "sub" and *this equals "sub"), computes
  // the "relative" slice of "sub" with respect to *this.
  //
  // In other words, if we use A>S to denote slicing a shape S with a slice A,
  // then the function is computing a slice X such that:
  //   X > (this > S) = sub > S
  // for any shape S.
  //
  // In general, along every dimension, the start of the relative slice is the
  // start of the "sub" slice minus the start of *this; the length of the
  // relative slice is the length of the "sub" slice.
  //
  // For example, say we have a shape of {3, 4, 5}, "this" is 0,2:-:1,2, and
  // "sub" is 1,1:2:2,1,2, then the related slice is 1,1:2,2:0,2.
  //
  // The caller needs to make sure that "sub" is indeed a sub-slice of *this;
  // otherwise the result is undefined.
  void ComputeRelative(const TensorSlice& sub, TensorSlice* relative) const;

  // Updates the slice in such a way that it fully covers "other" slice.
  // Note, "other" slice should refer to the same tensor shape.
  // Example:
  //   given a slice [2:4, :, 3:] and "other" slice [:, 1:4, 2:4] the
  //   updated slice would be [:, :, 2:]. Here is why:
  //   dim 0: "2:4"  U  ":"    ->  ":"
  //   dim 1: ":"    U  "1-4"  ->  ":"
  //   dim 2: "3:"   U  "2:4"  ->  "2:"
  void UpdateToCover(const TensorSlice& other);

  // Returns true if the length field was specified in an Extent.
  static bool HasExtentLength(const TensorSliceProto::Extent& extent);

  // Returns the value of the length field in an Extent, or -1 if it
  // is not present.
  static int64_t GetExtentLength(const TensorSliceProto::Extent& extent);

 private:
  // a length value of kFullExtent (-1) means we have a full slice at this
  // dimension. It's defined in tensor_slice.cc.
  static const int64_t kFullExtent;

  // TODO(yangke): switch to Eigen once it supports variable size arrays.
  // A value of
  gtl::InlinedVector<int64_t, 4> starts_;
  gtl::InlinedVector<int64_t, 4> lengths_;
};

template <int NDIMS>
void TensorSlice::FillIndicesAndSizes(
    const TensorShape& shape, Eigen::DSizes<Eigen::DenseIndex, NDIMS>* indices,
    Eigen::DSizes<Eigen::DenseIndex, NDIMS>* sizes) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTh mht_10(mht_10_v, 408, "", "./tensorflow/core/framework/tensor_slice.h", "TensorSlice::FillIndicesAndSizes");

  CHECK_EQ(shape.dims(), dims()) << "Incompatible dimensions between shape "
                                 << "slices: shape = " << shape.DebugString()
                                 << ", slice = " << DebugString();
  CHECK_GE(NDIMS, dims()) << "Asking for a " << NDIMS << "-dim slice from "
                          << "a slice of dimension " << dims();
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      (*indices)[d] = 0;
      (*sizes)[d] = shape.dim_size(d);
    } else {
      (*indices)[d] = starts_[d];
      (*sizes)[d] = lengths_[d];
    }
  }
  for (int d = dims(); d < NDIMS; ++d) {
    (*indices)[d] = 0;
    (*sizes)[d] = 1;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_SLICE_H_
