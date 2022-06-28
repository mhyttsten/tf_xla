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
class MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/reduction_ops_common.h"

#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

TensorShape ReductionHelper::out_reshape() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/kernels/reduction_ops_common.cc", "ReductionHelper::out_reshape");

  TensorShape shape;
  for (auto size : out_reshape_) shape.AddDim(size);
  return shape;
}

// The final output shape must be allocated with this shape.
TensorShape ReductionHelper::out_shape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/kernels/reduction_ops_common.cc", "ReductionHelper::out_shape");

  TensorShape shape;
  for (auto size : out_shape_) shape.AddDim(size);
  return shape;
}

TensorShape ReductionHelper::shuffled_shape() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc mht_2(mht_2_v, 210, "", "./tensorflow/core/kernels/reduction_ops_common.cc", "ReductionHelper::shuffled_shape");

  const int dims = data_reshape_.size();
  TensorShape shape;
  for (int i = reduce_first_axis_; i < dims; i += 2) {
    shape.AddDim(data_reshape_[i]);
  }
  for (int i = !reduce_first_axis_; i < dims; i += 2) {
    shape.AddDim(data_reshape_[i]);
  }
  return shape;
}

gtl::InlinedVector<int32, 8> ReductionHelper::permutation() {
  const int dims = data_reshape_.size();
  const int unreduced_dims = (dims + !reduce_first_axis_) / 2;
  gtl::InlinedVector<int32, 8> perm(dims);
  for (int i = 0; i < unreduced_dims; i++) {
    perm[i] = 2 * i + reduce_first_axis_;
  }
  for (int i = unreduced_dims; i < dims; i++) {
    perm[i] = 2 * (i - unreduced_dims) + !reduce_first_axis_;
  }
  return perm;
}

template <typename Tperm>
Status SimplifyHelper(const Tensor& data, const Tensor& axis,
                      gtl::InlinedVector<bool, 4>& bitmap) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/kernels/reduction_ops_common.cc", "SimplifyHelper");

  auto axis_vec = axis.flat<Tperm>();
  for (int64_t i = 0; i < axis.NumElements(); ++i) {
    Tperm index = axis_vec(i);
    if (index < -data.dims() || index >= data.dims()) {
      return errors::InvalidArgument("Invalid reduction dimension (", index,
                                     " for input with ", data.dims(),
                                     " dimension(s)");
    }
    index = (index + data.dims()) % data.dims();
    if (bitmap[index]) {
      return errors::InvalidArgument(
          "Invalid reduction arguments: Axes contains duplicate dimension: ",
          index);
    }
    bitmap[index] = true;
  }
  return Status::OK();
}

Status ReductionHelper::Simplify(const Tensor& data, const Tensor& axis,
                                 const bool keep_dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreduction_ops_commonDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/kernels/reduction_ops_common.cc", "ReductionHelper::Simplify");

  // bitmap[i] indicates whether to reduce data along i-th axis.
  gtl::InlinedVector<bool, 4> bitmap(data.dims(), false);
  if (axis.dtype() == DT_INT32) {
    TF_RETURN_IF_ERROR(SimplifyHelper<int32>(data, axis, bitmap));
  } else {
    TF_RETURN_IF_ERROR(SimplifyHelper<int64_t>(data, axis, bitmap));
  }
  // Output tensor's dim sizes.
  out_shape_.clear();
  for (int i = 0; i < data.dims(); ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      out_shape_.push_back(data.dim_size(i));
    } else if (keep_dims) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      out_shape_.push_back(1);
    }
  }

  // Depending on bitmap[i] and bitmap[i-1], we can collapse axis of
  // the input data before doing the reduction on the resulting
  // tensor.  The shape of the reduction is a reshape of the final
  // output.

  // We'll skip the leading 1s.
  int dim_index = 0;
  for (; dim_index < data.dims(); ++dim_index) {
    if (data.dim_size(dim_index) != 1) break;
  }
  if (dim_index >= data.dims()) {
    // Special case. The input is essentially a scalar.
    reduce_first_axis_ = true;
  } else {
    // Starting from the (dim_index)-th dimension, dimensions
    // alternates between runs that need to be reduced and runs that
    // don't.
    //
    // NOTE: If a dimension has size 1, we group it as the current
    // run so that we can minimize the number of runs.
    //
    // E.g., when we want to reduce a tensor of shape [2, 1, 3, 1,
    // 5] by axes = [1, 4], we should treat the tensor as a [6, 5]
    // and reduce by axes = [1] (i.e., the output is shape [6]).
    reduce_first_axis_ = bitmap[dim_index];
    data_reshape_.push_back(data.dim_size(dim_index));
    ++dim_index;
    for (; dim_index < data.dims(); ++dim_index) {
      const auto size = data.dim_size(dim_index);
      if (size == 1) {
        bitmap[dim_index] = bitmap[dim_index - 1];
      }
      if (bitmap[dim_index - 1] != bitmap[dim_index]) {
        // Starts a new run of reduce or !reduce.
        data_reshape_.push_back(size);
      } else {
        // Continue a run of reduce or !reduce.
        data_reshape_.back() *= size;
      }
    }
    // If reduce_first_axis_ is true (input's dimension 0, 2, 4, etc
    // are reduced), data_reshape_[1, 3, 5, ...]  is out_reshape_,
    // otherwise, data_reshape_[0, 2, 4, ...] is.
    for (size_t i = reduce_first_axis_ ? 1 : 0; i < data_reshape_.size();
         i += 2) {
      out_reshape_.push_back(data_reshape_[i]);
    }
  }

  VLOG(1) << "data reshape: " << absl::StrJoin(data_reshape_, ",");
  VLOG(1) << "out  reshape: " << absl::StrJoin(out_reshape_, ",");
  VLOG(1) << "out    shape: " << absl::StrJoin(out_shape_, ",");
  return Status::OK();
}

}  // namespace tensorflow
