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
class MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc() {
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

// See docs in ../ops/data_flow_ops.cc.

#include <vector>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
class DynamicPartitionOp_Shared : public OpKernel {
 public:
  explicit DynamicPartitionOp_Shared(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/dynamic_partition_op.cc", "DynamicPartitionOp_Shared");

    OP_REQUIRES_OK(c, c->GetAttr("num_partitions", &num_partitions_));
    //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8, etc.
    //   to input[1].  Should we have the framework do some sort of
    //   integer promotion automatically, or should that be something
    //   that users have to do explicitly with a conversion operator
    //   in the graph?
  }

  void ValidateAndAllocateOutputs(OpKernelContext* c, const Tensor** data,
                                  const Tensor** partitions,
                                  OpOutputList* Tout) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/dynamic_partition_op.cc", "ValidateAndAllocateOutputs");

    OP_REQUIRES_OK(c, c->input("data", data));
    OP_REQUIRES_OK(c, c->input("partitions", partitions));
    OP_REQUIRES(
        c,
        TensorShapeUtils::StartsWith((*data)->shape(), (*partitions)->shape()),
        errors::InvalidArgument(
            "data.shape must start with partitions.shape, ",
            "got data.shape = ", (*data)->shape().DebugString(),
            ", partitions.shape = ", (*partitions)->shape().DebugString()));

    // Count how many occurrences of each partition id we have in partitions
    gtl::InlinedVector<int, 32> partition_count(num_partitions_);
    auto e_partitions = (*partitions)->flat<int32>();
    const int64_t N = e_partitions.dimension(0);
    for (int64_t i = 0; i < N; i++) {
      const int32_t p = internal::SubtleMustCopy(e_partitions(i));
      OP_REQUIRES(c, FastBoundsCheck(p, num_partitions_),
                  errors::InvalidArgument(
                      "partitions", SliceDebugString((*partitions)->shape(), i),
                      " = ", p, " is not in [0, ", num_partitions_, ")"));
      partition_count[p]++;
    }

    // Allocate output tensors of the right size
    OP_REQUIRES_OK(c, c->output_list("outputs", Tout));
    for (int p = 0; p < num_partitions_; p++) {
      TensorShape shape;
      shape.AddDim(partition_count[p]);
      for (int i = (*partitions)->dims(); i < (*data)->dims(); i++) {
        shape.AddDim((*data)->dim_size(i));
      }
      Tensor* out;
      OP_REQUIRES_OK(c, Tout->allocate(p, shape, &out));
    }
  }

 protected:
  int num_partitions_;
};

template <class T>
class DynamicPartitionOp : public DynamicPartitionOp_Shared {
 public:
  explicit DynamicPartitionOp(OpKernelConstruction* c)
      : DynamicPartitionOp_Shared(c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/kernels/dynamic_partition_op.cc", "DynamicPartitionOp");
}
  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdynamic_partition_opDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/dynamic_partition_op.cc", "Compute");

    const Tensor* data;
    const Tensor* partitions;
    OpOutputList outputs;
    ValidateAndAllocateOutputs(c, &data, &partitions, &outputs);
    if (!c->status().ok()) return;
    if (num_partitions_ == 0 || data->NumElements() == 0) return;

    auto e_partitions = partitions->flat<int32>();
    const int64_t N = e_partitions.dimension(0);
    gtl::InlinedVector<int, 32> output_index(num_partitions_);

    if (partitions->dims() == data->dims()) {
      // Walk through data and copy the data to the appropriate output tensor
      const auto data_flat = data->flat<T>();
      std::vector<Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                                   Eigen::Aligned> >
          out_vec;
      out_vec.reserve(num_partitions_);
      for (int p = 0; p < num_partitions_; p++) {
        out_vec.push_back(outputs[p]->vec<T>());
      }
      for (int64_t i = 0; i < N; i++) {
        const int32_t p = internal::SubtleMustCopy(e_partitions(i));
        OP_REQUIRES(
            c, FastBoundsCheck(p, num_partitions_),
            errors::InvalidArgument("indices[", i, "] is out of range"));
        auto oi = output_index[p];
        OP_REQUIRES(c, FastBoundsCheck(oi, out_vec[p].size()),
                    errors::InvalidArgument(
                        "out_vec[", p, "] size: ", out_vec[p].size(),
                        " is not LTE output_index[", p, "] : ", oi));
        out_vec[p](oi) = data_flat(i);
        output_index[p]++;
      }
    } else {
      // If data has extra dimensions, use Eigen slices
      std::vector<Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                                   Eigen::Aligned> >
          out_flat;
      out_flat.reserve(num_partitions_);
      for (int p = 0; p < num_partitions_; p++) {
        out_flat.push_back(outputs[p]->flat_outer_dims<T>());
      }

      // Walk through data and copy the data to the appropriate output tensor
      const int64_t slice_size = data->NumElements() / N;
      const auto data_flat = data->shaped<T, 2>({N, slice_size});
      Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
      for (int64_t i = 0; i < N; i++) {
        // outputs[p][output_index[p]++] = data[i]
        const int32_t p = internal::SubtleMustCopy(e_partitions(i));
        OP_REQUIRES(
            c, FastBoundsCheck(p, num_partitions_),
            errors::InvalidArgument("indices[", i,
                                    "] has been asynchronously overwritten and "
                                    "is no longer in range!"));
        auto oi = output_index[p];
        OP_REQUIRES(c, FastBoundsCheck(oi, out_flat[p].dimension(0)),
                    errors::InvalidArgument("Size of output_index: ", oi,
                                            " is no longer in range."));
        Eigen::DSizes<Eigen::DenseIndex, 2> out_indices(oi, 0);
        Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
        out_flat[p].slice(out_indices, sizes) =
            data_flat.slice(data_indices, sizes);
        output_index[p]++;
      }
    }
  }
};

#define REGISTER_DYNAMIC_PARTITION(T)                                     \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DynamicPartition").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DynamicPartitionOp<T>)

TF_CALL_ALL_TYPES(REGISTER_DYNAMIC_PARTITION);
#undef REGISTER_DYNAMIC_PARTITION

}  // namespace tensorflow
