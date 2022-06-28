#include "tensorflow/core/framework/tensor_key.h"
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_VARIANT_H_
#define TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_VARIANT_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh() {
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


#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {

// Class used to store a RaggedTensor as a Variant scalar.
class RaggedTensorVariant {
 public:
  RaggedTensorVariant() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "RaggedTensorVariant");
}
  RaggedTensorVariant(Tensor values, const std::vector<Tensor>& nested_splits)
      : values_(std::move(values)), nested_splits_(nested_splits) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "RaggedTensorVariant");
}

  // Variant support methods.
  string TypeName() const;
  string DebugString() const;
  void Encode(VariantTensorData* data) const;
  bool Decode(const VariantTensorData& data);

  // The flat_values of the RaggedTensor.
  const Tensor& values() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_2(mht_2_v, 225, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "values");
 return values_; }
  Tensor* mutable_values() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_3(mht_3_v, 229, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "mutable_values");
 return &values_; }
  void set_values(const Tensor& new_values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_4(mht_4_v, 233, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "set_values");
 values_ = new_values; }

  // The nested row_splits of the RaggedTensor.
  int ragged_rank() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_5(mht_5_v, 239, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "ragged_rank");
 return nested_splits_.size(); }
  const std::vector<Tensor>& nested_splits() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_6(mht_6_v, 243, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "nested_splits");
 return nested_splits_; }
  std::vector<Tensor>* mutable_nested_splits() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_7(mht_7_v, 247, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "mutable_nested_splits");
 return &nested_splits_; }
  const Tensor& splits(int i) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_8(mht_8_v, 251, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "splits");
 return nested_splits_[i]; }
  Tensor* mutable_splits(int i) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_9(mht_9_v, 255, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "mutable_splits");
 return &nested_splits_[i]; }
  void set_nested_splits(const std::vector<Tensor>& nested_splits) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_10(mht_10_v, 259, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "set_nested_splits");

    nested_splits_ = nested_splits;
  }
  void append_splits(const Tensor& splits) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_11(mht_11_v, 265, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "append_splits");
 nested_splits_.push_back(splits); }

 private:
  Tensor values_;
  std::vector<Tensor> nested_splits_;
};

template <typename Device>
Status RaggedTensorVariantZerosLike(OpKernelContext* c,
                                    const RaggedTensorVariant& x,
                                    RaggedTensorVariant* y) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_12(mht_12_v, 278, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "RaggedTensorVariantZerosLike");

  y->set_nested_splits(x.nested_splits());
  TF_RETURN_IF_ERROR(
      ZerosLikeTensor<Device>(c, x.values(), y->mutable_values()));
  return Status::OK();
}

template <typename Device>
Status RaggedTensorVariantBinaryAdd(OpKernelContext* c,
                                    const RaggedTensorVariant& x,
                                    const RaggedTensorVariant& y,
                                    RaggedTensorVariant* out) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_variantDTh mht_13(mht_13_v, 292, "", "./tensorflow/core/kernels/ragged_tensor_variant.h", "RaggedTensorVariantBinaryAdd");

  if (x.values().dtype() != y.values().dtype()) {
    return errors::InvalidArgument(
        "Can't add RaggedTensorVariants of different dtypes. One is ",
        DataTypeString(x.values().dtype()), " and the other is ",
        DataTypeString(y.values().dtype()));
  }
  if (x.ragged_rank() != y.ragged_rank()) {
    return errors::InvalidArgument(
        "Can't add RaggedTensorVariants of different ragged rank. ", "One is ",
        x.ragged_rank(), " and the other is ", y.ragged_rank());
  }
  for (int i = 0; i < x.ragged_rank(); ++i) {
    if (TensorKey(x.splits(i)) != TensorKey(y.splits(i))) {
      return errors::InvalidArgument(
          "Can't add RaggedTensorVariants with different row_splits.");
    }
  }
  out->set_nested_splits(x.nested_splits());
  TF_RETURN_IF_ERROR(BinaryAddTensors<Device>(c, x.values(), y.values(),
                                              out->mutable_values()));
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_VARIANT_H_
