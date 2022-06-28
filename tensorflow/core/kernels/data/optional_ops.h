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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh() {
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


#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {
namespace data {

const char kOptionalVariantTypeName[] = "tensorflow::data::Optional";

// Stores a DT_VARIANT value representing an Optional with the given value
// in the `output_index`^th output of the given kernel execution context.
Status WriteOptionalWithValueToOutput(OpKernelContext* ctx, int output_index,
                                      std::vector<Tensor> value);

// Stores a DT_VARIANT value representing an Optional with no value
// in the `output_index`^th output of the given kernel execution context.
Status WriteOptionalNoneToOutput(OpKernelContext* ctx, int output_index);

// An `OptionalVariant` can represent either an "actual value" (a tuple of
// tensors) or "none", and may be stored in a DT_VARIANT tensor.
class OptionalVariant {
 public:
  // Create an `OptionalVariant` with no actual value.
  OptionalVariant() : values_(nullptr) {}

  // Create an `OptionalVariant` with the actual value given by the tuple of
  // tensors in `values`.
  explicit OptionalVariant(std::vector<Tensor> values) {
    values_ = std::make_shared<std::vector<Tensor>>(std::move(values));
  }

  OptionalVariant(const OptionalVariant& other) : values_(other.values_) {}

  // Returns true if `this` represents an actual value.
  bool has_value() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_0(mht_0_v, 223, "", "./tensorflow/core/kernels/data/optional_ops.h", "has_value");
 return values_ != nullptr; }

  // REQUIRES: `this->has_value()` must be true.
  const std::vector<Tensor>& get_values() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_1(mht_1_v, 229, "", "./tensorflow/core/kernels/data/optional_ops.h", "get_values");

    DCHECK(values_) << "Tried to get values from an empty OptionalVariant";
    return *values_;
  }

  // Implementations of the necessary methods for using `OptionalVariant`
  // objects in DT_VARIANT tensors.
  string TypeName() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_2(mht_2_v, 239, "", "./tensorflow/core/kernels/data/optional_ops.h", "TypeName");
 return kOptionalVariantTypeName; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_3(mht_3_v, 243, "", "./tensorflow/core/kernels/data/optional_ops.h", "Encode");

    data->set_metadata(values_ != nullptr);
    if (values_ != nullptr) {
      for (const auto& t : *values_) {
        *(data->add_tensors()) = t;
      }
    }
  }

  bool Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_4(mht_4_v, 255, "", "./tensorflow/core/kernels/data/optional_ops.h", "Decode");

    if (data.type_name() != TypeName()) {
      return false;
    }
    bool has_value = false;
    if (!data.get_metadata(&has_value)) {
      return false;
    }
    if (has_value) {
      values_ = std::make_shared<std::vector<Tensor>>(data.tensors());
    } else {
      values_.reset();
    }
    return true;
  }

  string DebugString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_5(mht_5_v, 274, "", "./tensorflow/core/kernels/data/optional_ops.h", "DebugString");

    if (values_) {
      return strings::StrCat("OptionalVariant<", "values: (",
                             absl::StrJoin(*values_, ", ",
                                           [](string* s, const Tensor& elem) {
                                             *s = elem.DebugString();
                                           }),
                             ")>");
    } else {
      return strings::StrCat("OptionalVariant<None>");
    }
  }

 private:
  std::shared_ptr<const std::vector<Tensor>> values_;
};

template <typename Device>
Status OptionalZerosLike(OpKernelContext* ctx, const OptionalVariant& x,
                         OptionalVariant* y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_6(mht_6_v, 296, "", "./tensorflow/core/kernels/data/optional_ops.h", "OptionalZerosLike");

  if (!x.has_value()) {
    *y = x;
    return Status::OK();
  }
  std::vector<Tensor> zero_tensors;
  for (const Tensor& tensor : x.get_values()) {
    Tensor zero_t;
    TF_RETURN_IF_ERROR(ZerosLikeTensor<Device>(ctx, tensor, &zero_t));
    zero_tensors.push_back(std::move(zero_t));
  }
  *y = OptionalVariant(zero_tensors);
  return Status::OK();
}

template <typename Device>
Status OptionalBinaryAdd(OpKernelContext* ctx, const OptionalVariant& a,
                         const OptionalVariant& b, OptionalVariant* out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSoptional_opsDTh mht_7(mht_7_v, 316, "", "./tensorflow/core/kernels/data/optional_ops.h", "OptionalBinaryAdd");

  // TODO(skyewm): should adding a value to a non-value be a no-op instead?
  if (a.has_value() != b.has_value()) {
    return errors::InvalidArgument(
        "Cannot add optionals because one has a value and the other doesn't.");
  }
  if (!a.has_value()) {
    *out = a;
    return Status::OK();
  }
  if (a.get_values().size() != b.get_values().size()) {
    return errors::InvalidArgument(
        "Cannot add optionals because they have different numbers of "
        "components (",
        a.get_values().size(), " vs. ", b.get_values().size(), ").");
  }
  std::vector<Tensor> out_tensors;
  for (int i = 0; i < a.get_values().size(); ++i) {
    const Tensor& a_tensor = a.get_values()[i];
    const Tensor& b_tensor = b.get_values()[i];
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(
        BinaryAddTensors<Device>(ctx, a_tensor, b_tensor, &out_tensor));
    out_tensors.push_back(std::move(out_tensor));
  }
  *out = OptionalVariant(out_tensors);
  return Status::OK();
}

class OptionalNoneOp : public OpKernel {
 public:
  explicit OptionalNoneOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class OptionalFromValueOp : public OpKernel {
 public:
  explicit OptionalFromValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class OptionalHasValueOp : public OpKernel {
 public:
  explicit OptionalHasValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

class OptionalGetValueOp : public OpKernel {
 public:
  explicit OptionalGetValueOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES(
        ctx, output_shapes_.size() == output_types_.size(),
        errors::InvalidArgument(
            "output_types and output_shapes must be same length, got:\n",
            "output_types: ", output_types_.size(), "\n",
            "output_shapes: ", output_shapes_.size()));
  }

  void Compute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_OPTIONAL_OPS_H_
