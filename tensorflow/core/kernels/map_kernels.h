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
#ifndef TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh() {
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


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/tensor_map.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {

Status GetInputMap(OpKernelContext* ctx, int index, const TensorMap** ret_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_0(mht_0_v, 194, "", "./tensorflow/core/kernels/map_kernels.h", "GetInputMap");

  if (!TensorShapeUtils::IsScalar(ctx->input(index).shape())) {
    return errors::InvalidArgument("Input map must be a scalar. Saw: ",
                                   ctx->input(index).shape().DebugString());
  }
  const TensorMap* map = ctx->input(index).scalar<Variant>()().get<TensorMap>();
  if (map == nullptr) {
    return errors::InvalidArgument(
        "Input handle is not a map. Saw: '",
        ctx->input(index).scalar<Variant>()().DebugString(), "'");
  }
  *ret_map = map;
  return Status::OK();
}

// TODO(kattian): change into templated function
Status ForwardInputOrCreateNewMap(OpKernelContext* ctx, int32_t input_index,
                                  int32_t output_index,
                                  const TensorMap& input_map,
                                  TensorMap** output_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/map_kernels.h", "ForwardInputOrCreateNewMap");

  // Attempt to forward the input tensor to the output if possible.
  std::unique_ptr<Tensor> maybe_output = ctx->forward_input(
      input_index, output_index, DT_VARIANT, TensorShape{},
      ctx->input_memory_type(input_index), AllocatorAttributes());
  Tensor* output_tensor;
  if (maybe_output != nullptr && maybe_output->dtype() == DT_VARIANT &&
      maybe_output->NumElements() == 1) {
    output_tensor = maybe_output.get();
    TensorMap* tmp_out = output_tensor->scalar<Variant>()().get<TensorMap>();
    if (tmp_out == nullptr) {
      return errors::InvalidArgument(
          "Expected input ", input_index, " to be a TensorMap but saw ",
          output_tensor->scalar<Variant>()().TypeName());
    }
    if (tmp_out->RefCountIsOne()) {
      // Woohoo, forwarding succeeded!
      ctx->set_output(output_index, *output_tensor);
      *output_map = tmp_out;
      return Status::OK();
    }
  }

  // If forwarding is not possible allocate a new output tensor and copy
  // the `input_map` to it.
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(
      ctx->allocate_output(output_index, {}, &output_tensor, attr));
  output_tensor->scalar<Variant>()() = input_map.Copy();

  *output_map = output_tensor->scalar<Variant>()().get<TensorMap>();
  return Status::OK();
}

class EmptyTensorMap : public OpKernel {
 public:
  explicit EmptyTensorMap(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_2(mht_2_v, 256, "", "./tensorflow/core/kernels/map_kernels.h", "EmptyTensorMap");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result, attr));
    TensorMap empty;
    result->scalar<Variant>()() = std::move(empty);
  }
};

class TensorMapSize : public OpKernel {
 public:
  explicit TensorMapSize(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_4(mht_4_v, 276, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapSize");
}
  ~TensorMapSize() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_5(mht_5_v, 280, "", "./tensorflow/core/kernels/map_kernels.h", "~TensorMapSize");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_6(mht_6_v, 285, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));
    result->scalar<int32>()() = map->tensors().size();
  }
};

class TensorMapLookup : public OpKernel {
 public:
  explicit TensorMapLookup(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_7(mht_7_v, 299, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapLookup");
}
  ~TensorMapLookup() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_8(mht_8_v, 303, "", "./tensorflow/core/kernels/map_kernels.h", "~TensorMapLookup");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_9(mht_9_v, 308, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    const TensorKey& key = ctx->input(1);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    OP_REQUIRES(
        ctx, map->tensors().find(key) != map->tensors().end(),
        errors::InvalidArgument("Trying to lookup non-existent key. Could not "
                                "find key \"" +
                                key.SummarizeValue(100) + "\"."));

    ctx->set_output(0, map->tensors().find(key)->second);
  }
};

class TensorMapInsert : public OpKernel {
 public:
  explicit TensorMapInsert(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_10(mht_10_v, 328, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapInsert");
}
  ~TensorMapInsert() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_11(mht_11_v, 332, "", "./tensorflow/core/kernels/map_kernels.h", "~TensorMapInsert");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_12(mht_12_v, 337, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    const TensorKey& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(ctx,
                   ForwardInputOrCreateNewMap(ctx, 0, 0, *map, &output_map));
    output_map->replace(key, value);
  }
};

class TensorMapErase : public OpKernel {
 public:
  explicit TensorMapErase(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_13(mht_13_v, 355, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapErase");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_14(mht_14_v, 360, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    const TensorKey& key = ctx->input(1);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    OP_REQUIRES(
        ctx, map->tensors().find(key) != map->tensors().end(),
        errors::InvalidArgument("Trying to erase non-existent item. Could not "
                                "find key \"" +
                                key.SummarizeValue(100) + "\"."));

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(ctx,
                   ForwardInputOrCreateNewMap(ctx, 0, 0, *map, &output_map));
    output_map->tensors().erase(key);
  }
};

class TensorMapHasKey : public OpKernel {
 public:
  explicit TensorMapHasKey(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_15(mht_15_v, 383, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapHasKey");
}
  ~TensorMapHasKey() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_16(mht_16_v, 387, "", "./tensorflow/core/kernels/map_kernels.h", "~TensorMapHasKey");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_17(mht_17_v, 392, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    const TensorKey& key = ctx->input(1);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));
    result->scalar<bool>()() = map->tensors().find(key) != map->tensors().end();
  }
};

class TensorMapStackKeys : public OpKernel {
 public:
  explicit TensorMapStackKeys(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_18(mht_18_v, 407, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapStackKeys");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_dtype", &key_dtype_));
  }
  ~TensorMapStackKeys() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_19(mht_19_v, 413, "", "./tensorflow/core/kernels/map_kernels.h", "~TensorMapStackKeys");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_20(mht_20_v, 418, "", "./tensorflow/core/kernels/map_kernels.h", "Compute");

    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    OP_REQUIRES(ctx, map->size() != 0,
                errors::InvalidArgument(
                    "TensorMapStackKeys cannot be called on empty map."));

    auto it = map->tensors().begin();
    TensorShape output_shape = it->first.shape();
    output_shape.InsertDim(0, map->tensors().size());
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &result));

    int i = 0;
    size_t sz = map->tensors().size();
    TensorShape key_shape = it->first.shape();
    while (it != map->tensors().end() && i < sz) {
      OP_REQUIRES(
          ctx, it->first.dtype() == key_dtype_,
          errors::InvalidArgument("Key does not match requested dtype."));
      OP_REQUIRES(
          ctx, it->first.shape() == key_shape,
          errors::InvalidArgument("Keys must all have the same shape."));
      OP_REQUIRES_OK(ctx, batch_util::CopyElementToSlice(it->first, result, i));
      i++;
      it++;
    }
  }

 private:
  DataType key_dtype_;
};

template <typename Device>
Status TensorMapBinaryAdd(OpKernelContext* ctx, const TensorMap& a,
                          const TensorMap& b, TensorMap* out) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_21(mht_21_v, 457, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapBinaryAdd");

  // Binary add returns a map containing the union of keys.
  // Values with keys in the intersection are added.
  out->tensors() = a.tensors();
  for (const std::pair<TensorKey, Tensor>& p : b.tensors()) {
    absl::flat_hash_map<TensorKey, Tensor>::iterator it =
        out->tensors().find(p.first);
    if (it != out->tensors().end()) {
      Tensor out_tensor;
      TF_RETURN_IF_ERROR(
          BinaryAddTensors<Device>(ctx, p.second, it->second, &out_tensor));
      it->second = out_tensor;
    } else {
      out->tensors().emplace(p.first, p.second);
    }
  }
  return Status::OK();
}

template <typename Device>
Status TensorMapZerosLike(OpKernelContext* ctx, const TensorMap& x,
                          TensorMap* y) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmap_kernelsDTh mht_22(mht_22_v, 481, "", "./tensorflow/core/kernels/map_kernels.h", "TensorMapZerosLike");

  // Zeros like returns an empty map.
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
