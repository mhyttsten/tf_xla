/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh() {
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


#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"
#include "tensorflow/lite/mutable_op_resolver.h"

// This file contains the TFLite adapter. That is, it takes a `OpKernelShim`
// class and provides a TFLite kernel out of it.

namespace tflite {
namespace shim {

// TfLite implementation of the methods during an op kernel initialization
class TfLiteInitContext : public InitContext<TfLiteInitContext> {
 public:
  TfLiteInitContext(const TfLiteContext* context,
                    const flexbuffers::Map* attr_map);
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;

 private:
  const flexbuffers::Map* attr_map_;
};

// TfLite implementation of the methods during an op kernel invocation
class TfLiteInvokeContext : public InvokeContext<TfLiteInvokeContext> {
 public:
  TfLiteInvokeContext(TfLiteContext* context_, TfLiteNode* node_);
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const;
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape& shape) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  absl::Status AssertShapesEqual(const TfLiteIntArray* dims,
                                 const Shape& output_shape) const;

  std::string ShapeMismatchErrorMsg(const TfLiteIntArray* actual_shape,
                                    const Shape& expected_shape) const;

  TfLiteContext* context_;
  TfLiteNode* node_;
};

// TfLite implementation of the methods during shape inference
class TfLiteShapeInferenceContext
    : public ShapeInferenceContext<TfLiteShapeInferenceContext> {
 public:
  TfLiteShapeInferenceContext(TfLiteContext* context, TfLiteNode* node,
                              const flexbuffers::Map* attr_map,
                              std::vector<Shape>* inferred_shapes);
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const;
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  TfLiteContext* context_;
  TfLiteNode* node_;
  const flexbuffers::Map* attr_map_;
  std::vector<Shape>* inferred_shapes_;
};

// Convert the absl::Status to a TfLiteStatus and report the error message.
TfLiteStatus StatusToTfLiteStatus(TfLiteContext* context,
                                  const absl::Status& status);

// Converts a vector of dims into an int array for TFLite use.
TfLiteIntArray* ShapeToTfLiteShape(const std::vector<int>& shape);

// Converts an int array representing shape in TFLite to Shape.
Shape TfLiteShapeToShape(const TfLiteIntArray* tflite_shape);

// An op kernel base class which is an adapter between an Op implementation
// (OpKernelShim subclass) and TFLite runtime
template <template <Runtime> typename Impl>
class TfLiteOpKernel {
 public:
  using ImplType = Impl<Runtime::kTfLite>;

  // Builds a TfLiteRegistration object to register this with the TfLite runtime
  static TfLiteRegistration* GetTfLiteRegistration() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_0(mht_0_v, 289, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "GetTfLiteRegistration");

    static TfLiteRegistration r =
        TfLiteRegistration{Init, Free, Prepare, Invoke};
    return &r;
  }

  // Adds this op kernel to the passed in op resolver
  static void Add(MutableOpResolver* resolver) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_1(mht_1_v, 299, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "Add");

    resolver->AddCustom(ImplType::kOpName, GetTfLiteRegistration());
  }

  // The operation name
  static const char* OpName() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_2(mht_2_v, 307, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "OpName");
 return ImplType::kOpName; }

 protected:
  // The data that is stored in node::user_data.
  struct UserData {
    UserData(const char* buffer, size_t length) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_3(mht_3_v, 316, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "UserData");

      impl = new ImplType;
      attr_map = new flexbuffers::Map(
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
              .AsMap());
    }

    // An instance of OpKernelShim<TF or TFLite>.
    // This is so that the Invoke(), Prepare(), etc. can call Invoke(),
    // ShapeInference(), ... on the kernel defined using this library.
    ImplType* impl = nullptr;
    // Attribute map for the op kernel.
    // The map needs to be accessible because the library provides
    // GetAttr() during ShapeInference() which is called during Prepare(). So
    // this needs to be accessible at that point.
    const flexbuffers::Map* attr_map = nullptr;

    ~UserData() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_4(mht_4_v, 336, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "~UserData");

      if (impl) delete impl;
      if (attr_map) delete attr_map;
    }
  };

  static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_5(mht_5_v, 346, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "Init");

    auto* user_data = new UserData(buffer, length);
    TfLiteInitContext ctx(context, user_data->attr_map);
    auto status = user_data->impl->Init(&ctx);
    StatusToTfLiteStatus(context, status);
    return user_data;
  }

  static void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_6(mht_6_v, 357, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "Free");

    if (buffer) delete static_cast<UserData*>(buffer);
  }

  // Resizes the Output Tensor to their shape. There are two cases:
  //
  // case 1: output shape is known after ShapeInference() was called during
  //     Prepare()
  //   ResizeTensor(output_of_shape_inference)
  // case 2: output shape is not fully defined even after shape inference
  //   SetTensorToDynamic(...)
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_7(mht_7_v, 371, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "Prepare");

    const size_t num_outputs = ::tflite::NumOutputs(node);
    std::vector<Shape> inferred_output_shapes(num_outputs);
    const auto* attr_map = static_cast<UserData*>(node->user_data)->attr_map;
    TfLiteShapeInferenceContext ctx(context, node, attr_map,
                                    &inferred_output_shapes);
    auto status = ImplType::ShapeInference(&ctx);
    TF_LITE_ENSURE_STATUS(StatusToTfLiteStatus(context, status));
    // Output shapes.
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
      TfLiteTensor* output_tensor =
          tflite::GetOutput(context, node, output_idx);
      TF_LITE_ENSURE(context, output_tensor != nullptr);
      if (inferred_output_shapes[output_idx].FullyDefined()) {
        // Case: output shape can be inferred during `Prepare`
        TF_LITE_ENSURE_OK(context,
                          context->ResizeTensor(
                              context, output_tensor,
                              ShapeToTfLiteShape(
                                  inferred_output_shapes[output_idx].value())));
      } else {
        // Case: output shape is dynamic
        tflite::SetTensorToDynamic(output_tensor);
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTh mht_8(mht_8_v, 402, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.h", "Invoke");

    TfLiteInvokeContext ctx(context, node);
    return StatusToTfLiteStatus(
        context, static_cast<UserData*>(node->user_data)->impl->Invoke(&ctx));
  }
};

template <>
struct ContextTypeForRuntime<Runtime::kTfLite> {
  using Init = TfLiteInitContext;
  using Invoke = TfLiteInvokeContext;
  using ShapeInference = TfLiteShapeInferenceContext;
};

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_
