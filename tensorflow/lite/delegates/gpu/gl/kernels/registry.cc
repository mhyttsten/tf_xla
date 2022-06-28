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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/registry.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/add.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/concat.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/conv.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/custom_registry.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/lstm.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/mean.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/mul.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/pad.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/pooling.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/prelu.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/quantize_and_dequantize.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/relu.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/resampler.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/reshape.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/resize.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/slice.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/softmax.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/space_to_depth.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/tile.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/transpose_conv.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "tensorflow/lite/delegates/gpu/gl/kernels/max_unpooling.h"
#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Registry : public NodeShader {
 public:
  Registry() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc mht_0(mht_0_v, 234, "", "./tensorflow/lite/delegates/gpu/gl/kernels/registry.cc", "Registry");

    using Type = OperationType;
    using NewShaderFunc = std::function<std::unique_ptr<NodeShader>()>;

    const auto insert_op = [&](Type type, NewShaderFunc func) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/delegates/gpu/gl/kernels/registry.cc", "lambda");

      shaders_[ToString(type)].push_back(func());
    };
    const auto insert_elementwise_op = [&](Type operation_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/delegates/gpu/gl/kernels/registry.cc", "lambda");

      shaders_[ToString(operation_type)].push_back(
          NewElementwiseNodeShader(operation_type));
    };

    insert_op(Type::ADD, NewAddNodeShader);
    insert_op(Type::CONCAT, NewAlignedConcatNodeShader);
    insert_op(Type::CONCAT, NewFlatConcatNodeShader);
    insert_op(Type::CONCAT, NewConcatNodeShader);
    insert_op(Type::CONVOLUTION_2D, NewConvolution1x1NodeShader);
    insert_op(Type::CONVOLUTION_2D, NewConvolutionNodeShader);
    insert_op(Type::CONVOLUTION_TRANSPOSED, NewConvolutionTransposedNodeShader);
    insert_op(Type::DEPTHWISE_CONVOLUTION, NewDepthwiseConvolutionNodeShader);
    insert_op(Type::DEPTH_TO_SPACE, NewDepthToSpaceNodeShader);
    insert_op(Type::FULLY_CONNECTED, NewFullyConnectedNodeShader);
    insert_op(Type::LSTM, NewLstmNodeShader);
    insert_op(Type::MEAN, NewMeanNodeShader);
    // TODO(b/162763635): implement MeanStddevNormalization for OpenGL.
    insert_op(Type::MUL, NewMultiplyNodeShader);
    insert_op(Type::PAD, NewPadNodeShader);
    insert_op(Type::POOLING_2D, NewPoolingNodeShader);
    insert_op(Type::PRELU, NewPReLUNodeShader);
    insert_op(Type::QUANTIZE_AND_DEQUANTIZE,
              NewQuantizeAndDequantizeNodeShader);
    insert_op(Type::RELU, NewReLUNodeShader);
    insert_op(Type::RESAMPLER, NewResamplerNodeShader);
    insert_op(Type::RESIZE, NewResizeNodeShader);
    insert_op(Type::RESHAPE, NewReshapeNodeShader);
    insert_op(Type::SLICE, NewSliceNodeShader);
    insert_op(Type::SOFTMAX, NewSoftmaxNodeShader);
    insert_op(Type::SPACE_TO_DEPTH, NewSpaceToDepthNodeShader);
    insert_op(Type::TILE, NewTileNodeShader);

    insert_elementwise_op(Type::ABS);
    insert_elementwise_op(Type::COPY);
    insert_elementwise_op(Type::COS);
    insert_elementwise_op(Type::DIV);
    insert_elementwise_op(Type::ELU);
    insert_elementwise_op(Type::EXP);
    insert_elementwise_op(Type::FLOOR);
    insert_elementwise_op(Type::FLOOR_DIV);
    insert_elementwise_op(Type::FLOOR_MOD);
    insert_elementwise_op(Type::HARD_SWISH);
    insert_elementwise_op(Type::LOG);
    insert_elementwise_op(Type::NEG);
    insert_elementwise_op(Type::MAXIMUM);
    insert_elementwise_op(Type::MINIMUM);
    insert_elementwise_op(Type::POW);
    insert_elementwise_op(Type::RSQRT);
    insert_elementwise_op(Type::SIGMOID);
    insert_elementwise_op(Type::SIN);
    insert_elementwise_op(Type::SQRT);
    insert_elementwise_op(Type::SQUARE);
    insert_elementwise_op(Type::SQUARED_DIFF);
    insert_elementwise_op(Type::SUB);
    insert_elementwise_op(Type::TANH);

#ifndef TFLITE_GPU_BINARY_RELEASE
    insert_op(Type::MAX_UNPOOLING_2D, NewMaxUnpoolingNodeShader);
    RegisterCustomOps(&shaders_);
#endif  // TFLITE_GPU_BINARY_RELEASE
  }

  ~Registry() final = default;

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSregistryDTcc mht_3(mht_3_v, 316, "", "./tensorflow/lite/delegates/gpu/gl/kernels/registry.cc", "GenerateCode");

    auto it = shaders_.find(ctx.op_type);
    if (it == shaders_.end()) {
      return absl::NotFoundError(
          absl::StrCat("No shader implementation for ", ctx.op_type));
    }
    std::vector<std::string> errors;
    for (const auto& shader : it->second) {
      const auto status = shader->GenerateCode(ctx, generated_code);
      // Return the first suitable shader.
      if (status.ok()) return absl::OkStatus();
      errors.push_back(std::string(status.message()));
    }
    return errors.empty() ? absl::OkStatus()
                          : absl::UnknownError(absl::StrJoin(errors, ", "));
  }

 private:
  absl::flat_hash_map<std::string, std::vector<std::unique_ptr<NodeShader>>>
      shaders_;
};

}  // namespace

std::unique_ptr<NodeShader> NewNodeShaderRegistry() {
  return absl::make_unique<Registry>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
