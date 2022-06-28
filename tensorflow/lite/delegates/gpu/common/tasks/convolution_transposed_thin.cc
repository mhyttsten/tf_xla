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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConvolutionTransposedThin::ConvolutionTransposedThin(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr,
    const GpuInfo& gpu_info)
    : GPUOperation(definition) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "ConvolutionTransposedThin::ConvolutionTransposedThin");

  code_ = GenerateConvolutionTransposedCode(
      definition_, DivideRoundUp(attr.weights.shape.i, 4), attr.weights.shape.o,
      int2(attr.weights.shape.w, attr.weights.shape.h));
  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    compiler_options_.push_back(CompilerOptions::kAdrenoFullSimd);
  }
}

ConvolutionTransposedThin::ConvolutionTransposedThin(
    ConvolutionTransposedThin&& operation)
    : GPUOperation(std::move(operation)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "ConvolutionTransposedThin::ConvolutionTransposedThin");
}

ConvolutionTransposedThin& ConvolutionTransposedThin::operator=(
    ConvolutionTransposedThin&& operation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_2(mht_2_v, 220, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "=");

  if (this != &operation) {
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ConvolutionTransposedThin::GenerateConvolutionTransposedCode(
    const OperationDef& op_def, int src_depth, int dst_channels,
    const int2& kernel_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_3(mht_3_v, 232, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "ConvolutionTransposedThin::GenerateConvolutionTransposedCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  const std::string channel_x = dst_channels == 1 ? "" : ".x";
  const std::vector<std::string> postfix = {channel_x, ".y", ".z", ".w"};
  const std::vector<std::string> channel = {".x", ".y", ".z", ".w"};

  const std::string type_postfix =
      dst_channels == 1 ? "" : std::to_string(dst_channels);

  std::string accum_type;

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F32_F16:
      accum_type = "float" + type_postfix;
      break;
    case CalculationsPrecision::F16:
      accum_type = "half" + type_postfix;
      break;
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  if (X >= args.src_tensor.Width() || Y >= args.src_tensor.Height()) "
       "return;\n";
  c += "  " + accum_type + " r[" + std::to_string(kernel_size.y) + "][" +
       std::to_string(kernel_size.x) + "];\n";
  c += "  {\n";
  c += "  FLT4 src = args.src_tensor.Read(X, Y, 0);\n";
  int index = 0;
  for (int y = 0; y < kernel_size.y; ++y) {
    for (int x = 0; x < kernel_size.x; ++x) {
      std::string r_s =
          "  r[" + std::to_string(y) + "][" + std::to_string(x) + "]";
      for (int d = 0; d < dst_channels; ++d) {
        c += r_s + postfix[d] + " = dot(src, args.weights.Read(" +
             std::to_string(index) + "));\n";
        index++;
      }
    }
  }
  c += "  }\n";
  for (int i = 1; i < src_depth; ++i) {
    c += "  if (X > " + std::to_string(-i) +
         ") {  // always true, to reduce registers usage\n";
    c +=
        "  FLT4 src = args.src_tensor.Read(X, Y, " + std::to_string(i) + ");\n";
    for (int y = 0; y < kernel_size.y; ++y) {
      for (int x = 0; x < kernel_size.x; ++x) {
        std::string r_s =
            "  r[" + std::to_string(y) + "][" + std::to_string(x) + "]";
        for (int d = 0; d < dst_channels; ++d) {
          c += r_s + postfix[d] + " += dot(src, args.weights.Read(" +
               std::to_string(index) + "));\n";
          index++;
        }
      }
    }
    c += "  }\n";
  }
  c += "  X *= " + std::to_string(kernel_size.x) + ";\n";
  c += "  Y *= " + std::to_string(kernel_size.y) + ";\n";
  for (int y = 0; y < kernel_size.y; ++y) {
    for (int x = 0; x < kernel_size.x; ++x) {
      const std::string x_coord = "X + " + std::to_string(x);
      const std::string y_coord = "Y + " + std::to_string(y);
      c += "  if (" + x_coord + " < args.dst_tensor.Width() && " + y_coord +
           " < args.dst_tensor.Height()) {\n";
      c += "    FLT4 result = args.weights.Read(" + std::to_string(index) +
           ");\n";
      for (int d = 0; d < dst_channels; ++d) {
        c += "    result" + channel[d] + " += r[" + std::to_string(y) + "][" +
             std::to_string(x) + "]" + postfix[d] + ";\n";
      }
      c += "    args.dst_tensor.Write(result, " + x_coord + ", " + y_coord +
           ", 0);\n";
      c += "  }\n";
    }
  }
  c += "}\n";

  return c;
}

int3 ConvolutionTransposedThin::GetGridSize() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_4(mht_4_v, 331, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "ConvolutionTransposedThin::GetGridSize");

  const int grid_x = src_[0]->Width() * dst_[0]->Batch();
  const int grid_y = src_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

bool IsConvolutionTransposedThinSupported(
    const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_5(mht_5_v, 342, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "IsConvolutionTransposedThinSupported");

  return attr.weights.shape.o <= 4 && attr.weights.shape.w == attr.stride.w &&
         attr.weights.shape.h == attr.stride.h &&
         attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0 &&
         attr.padding.appended.w == 0 && attr.padding.appended.h == 0;
}

ConvolutionTransposedThin CreateConvolutionTransposedThin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconvolution_transposed_thinDTcc mht_6(mht_6_v, 354, "", "./tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_thin.cc", "CreateConvolutionTransposedThin");

  ConvolutionTransposedThin result(definition, attr, gpu_info);
  result.UploadData(attr.weights, attr.bias);
  return result;
}

}  // namespace gpu
}  // namespace tflite
