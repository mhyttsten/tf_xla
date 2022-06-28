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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.h"

#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

DepthwiseConv3x3::DepthwiseConv3x3(const OperationDef& definition,
                                   bool weights_are_buffer,
                                   bool local_mem_uploads,
                                   const GpuInfo& gpu_info)
    : GPUOperation(definition), local_mem_uploads_(local_mem_uploads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "DepthwiseConv3x3::DepthwiseConv3x3");

  work_group_size_ = int3(8, 4, 1);
  code_ = GenerateDepthwiseConvCode(gpu_info, definition_, weights_are_buffer,
                                    local_mem_uploads_);

  if (definition_.precision == CalculationsPrecision::F16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

DepthwiseConv3x3::DepthwiseConv3x3(DepthwiseConv3x3&& operation)
    : GPUOperation(std::move(operation)),
      local_mem_uploads_(operation.local_mem_uploads_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "DepthwiseConv3x3::DepthwiseConv3x3");
}

DepthwiseConv3x3& DepthwiseConv3x3::operator=(DepthwiseConv3x3&& operation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "=");

  if (this != &operation) {
    std::swap(local_mem_uploads_, operation.local_mem_uploads_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string DepthwiseConv3x3::GenerateDepthwiseConvCode(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    bool weights_are_buffer, bool local_mem_uploads) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_3(mht_3_v, 235, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "DepthwiseConv3x3::GenerateDepthwiseConvCode");

  auto src_desc = op_def.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  AddSrcTensor("src_tensor", src_desc);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;

  const bool manual_clamp = src_tensor_type == TensorStorageType::BUFFER ||
                            src_tensor_type == TensorStorageType::IMAGE_BUFFER;

  std::string c;
  if (local_mem_uploads && gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(8, 4, 1)))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = (linear_id / args.dst_tensor.Batch()) * 2;\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0 * 2;\n";
  }
  c += "  int Y = GLOBAL_ID_1 * 2;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "   ACCUM_FLT4 r0 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "   ACCUM_FLT4 r1 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "   ACCUM_FLT4 r2 = INIT_ACCUM_FLT4(0.0f);\n";
  c += "   ACCUM_FLT4 r3 = INIT_ACCUM_FLT4(0.0f);\n";
  if (!local_mem_uploads) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| S >= args.dst_tensor.Slices()) { \n";
    c += "    return; \n";
    c += "  } \n";
  }
  if (local_mem_uploads) {
    c += "  __local FLT4 f[10];\n";
    if (gpu_info.IsApiOpenCl() && gpu_info.IsPowerVR()) {
      c += "  event_t e = async_work_group_copy(f, args.weights.GetPtr() + S * "
           "10, 10, 0);\n";
      c += "  wait_group_events(1, &e);\n";
    } else {
      c += "  int local_id = LOCAL_ID_1 * 8 + LOCAL_ID_0;\n";
      c += "  if (local_id < 10) {\n";
      c += "    f[local_id] = args.weights.Read(S * 10 + local_id);\n";
      c += "  }\n";
      c += "  LOCAL_MEM_BARRIER;\n";
    }
  } else if (weights_are_buffer && gpu_info.SupportsPointersInKernels()) {
    c += "  __global FLT4* f = args.weights.GetPtr() + S * 10;\n";
  }
  c += "  FLT4 s0;\n";
  c += "  FLT4 s1;\n";
  c += "  FLT4 s2;\n";
  c += "  FLT4 s3;\n";
  std::string W[9] = {"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"};
  std::string bias = "bias";
  std::string xc[4] = {"X - 1", "X", "X + 1", "X + 2"};
  std::string yc[4] = {"Y - 1", "Y", "Y + 1", "Y + 2"};
  if (!weights_are_buffer) {
    c += "   FLT4 f0 = args.weights.Read(0, S);\n";
    c += "   FLT4 f1 = args.weights.Read(1, S);\n";
    c += "   FLT4 f2 = args.weights.Read(2, S);\n";
    c += "   FLT4 f3 = args.weights.Read(3, S);\n";
    c += "   FLT4 f4 = args.weights.Read(4, S);\n";
    c += "   FLT4 f5 = args.weights.Read(5, S);\n";
    c += "   FLT4 f6 = args.weights.Read(6, S);\n";
    c += "   FLT4 f7 = args.weights.Read(7, S);\n";
    c += "   FLT4 f8 = args.weights.Read(8, S);\n";
  }
  if (manual_clamp) {
    c += "  int x0 = X - 1;\n";
    c += "  int x1 = X;\n";
    c += "  int x2 = X + 1;\n";
    c += "  int x3 = X + 2;\n";
    c += "  int y0 = Y - 1;\n";
    c += "  int y1 = Y;\n";
    c += "  int y2 = Y + 1;\n";
    c += "  int y3 = Y + 2;\n";
    c += "  bool x0_in = x0 >= 0 && x0 < args.dst_tensor.Width();\n";
    c += "  bool x1_in = x1 >= 0 && x1 < args.dst_tensor.Width();\n";
    c += "  bool x2_in = x2 >= 0 && x2 < args.dst_tensor.Width();\n";
    c += "  bool x3_in = x3 >= 0 && x3 < args.dst_tensor.Width();\n";
    c += "  bool y0_in = y0 >= 0 && y0 < args.dst_tensor.Height();\n";
    c += "  bool y1_in = y1 >= 0 && y1 < args.dst_tensor.Height();\n";
    c += "  bool y2_in = y2 >= 0 && y2 < args.dst_tensor.Height();\n";
    c += "  bool y3_in = y3 >= 0 && y3 < args.dst_tensor.Height();\n";
    c += "  x0 = clamp(x0, 0, args.dst_tensor.Width() - 1);\n";
    c += "  x1 = clamp(x1, 0, args.dst_tensor.Width() - 1);\n";
    c += "  x2 = clamp(x2, 0, args.dst_tensor.Width() - 1);\n";
    c += "  x3 = clamp(x3, 0, args.dst_tensor.Width() - 1);\n";
    c += "  y0 = clamp(y0, 0, args.dst_tensor.Height() - 1);\n";
    c += "  y1 = clamp(y1, 0, args.dst_tensor.Height() - 1);\n";
    c += "  y2 = clamp(y2, 0, args.dst_tensor.Height() - 1);\n";
    c += "  y3 = clamp(y3, 0, args.dst_tensor.Height() - 1);\n";
    if (src_tensor_type == TensorStorageType::BUFFER &&
        gpu_info.SupportsPointersInKernels()) {
      c += "  __global FLT4* src_loc = "
           "args.src_tensor.GetPtrWithSliceOffset(S);\n";
    }
    xc[0] = "x0";
    xc[1] = "x1";
    xc[2] = "x2";
    xc[3] = "x3";
    yc[0] = "y0";
    yc[1] = "y1";
    yc[2] = "y2";
    yc[3] = "y3";
  }
  if (local_mem_uploads || weights_are_buffer) {
    const bool use_direct_buffer =
        !local_mem_uploads && !gpu_info.SupportsPointersInKernels();
    const std::string fetch_start =
        use_direct_buffer ? "args.weights.Read(S * 10 + " : "f[";
    const std::string fetch_end = use_direct_buffer ? ")" : "]";
    W[0] = fetch_start + "0" + fetch_end;
    W[1] = fetch_start + "1" + fetch_end;
    W[2] = fetch_start + "2" + fetch_end;
    W[3] = fetch_start + "3" + fetch_end;
    W[4] = fetch_start + "4" + fetch_end;
    W[5] = fetch_start + "5" + fetch_end;
    W[6] = fetch_start + "6" + fetch_end;
    W[7] = fetch_start + "7" + fetch_end;
    W[8] = fetch_start + "8" + fetch_end;
    bias = fetch_start + "9" + fetch_end;
  }
  auto read_4x_line = [&](int y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_4(mht_4_v, 366, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "lambda");

    if (src_tensor_type == TensorStorageType::BUFFER &&
        gpu_info.SupportsPointersInKernels()) {
      const std::string y_in = "y" + std::to_string(y) + "_in";
      c += "    s0 = src_loc[args.src_tensor.GetWHOffset(" + xc[0] + ", " +
           yc[y] + ")] * INIT_FLT(x0_in && " + y_in + ");\n";
      c += "    s1 = src_loc[args.src_tensor.GetWHOffset(" + xc[1] + ", " +
           yc[y] + ")] * INIT_FLT(x1_in && " + y_in + ");\n";
      c += "    s2 = src_loc[args.src_tensor.GetWHOffset(" + xc[2] + ", " +
           yc[y] + ")] * INIT_FLT(x2_in && " + y_in + ");\n";
      c += "    s3 = src_loc[args.src_tensor.GetWHOffset(" + xc[3] + ", " +
           yc[y] + ")] * INIT_FLT(x3_in && " + y_in + ");\n";
    } else if (src_tensor_type == TensorStorageType::IMAGE_BUFFER ||
               src_tensor_type == TensorStorageType::BUFFER) {
      const std::string y_in = "y" + std::to_string(y) + "_in";
      c += "    s0 = args.src_tensor.Read(" + xc[0] + ", " + yc[y] +
           ", S) * INIT_FLT(x0_in && " + y_in + ");\n";
      c += "    s1 = args.src_tensor.Read(" + xc[1] + ", " + yc[y] +
           ", S) * INIT_FLT(x1_in && " + y_in + ");\n";
      c += "    s2 = args.src_tensor.Read(" + xc[2] + ", " + yc[y] +
           ", S) * INIT_FLT(x2_in && " + y_in + ");\n";
      c += "    s3 = args.src_tensor.Read(" + xc[3] + ", " + yc[y] +
           ", S) * INIT_FLT(x3_in && " + y_in + ");\n";
    } else {
      c += "    s0 = args.src_tensor.Read(" + xc[0] + ", " + yc[y] + ", S);\n";
      c += "    s1 = args.src_tensor.Read(" + xc[1] + ", " + yc[y] + ", S);\n";
      c += "    s2 = args.src_tensor.Read(" + xc[2] + ", " + yc[y] + ", S);\n";
      c += "    s3 = args.src_tensor.Read(" + xc[3] + ", " + yc[y] + ", S);\n";
    }
  };
  c += "  {\n";
  read_4x_line(0);
  c += "    r0 += TO_ACCUM_TYPE(" + W[0] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[1] + " * s1);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[0] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[2] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[1] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[2] + " * s3);\n";
  c += "  }\n";
  c += "  {\n";
  read_4x_line(1);
  c += "    r0 += TO_ACCUM_TYPE(" + W[3] + " * s0);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[0] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[4] + " * s1);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[3] + " * s1);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[1] + " * s1);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[0] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[5] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[4] + " * s2);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[2] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[1] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[5] + " * s3);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[2] + " * s3);\n";
  c += "  }\n";
  c += "  {\n";
  read_4x_line(2);
  c += "    r0 += TO_ACCUM_TYPE(" + W[6] + " * s0);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[3] + " * s0);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[7] + " * s1);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[6] + " * s1);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[4] + " * s1);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[3] + " * s1);\n";
  c += "    r0 += TO_ACCUM_TYPE(" + W[8] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[7] + " * s2);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[5] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[4] + " * s2);\n";
  c += "    r1 += TO_ACCUM_TYPE(" + W[8] + " * s3);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[5] + " * s3);\n";
  c += "  }\n";
  c += "  {\n";
  read_4x_line(3);
  c += "    r2 += TO_ACCUM_TYPE(" + W[6] + " * s0);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[7] + " * s1);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[6] + " * s1);\n";
  c += "    r2 += TO_ACCUM_TYPE(" + W[8] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[7] + " * s2);\n";
  c += "    r3 += TO_ACCUM_TYPE(" + W[8] + " * s3);\n";
  c += "  }\n";
  if (!weights_are_buffer) {
    c += "   FLT4 bias = args.weights.Read(9, S);\n";
  }
  c += "  r0 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  r1 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  r2 += TO_ACCUM_TYPE(" + bias + ");\n";
  c += "  r3 += TO_ACCUM_TYPE(" + bias + ");\n";
  if (local_mem_uploads) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| S >= args.dst_tensor.Slices()) { \n";
    c += "    return; \n";
    c += "  } \n";
  }
  c += "  if(X + 0 < args.dst_tensor.Width() && Y + 0 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r0);\n";
  c += "    args.dst_tensor.Write(result, X + 0, Y + 0, S);\n";
  c += "  }\n";
  c += "  if(X + 1 < args.dst_tensor.Width() && Y + 0 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r1);\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y + 0, S);\n";
  c += "  }\n";
  c += "  if(X + 0 < args.dst_tensor.Width() && Y + 1 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r2);\n";
  c += "    args.dst_tensor.Write(result, X + 0, Y + 1, S);\n";
  c += "  }\n";
  c += "  if(X + 1 < args.dst_tensor.Width() && Y + 1 < "
       "args.dst_tensor.Height()) {\n";
  c += "    FLT4 result = TO_FLT4(r3);\n";
  c += "    args.dst_tensor.Write(result, X + 1, Y + 1, S);\n";
  c += "  }\n";
  c += "}\n";

  return c;
}

int3 DepthwiseConv3x3::GetGridSize() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_5(mht_5_v, 485, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "DepthwiseConv3x3::GetGridSize");

  const int grid_x = DivideRoundUp(dst_[0]->Width(), 2) * dst_[0]->Batch();
  const int grid_y = DivideRoundUp(dst_[0]->Height(), 2);
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

void DepthwiseConv3x3::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info, std::vector<int3>* work_groups) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_6(mht_6_v, 497, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "DepthwiseConv3x3::GetPossibleKernelWorkGroups");

  if (local_mem_uploads_) {
    work_groups->push_back(work_group_size_);
  } else {
    GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_,
                          work_groups);
  }
}

bool IsDepthwiseConv3x3Supported(const GpuInfo& gpu_info,
                                 const DepthwiseConvolution2DAttributes& attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_7(mht_7_v, 510, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "IsDepthwiseConv3x3Supported");

  if (gpu_info.IsApiOpenCl() && gpu_info.IsAdreno()) {
    const std::string kBadDriver =
        "OpenCL 2.0 QUALCOMM build: commit #7daed58 changeid #I7ece6fe30d "
        "Date: 10/19/16";
    if (absl::StrContains(gpu_info.opencl_info.platform_version, kBadDriver)) {
      return false;
    }
  }
  return attr.weights.shape.o == 1 && attr.dilations.w == 1 &&
         attr.dilations.h == 1 && attr.weights.shape.w == 3 &&
         attr.weights.shape.h == 3 && attr.strides.w == 1 &&
         attr.strides.h == 1 && attr.padding.prepended.w == 1 &&
         attr.padding.prepended.h == 1 && attr.padding.appended.w == 1 &&
         attr.padding.appended.h == 1;
}

DepthwiseConv3x3 CreateDepthwiseConv3x3(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTcc mht_8(mht_8_v, 532, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.cc", "CreateDepthwiseConv3x3");

  bool weights_are_buffer = !gpu_info.SupportsImages() ||
                            gpu_info.IsPowerVR() || gpu_info.IsMali() ||
                            gpu_info.IsApple();
  bool local_mem_uploads = weights_are_buffer && gpu_info.IsPowerVR();
  if (gpu_info.IsApple() &&
      gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal()) {
    local_mem_uploads = true;
  }
  DepthwiseConv3x3 result(definition, weights_are_buffer, local_mem_uploads,
                          gpu_info);
  result.UploadWeightsAndBiases(attr.weights, attr.bias, weights_are_buffer);
  return result;
}

}  // namespace gpu
}  // namespace tflite
