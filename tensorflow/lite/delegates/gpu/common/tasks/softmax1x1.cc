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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {

Softmax1x1::Softmax1x1(const OperationDef& definition)
    : GPUOperation(definition) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "Softmax1x1::Softmax1x1");

  work_group_size_ = int3(32, 1, 1);
  code_ = GetSoftmaxKernelCode(definition_);
}

Softmax1x1::Softmax1x1(Softmax1x1&& kernel) : GPUOperation(std::move(kernel)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "Softmax1x1::Softmax1x1");
}

Softmax1x1& Softmax1x1::operator=(Softmax1x1&& kernel) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_2(mht_2_v, 210, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "=");

  if (this != &kernel) {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

std::string Softmax1x1::GetSoftmaxKernelCode(const OperationDef& op_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_3(mht_3_v, 220, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "Softmax1x1::GetSoftmaxKernelCode");

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int batch_id = GLOBAL_ID_1;\n";
    c += "  if (batch_id >= args.dst_tensor.Batch()) return;\n";
    c += "  args.dst_tensor.SetBatchRef(batch_id);\n";
    c += "  args.src_tensor.SetBatchRef(batch_id);\n";
  }
  c += "  float4 mask = INIT_FLOAT4v4(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c +=
      "  float4 maxx4 = INIT_FLOAT4(args.src_tensor.Read<float>(0, 0, 0).x);\n";
  c += "  int tid = LOCAL_ID_0;\n";
  c += "  for (int s = tid; s < args.src_tensor.Slices(); s += 32) {\n";
  c += "    float4 mask_a = s == args.src_tensor.Slices() - 1 ? mask : "
       "INIT_FLOAT4(1.0f);\n";
  c += "    float4 mask_b = INIT_FLOAT4(1.0f) - mask_a;\n";
  c += "    float4 src = args.src_tensor.Read<float>(0, 0, s);\n";
  c += "    src = src * mask_a + mask_b * src.x;\n";
  c += "    maxx4 = max(maxx4, src);\n";
  c += "  }\n";
  c += "  float maximum = max(maxx4.x, maxx4.y);\n";
  c += "  maximum = max(maximum, maxx4.z);\n";
  c += "  maximum = max(maximum, maxx4.w);\n";
  c += "  __local float loc_mem[32];\n";
  c += "  loc_mem[tid] = maximum;\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid % 8 == 0) {\n";
  c += "    maximum = max(loc_mem[tid], loc_mem[tid + 1]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 2]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 3]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 4]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 5]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 6]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 7]);\n";
  c += "    loc_mem[tid] = maximum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid == 0) {\n";
  c += "    maximum = max(loc_mem[0], loc_mem[8]);\n";
  c += "    maximum = max(maximum, loc_mem[16]);\n";
  c += "    maximum = max(maximum, loc_mem[24]);\n";
  c += "    loc_mem[0] = maximum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  maximum = loc_mem[0];\n";
  c += "  float sum = 0.0f;\n";
  c += "  for (int s = tid; s < args.src_tensor.Slices(); s += 32) {\n";
  c += "    float4 mask_temp = s == args.src_tensor.Slices() - 1 ? mask : "
       "INIT_FLOAT4(1.0f);\n";
  c += "    float4 src = args.src_tensor.Read<float>(0, 0, s) - "
       "INIT_FLOAT4(maximum);\n";
  c += "    sum += dot(mask_temp, exp(src));\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  loc_mem[tid] = sum;\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid % 8 == 0) {\n";
  c += "    sum = loc_mem[tid] + loc_mem[tid + 1];\n";
  c += "    sum += loc_mem[tid + 2];\n";
  c += "    sum += loc_mem[tid + 3];\n";
  c += "    sum += loc_mem[tid + 4];\n";
  c += "    sum += loc_mem[tid + 5];\n";
  c += "    sum += loc_mem[tid + 6];\n";
  c += "    sum += loc_mem[tid + 7];\n";
  c += "    loc_mem[tid] = sum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid == 0) {\n";
  c += "    sum = loc_mem[0] + loc_mem[8] + loc_mem[16] + loc_mem[24];\n";
  c += "    loc_mem[0] = 1.0f / sum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  sum = loc_mem[0];\n";
  c += "\n";
  c += "  int dst_s = GLOBAL_ID_0;\n";
  c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
  c += "    float4 src = args.src_tensor.Read<float>(0, 0, dst_s) - "
       "INIT_FLOAT4(maximum);\n";
  c += "    FLT4 res = TO_FLT4(exp(src) * sum);\n";
  c += "    args.dst_tensor.Write(res, 0, 0, dst_s);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

absl::Status Softmax1x1::BindArguments(ArgumentsBinder* args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_4(mht_4_v, 317, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "Softmax1x1::BindArguments");

  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  RETURN_IF_ERROR(args->SetFloat("mask_w", mask.w));
  return absl::OkStatus();
}

int3 Softmax1x1::GetGridSize() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_5(mht_5_v, 329, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "Softmax1x1::GetGridSize");

  return int3(dst_[0]->Slices(), dst_[0]->Batch(), 1);
}

Softmax1x1 CreateSoftmax1x1(const OperationDef& definition) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSsoftmax1x1DTcc mht_6(mht_6_v, 336, "", "./tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.cc", "CreateSoftmax1x1");

  return Softmax1x1(definition);
}

}  // namespace gpu
}  // namespace tflite
