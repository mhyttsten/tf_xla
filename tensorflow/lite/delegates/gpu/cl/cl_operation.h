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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_OPERATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_OPERATION_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh() {
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


#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {
namespace cl {

struct CreationContext {
  const CLDevice* device;
  CLContext* context;
  CLCommandQueue* queue;
  ProgramCache* cache;

  const GpuInfo& GetGpuInfo() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_0(mht_0_v, 212, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "GetGpuInfo");
 return device->info_; }
};

class ClOperation {
 public:
  ClOperation() = default;
  virtual ~ClOperation() = default;
  // Move only
  ClOperation(ClOperation&& operation) = default;
  ClOperation& operator=(ClOperation&& operation) = default;
  ClOperation(const ClOperation&) = delete;
  ClOperation& operator=(const ClOperation&) = delete;

  void Init(std::unique_ptr<GPUOperation>&& gpu_operation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_1(mht_1_v, 228, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "Init");

    operation_ = std::move(gpu_operation);
  }

  GPUOperation& GetGpuOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_2(mht_2_v, 235, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "GetGpuOperation");
 return *operation_; }
  const GPUOperation& GetGpuOperation() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_3(mht_3_v, 239, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "GetGpuOperation");
 return *operation_; }
  uint64_t GetKernelFingerprint() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_4(mht_4_v, 243, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "GetKernelFingerprint");
 return kernel_fingerprint_; }

  const OperationDef& GetDefinition() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_5(mht_5_v, 248, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "GetDefinition");

    return operation_->GetDefinition();
  }

  absl::Status AddOperation(ClOperation* operation);

  // should be called after changes of inputs/outputs.
  absl::Status UpdateParams();

  absl::Status SetSrcTensor(int index, Tensor* tensor);
  absl::Status SetDstTensor(int index, Tensor* tensor);

  absl::Status AddToQueue(CLCommandQueue* queue) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_6(mht_6_v, 263, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "AddToQueue");

    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    return queue->Dispatch(kernel_, operation_->GetWorkGroupsCount(),
                           operation_->work_group_size_);
  }

  // for better profiling
  absl::Status AddToQueueNTimes(ProfilingCommandQueue* queue, int n,
                                int flush_period = 0) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_7(mht_7_v, 274, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "AddToQueueNTimes");

    RETURN_IF_ERROR(cl_args_.Bind(kernel_.kernel()));
    return queue->DispatchNTimes(kernel_, operation_->GetWorkGroupsCount(),
                                 operation_->work_group_size_, n, flush_period);
  }

  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);

  absl::Status Compile(const CreationContext& creation_context);

  absl::Status RestoreDeserialized(const ProgramCache& program_cache,
                                   uint64_t fingerprint,
                                   const GpuInfo& gpu_info,
                                   const int3& work_group_size,
                                   CLContext* context);

  int3 GetWorkGroupSize() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_operationDTh mht_8(mht_8_v, 294, "", "./tensorflow/lite/delegates/gpu/cl/cl_operation.h", "GetWorkGroupSize");
 return operation_->work_group_size_; }

 private:
  std::unique_ptr<GPUOperation> operation_;
  CLKernel kernel_;
  uint64_t kernel_fingerprint_;
  CLArguments cl_args_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_OPERATION_H_
