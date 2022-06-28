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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_WINOGRAD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_WINOGRAD_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh() {
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


#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"

namespace tflite {
namespace gpu {

// You can read https://arxiv.org/pdf/1509.09308.pdf for understanding of basic
// principles. In this kernels used different matrices for transformations than
// in original work.
class Winograd4x4To36 : public GPUOperation {
 public:
  Winograd4x4To36() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.h", "GetPossibleKernelWorkGroups");

    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;

  // Move only
  Winograd4x4To36(Winograd4x4To36&& kernel) = default;
  Winograd4x4To36& operator=(Winograd4x4To36&& kernel) = default;
  Winograd4x4To36(const Winograd4x4To36&) = delete;
  Winograd4x4To36& operator=(const Winograd4x4To36&) = delete;

 private:
  Winograd4x4To36(const OperationDef& definition, const Padding2D& padding)
      : GPUOperation(definition), padding_(padding) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh mht_1(mht_1_v, 223, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.h", "Winograd4x4To36");
}
  friend Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                               const Padding2D& padding);

  Padding2D padding_;
};

Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                      const Padding2D& padding);

class Winograd4x4To36TileX6 : public GPUOperation {
 public:
  Winograd4x4To36TileX6() = default;
  Winograd4x4To36TileX6(const OperationDef& definition,
                        const Padding2D& padding, const GpuInfo& gpu_info);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;

  // Move only
  Winograd4x4To36TileX6(Winograd4x4To36TileX6&& operation) = default;
  Winograd4x4To36TileX6& operator=(Winograd4x4To36TileX6&& operation) = default;
  Winograd4x4To36TileX6(const Winograd4x4To36TileX6&) = delete;
  Winograd4x4To36TileX6& operator=(const Winograd4x4To36TileX6&) = delete;

 private:
  friend Winograd4x4To36TileX6 CreateWinograd4x4To36TileX6(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Padding2D& padding);

  void UploadBt();

  std::string GetWinograd4x4To36TileX6Code(const OperationDef& op_def,
                                           const GpuInfo& gpu_info);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup(const KernelInfo& kernel_info) const;

  Padding2D padding_;
};

Winograd4x4To36TileX6 CreateWinograd4x4To36TileX6(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Padding2D& padding);

class Winograd36To4x4 : public GPUOperation {
 public:
  Winograd36To4x4() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh mht_2(mht_2_v, 280, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.h", "GetPossibleKernelWorkGroups");

    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  Winograd36To4x4(Winograd36To4x4&& kernel) = default;
  Winograd36To4x4& operator=(Winograd36To4x4&& kernel) = default;
  Winograd36To4x4(const Winograd36To4x4&) = delete;
  Winograd36To4x4& operator=(const Winograd36To4x4&) = delete;

 private:
  explicit Winograd36To4x4(const OperationDef& definition)
      : GPUOperation(definition) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSwinogradDTh mht_3(mht_3_v, 296, "", "./tensorflow/lite/delegates/gpu/common/tasks/winograd.h", "Winograd36To4x4");
}
  friend Winograd36To4x4 CreateWinograd36To4x4(
      const OperationDef& definition,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);
};

Winograd36To4x4 CreateWinograd36To4x4(
    const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

class Winograd36To4x4Tile4x1 : public GPUOperation {
 public:
  Winograd36To4x4Tile4x1() = default;
  Winograd36To4x4Tile4x1(const OperationDef& definition,
                         const GpuInfo& gpu_info);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;

  // Move only
  Winograd36To4x4Tile4x1(Winograd36To4x4Tile4x1&& operation) = default;
  Winograd36To4x4Tile4x1& operator=(Winograd36To4x4Tile4x1&& operation) =
      default;
  Winograd36To4x4Tile4x1(const Winograd36To4x4Tile4x1&) = delete;
  Winograd36To4x4Tile4x1& operator=(const Winograd36To4x4Tile4x1&) = delete;

 private:
  friend Winograd36To4x4Tile4x1 CreateWinograd36To4x4Tile4x1(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

  void UploadAt();

  std::string GetWinograd36To4x4Tile4x1Code(const OperationDef& op_def,
                                            const GpuInfo& gpu_info);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup(const KernelInfo& kernel_info) const;
};

Winograd36To4x4Tile4x1 CreateWinograd36To4x4Tile4x1(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_WINOGRAD_H_
