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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_ARGUMENTS_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTh() {
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


#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base_generated.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
class CLArguments;
}

class ArgumentsBinder {
 public:
  virtual absl::Status SetInt(const std::string& name, int value) = 0;
  virtual absl::Status SetFloat(const std::string& name, float value) = 0;
  virtual absl::Status SetHalf(const std::string& name, half value) = 0;
  virtual ~ArgumentsBinder() = default;
};

class Arguments : public ArgumentsBinder {
 public:
  Arguments() = default;
  ~Arguments() override = default;

  // Move only
  Arguments(Arguments&& args) = default;
  Arguments& operator=(Arguments&& args) = default;
  Arguments(const Arguments&) = delete;
  Arguments& operator=(const Arguments&) = delete;

  void AddFloat(const std::string& name, float value = 0.0f);
  void AddHalf(const std::string& name, half value = half(0.0f));
  void AddInt(const std::string& name, int value = 0);
  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetFloat(const std::string& name, float value) override;
  absl::Status SetHalf(const std::string& name, half value) override;
  void AddObjectRef(const std::string& name, AccessType access_type,
                    GPUObjectDescriptorPtr&& descriptor_ptr);
  void AddObject(const std::string& name,
                 GPUObjectDescriptorPtr&& descriptor_ptr);

  void RenameArgs(const std::string& postfix, std::string* code) const;
  absl::Status Merge(Arguments&& args, const std::string& postfix,
                     const std::vector<std::string>& exception_names = {});

  absl::Status GetDescriptor(const std::string& name,
                             GPUObjectDescriptor** descriptor) const;

  int GetReadTexturesCount(const GpuInfo& gpu_info) const;
  int GetWriteTexturesCount(const GpuInfo& gpu_info) const;

  void ReleaseCPURepresentation();

  void GetActiveArguments(const std::string& code);

  void SetStateValueForAllObjects(const std::string& key,
                                  const std::string& value);

  struct IntValue {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  struct FloatValue {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  struct HalfValue {
    half value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };

  const std::map<std::string, IntValue>& GetIntValues() const {
    return int_values_;
  }
  const std::map<std::string, FloatValue>& GetFloatValues() const {
    return float_values_;
  }
  const std::map<std::string, HalfValue>& GetHalfValues() const {
    return half_values_;
  }

  const std::map<std::string, GPUObjectDescriptorPtr>& GetObjectRefs() const {
    return object_refs_;
  }
  const std::map<std::string, GPUObjectDescriptorPtr>& GetObjects() const {
    return objects_;
  }
  void MoveObjectRefs(std::map<std::string, GPUObjectDescriptorPtr>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTh mht_0(mht_0_v, 291, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.h", "MoveObjectRefs");

    *result = std::move(object_refs_);
  }

  absl::Status Compile(const GpuInfo& gpu_info,
                       const std::map<std::string, std::string>& linkables,
                       std::string* code);

  absl::Status ResolveConstExprPass(const GpuInfo& gpu_info,
                                    std::string* code) const;

  absl::Status ResolveConstExpr(const GpuInfo& gpu_info,
                                const std::string& object_name,
                                const std::string& const_expr,
                                std::string* result) const;

  absl::Status ResolveSelectorsPass(
      const GpuInfo& gpu_info,
      const std::map<std::string, std::string>& linkables,
      std::string* code) const;

  absl::Status ResolveSelector(
      const GpuInfo& gpu_info,
      const std::map<std::string, std::string>& linkables,
      const std::string& object_name, const std::string& selector,
      const std::vector<std::string>& function_args,
      const std::vector<std::string>& template_args, std::string* result) const;

  void ResolveObjectNames(const std::string& object_name,
                          const std::vector<std::string>& member_names,
                          std::string* code) const;
  absl::Status AddObjectsScalarArgs(const GpuInfo& gpu_info);
  void ResolveArgsPass(std::string* code) const;

 private:
  friend flatbuffers::Offset<tflite::gpu::data::Arguments> Encode(
      const Arguments& args, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const tflite::gpu::data::Arguments* fb_args,
                             Arguments* args);

  absl::Status ResolveKernelGlobalSpaceBuffers(const GpuInfo& gpu_info,
                                               std::string* code);

  friend class cl::CLArguments;

  static constexpr char kArgsPrefix[] = "args.";

  std::map<std::string, IntValue> int_values_;
  std::map<std::string, FloatValue> float_values_;
  std::map<std::string, HalfValue> half_values_;

  std::map<std::string, GPUObjectDescriptorPtr> object_refs_;
  std::map<std::string, GPUObjectDescriptorPtr> objects_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_ARGUMENTS_H_
