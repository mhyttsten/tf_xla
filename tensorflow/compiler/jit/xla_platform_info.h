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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_
#define TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_
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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh() {
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


#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tensorflow {

// Holds some information about the platform on which an
// XlaLaunch/_XlaCompile/_XlaRun op must run on. Provides a common layer of
// abstraction for normal and XLA devices.
class XlaPlatformInfo {
 public:
  XlaPlatformInfo() : device_type_("") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_0(mht_0_v, 199, "", "./tensorflow/compiler/jit/xla_platform_info.h", "XlaPlatformInfo");
}
  XlaPlatformInfo(XlaPlatformInfo&&) = default;
  explicit XlaPlatformInfo(
      const DeviceType device_type, se::Platform::Id platform_id,
      const XlaDevice::Metadata* xla_device_metadata,
      std::shared_ptr<se::DeviceMemoryAllocator> device_allocator)
      : device_type_(device_type),
        platform_id_(platform_id),
        xla_device_metadata_(xla_device_metadata),
        device_allocator_(device_allocator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_1(mht_1_v, 211, "", "./tensorflow/compiler/jit/xla_platform_info.h", "XlaPlatformInfo");
}

  XlaPlatformInfo& operator=(XlaPlatformInfo&& other) = default;

  bool UseMultipleStreams() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_2(mht_2_v, 218, "", "./tensorflow/compiler/jit/xla_platform_info.h", "UseMultipleStreams");

    return xla_device_metadata_ && xla_device_metadata_->UseMultipleStreams();
  }

  // Non-null only when run on an XLA device.
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator() const {
    return device_allocator_;
  }

  DeviceType device_type() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_3(mht_3_v, 230, "", "./tensorflow/compiler/jit/xla_platform_info.h", "device_type");
 return device_type_; }

  // This is equal to xla_device_metadata()->platform()->id() if
  // xla_device_metadata() is not nullptr.
  se::Platform::Id platform_id() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_4(mht_4_v, 237, "", "./tensorflow/compiler/jit/xla_platform_info.h", "platform_id");
 return platform_id_; }

  // This may be null if the op this XlaPlatformInfo is for was not placed on an
  // XLA device.
  const XlaDevice::Metadata* xla_device_metadata() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_5(mht_5_v, 244, "", "./tensorflow/compiler/jit/xla_platform_info.h", "xla_device_metadata");

    return xla_device_metadata_;
  }
  bool is_on_xla_device() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_platform_infoDTh mht_6(mht_6_v, 250, "", "./tensorflow/compiler/jit/xla_platform_info.h", "is_on_xla_device");
 return xla_device_metadata() != nullptr; }

 private:
  DeviceType device_type_;
  se::Platform::Id platform_id_;

  // xla_device_metadata_ lives in the tensorflow::DeviceBase in which the
  // XlaLaunch/_XlaCompile/_XlaRun op is placed and thus does not die before the
  // XlaLaunch/_XlaCompile/_XlaRun OpKernel.
  const XlaDevice::Metadata* xla_device_metadata_;

  // If the op associated with this XlaPlatformInfo is placed on an XLA device
  // then device_allocator_ is the xla::Backend's memory allocator.  If the op
  // is placed on a regular CPU or GPU device then device_allocator_ is null.
  // The allocator is of unknown provenance; keep it in a shared pointer to
  // set an artificial refcount of one.
  std::shared_ptr<se::DeviceMemoryAllocator> device_allocator_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaPlatformInfo);
};

// Returns a set containing the device ids contained in visible_device_list or
// nullopt if it is empty. It returns error in case of malformed configuration
// string.
StatusOr<absl::optional<std::set<int>>> ParseVisibleDeviceList(
    absl::string_view visible_device_list);

// Returns created XLA compilation cache.
Status BuildXlaCompilationCache(DeviceBase* dev, FunctionLibraryRuntime* flr,
                                const XlaPlatformInfo& platform_info,
                                XlaCompilationCache** cache);

// Returns information about the platform from kernel context.
XlaPlatformInfo XlaPlatformInfoFromDevice(DeviceBase* device);

// Returns allocator from platform info if non-null, or populate and return a
// pointer to the allocator adapter with allocator from context.
//
// This is necessary because for XLA devices the underlying TF allocator returns
// dummy tensors.
//
// `stream` parameter is nullable when running on host.
std::shared_ptr<se::DeviceMemoryAllocator> GetAllocator(
    DeviceBase* device, se::Stream* stream,
    const XlaPlatformInfo& platform_info);

// Returns created options for the XLA compiler, and writes the used allocator
// into `tf_allocator_adapter`.
XlaCompiler::Options GenerateCompilerOptions(
    const XlaCompilationCache& cache,
    const FunctionLibraryRuntime& function_library, DeviceBase* device,
    se::Stream* stream, const XlaPlatformInfo& platform_info,
    bool has_ref_vars);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_
