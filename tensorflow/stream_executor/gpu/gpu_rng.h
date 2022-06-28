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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_RNG_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_RNG_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh() {
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


#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rng.h"

namespace stream_executor {

class Stream;
template <typename ElemT>
class DeviceMemory;

namespace gpu {

// Opaque and unique identifier for the GPU RNG plugin.
extern const PluginId kGpuRandPlugin;

class GpuExecutor;

// GPU-platform implementation of the random number generation support
// interface.
//
// Thread-safe post-initialization.
class GpuRng : public rng::RngSupport {
 public:
  explicit GpuRng(GpuExecutor* parent);

  // Retrieves a gpu rng library generator handle. This is necessary for
  // enqueuing random number generation work onto the device.
  // TODO(leary) provide a way for users to select the RNG algorithm.
  bool Init();

  // Releases a gpu rng library generator handle, if one was acquired.
  ~GpuRng() override;

  // See rng::RngSupport for details on the following overrides.
  bool DoPopulateRandUniform(Stream* stream, DeviceMemory<float>* v) override;
  bool DoPopulateRandUniform(Stream* stream, DeviceMemory<double>* v) override;
  bool DoPopulateRandUniform(Stream* stream,
                             DeviceMemory<std::complex<float>>* v) override;
  bool DoPopulateRandUniform(Stream* stream,
                             DeviceMemory<std::complex<double>>* v) override;
  bool DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                              DeviceMemory<float>* v) override;
  bool DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                              DeviceMemory<double>* v) override;

  bool SetSeed(Stream* stream, const uint8* seed, uint64_t seed_bytes) override;

 private:
  // Actually performs the work of generating random numbers - the public
  // methods are thin wrappers to this interface.
  template <typename T>
  bool DoPopulateRandUniformInternal(Stream* stream, DeviceMemory<T>* v);
  template <typename ElemT, typename FuncT>
  bool DoPopulateRandGaussianInternal(Stream* stream, ElemT mean, ElemT stddev,
                                      DeviceMemory<ElemT>* v, FuncT func);

  // Sets the stream for the internal gpu rng generator.
  //
  // This is a stateful operation, as the handle can only have one stream set at
  // a given time, so it is usually performed right before enqueuing work to do
  // with random number generation.
  bool SetStream(Stream* stream) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Guards the gpu rng library handle for this device.
  absl::Mutex mu_;

  // GpuExecutor which instantiated this GpuRng.
  // Immutable post-initialization.
  GpuExecutor* parent_;

  // gpu rng library handle on the device.
  GpuRngHandle rng_ TF_GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(GpuRng);
};

template <typename T>
std::string TypeString();

template <>
std::string TypeString<float>() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh mht_0(mht_0_v, 271, "", "./tensorflow/stream_executor/gpu/gpu_rng.h", "TypeString<float>");

  return "float";
}

template <>
std::string TypeString<double>() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh mht_1(mht_1_v, 279, "", "./tensorflow/stream_executor/gpu/gpu_rng.h", "TypeString<double>");

  return "double";
}

template <>
std::string TypeString<std::complex<float>>() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh mht_2(mht_2_v, 287, "", "./tensorflow/stream_executor/gpu/gpu_rng.h", "TypeString<std::complex<float>>");

  return "std::complex<float>";
}

template <>
std::string TypeString<std::complex<double>>() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_rngDTh mht_3(mht_3_v, 295, "", "./tensorflow/stream_executor/gpu/gpu_rng.h", "TypeString<std::complex<double>>");

  return "std::complex<double>";
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_RNG_H_
