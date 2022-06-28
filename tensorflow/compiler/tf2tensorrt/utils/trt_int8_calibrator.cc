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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.h"

#include <atomic>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {

// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const noexcept { return batch_size_; }

TRTInt8Calibrator::TRTInt8Calibrator(
    const std::unordered_map<string, std::pair<void*, size_t>>& dev_buffers,
    int batch_size, string engine_name)
    : batch_size_(batch_size),
      done_(false),
      dev_buffers_(dev_buffers),
      // Make sure setBatch() waits until getBatch() is called (the first time).
      calib_running_(true),
      batch_is_set_(false),
      engine_name_(engine_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("engine_name: \"" + engine_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.cc", "TRTInt8Calibrator::TRTInt8Calibrator");
}

TRTInt8Calibrator::TRTInt8Calibrator(const string& calib_data)
    : batch_size_(0),
      done_(true),
      calib_running_(false),
      batch_is_set_(false),
      calibration_table_(calib_data) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("calib_data: \"" + calib_data + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.cc", "TRTInt8Calibrator::TRTInt8Calibrator");
}

bool TRTInt8Calibrator::setBatch(const std::unordered_map<string, void*>& data,
                                 const cudaStream_t stream) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc mht_2(mht_2_v, 228, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.cc", "TRTInt8Calibrator::setBatch");

  mutex_lock lock(cond_mtx_);

  // Wait while the queue is full or calibration is running.
  while ((calib_running_ || batch_is_set_) && !done_) cond_.wait(lock);
  if (done_) return false;
  CHECK(!calib_running_ && !batch_is_set_);
  VLOG(1) << "Set Batch Waiting finished";

  // Sets the batch.
  for (const auto& it : data) {
    auto devptr = dev_buffers_.find(it.first);
    if (devptr == dev_buffers_.end()) {
      LOG(FATAL) << "FATAL " << engine_name_ << " input name '" << it.first
                 << "' does not match with the buffer names";
    }
    const auto& d = devptr->second;

    // TODO(sami,aaroey): Need to figure out a way to ensure synchronization
    // between stream, perhaps using a tensor?
    auto status = cudaMemcpyAsync(d.first, it.second, d.second,
                                  cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
      LOG(FATAL) << "cudaMemcpy " << engine_name_ << " for '" << it.first
                 << "' failed with " << status;
    }
  }

  // TODO(Sami, aaorey): Find an alternative way!
  // we have to wait for the stream before returning!
  cudaStreamSynchronize(stream);
  batch_is_set_ = true;
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,
                                 int num_bindings) noexcept {
  mutex_lock lock(cond_mtx_);
  // Notify finish of last round of calibration.
  calib_running_ = false;
  cond_.notify_all();

  // Wait until new batch arrives
  while ((!batch_is_set_ && !done_)) cond_.wait(lock);
  if (done_) return false;

  // Gets the batch
  for (int i = 0; i < num_bindings; i++) {
    auto it = dev_buffers_.find(names[i]);
    if (it == dev_buffers_.end()) {
      LOG(FATAL) << "Calibration engine asked for unknown tensor name '"
                 << names[i] << "' at position " << i;
    }
    bindings[i] = it->second.first;
  }
  batch_is_set_ = false;
  calib_running_ = true;
  return true;
}

void TRTInt8Calibrator::waitAndSetDone() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.cc", "TRTInt8Calibrator::waitAndSetDone");

  mutex_lock lock(cond_mtx_);
  // Wait while the queue is full or calibration is running, so we don't miss
  // the last batch.
  while ((calib_running_ || batch_is_set_) && !done_) cond_.wait(lock);
  if (!done_) {
    done_ = true;
    cond_.notify_all();
    dev_buffers_.clear();
  }
}

const void* TRTInt8Calibrator::readCalibrationCache(
    std::size_t& length) noexcept {
  if (calibration_table_.empty()) return nullptr;
  length = calibration_table_.size();
  return calibration_table_.data();
}

void TRTInt8Calibrator::setDone() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc mht_4(mht_4_v, 314, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.cc", "TRTInt8Calibrator::setDone");

  mutex_lock lock(cond_mtx_);
  done_ = true;
  cond_.notify_all();
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,
                                              std::size_t length) noexcept {
  calibration_table_ = string(static_cast<const char*>(ptr), length);
  VLOG(1) << "Got calibration data for " << engine_name_ << " @" << ptr
          << " length=" << length;
}
TRTInt8Calibrator::~TRTInt8Calibrator() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSutilsPStrt_int8_calibratorDTcc mht_5(mht_5_v, 329, "", "./tensorflow/compiler/tf2tensorrt/utils/trt_int8_calibrator.cc", "TRTInt8Calibrator::~TRTInt8Calibrator");

  VLOG(1) << "Destroying calibrator for " << engine_name_;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
