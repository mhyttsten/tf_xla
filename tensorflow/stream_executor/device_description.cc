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
class MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/device_description.h"

#include <algorithm>

#include "absl/strings/str_cat.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/mathutil.h"

namespace stream_executor {

static const uint64_t kUninitializedUint64 = -1ULL;
/* static */ const char *DeviceDescription::kUndefinedString = "<undefined>";

DeviceDescription::DeviceDescription()
    : device_vendor_(kUndefinedString),
      platform_version_(kUndefinedString),
      driver_version_(kUndefinedString),
      runtime_version_(kUndefinedString),
      pci_bus_id_(kUndefinedString),
      name_(kUndefinedString),
      thread_dim_limit_(kUninitializedUint64, kUninitializedUint64,
                        kUninitializedUint64),
      block_dim_limit_(kUninitializedUint64, kUninitializedUint64,
                       kUninitializedUint64),
      threads_per_core_limit_(kUninitializedUint64),
      threads_per_block_limit_(kUninitializedUint64),
      threads_per_warp_(kUninitializedUint64),
      registers_per_core_limit_(kUninitializedUint64),
      registers_per_block_limit_(kUninitializedUint64),
      device_address_bits_(kUninitializedUint64),
      device_memory_size_(kUninitializedUint64),
      memory_bandwidth_(kUninitializedUint64),
      shared_memory_per_core_(kUninitializedUint64),
      shared_memory_per_block_(kUninitializedUint64),
      clock_rate_ghz_(-1.0),
      numa_node_(-1),
      core_count_(-1),
      ecc_enabled_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_0(mht_0_v, 222, "", "./tensorflow/stream_executor/device_description.cc", "DeviceDescription::DeviceDescription");
}

std::unique_ptr<std::map<std::string, std::string>> DeviceDescription::ToMap()
    const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_1(mht_1_v, 228, "", "./tensorflow/stream_executor/device_description.cc", "DeviceDescription::ToMap");

  std::unique_ptr<std::map<std::string, std::string>> owned_result{
      new std::map<std::string, std::string>};
  std::map<std::string, std::string> &result = *owned_result;
  result["Device Vendor"] = device_vendor();
  result["Platform Version"] = platform_version();
  result["Driver Version"] = driver_version();
  result["Runtime Version"] = runtime_version();
  result["PCI bus ID"] = pci_bus_id_;
  result["Device Name"] = name_;

  const ThreadDim &thread_dim = thread_dim_limit();
  result["ThreadDim Limit"] =
      absl::StrCat(thread_dim.x, ",", thread_dim.y, ",", thread_dim.z);
  const BlockDim &block_dim = block_dim_limit();
  result["BlockDim Limit"] =
      absl::StrCat(block_dim.x, ",", block_dim.y, ",", block_dim.z);

  result["Threads Per Core Limit"] = absl::StrCat(threads_per_core_limit());
  result["Threads Per Block Limit"] = absl::StrCat(threads_per_block_limit());
  result["Registers Per Block Limit"] =
      absl::StrCat(registers_per_block_limit());

  result["Device Address Bits"] = absl::StrCat(device_address_bits());
  result["Device Memory Size"] =
      port::HumanReadableNumBytes::ToString(device_memory_size());
  result["Memory Bandwidth"] = absl::StrCat(
      port::HumanReadableNumBytes::ToString(memory_bandwidth_), "/s");

  result["Shared Memory Per Core"] =
      port::HumanReadableNumBytes::ToString(shared_memory_per_core_);
  result["Shared Memory Per Block"] =
      port::HumanReadableNumBytes::ToString(shared_memory_per_block_);

  result["Clock Rate GHz"] = absl::StrCat(clock_rate_ghz());

  result["CUDA Compute Capability"] = cuda_compute_capability().ToString();

  result["AMDGPU GCN Arch Name"] = rocm_compute_capability().gcn_arch_name();

  result["NUMA Node"] = absl::StrCat(numa_node());
  result["Core Count"] = absl::StrCat(core_count());
  result["ECC Enabled"] = absl::StrCat(ecc_enabled());
  return owned_result;
}

namespace internal {

DeviceDescriptionBuilder::DeviceDescriptionBuilder()
    : device_description_(new DeviceDescription) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_2(mht_2_v, 280, "", "./tensorflow/stream_executor/device_description.cc", "DeviceDescriptionBuilder::DeviceDescriptionBuilder");
}

}  // namespace internal

CudaComputeCapability DeviceDescription::cuda_compute_capability() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_3(mht_3_v, 287, "", "./tensorflow/stream_executor/device_description.cc", "DeviceDescription::cuda_compute_capability");

  return cuda_compute_capability_;
}

RocmComputeCapability DeviceDescription::rocm_compute_capability() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_4(mht_4_v, 294, "", "./tensorflow/stream_executor/device_description.cc", "DeviceDescription::rocm_compute_capability");

  return rocm_compute_capability_;
}

bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_5(mht_5_v, 302, "", "./tensorflow/stream_executor/device_description.cc", "ThreadDimOk");

  const int64_t total_threads = thread_dim.x * thread_dim.y * thread_dim.z;
  const int64_t threads_per_block_limit =
      device_description.threads_per_block_limit();
  if (total_threads > threads_per_block_limit) {
    VLOG(2) << "exceeded total-thread-per-block limit: " << total_threads
            << " vs limit " << threads_per_block_limit;
    return false;
  }

  const auto &limit = device_description.thread_dim_limit();
  bool ok = thread_dim.x <= limit.x && thread_dim.y <= limit.y &&
            thread_dim.z <= limit.z;
  if (!ok) {
    VLOG(2) << "thread dim " << thread_dim.ToString()
            << " exceeds limit constraints of " << limit.ToString();
  }
  return ok;
}

uint64_t DivideCeil(uint64 x, uint64 y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_6(mht_6_v, 325, "", "./tensorflow/stream_executor/device_description.cc", "DivideCeil");

  return port::MathUtil::CeilOfRatio(x, y);
}

void CalculateDimensionality(const DeviceDescription &device_description,
                             int64_t element_count, int64_t *threads_per_block,
                             int64_t *block_count) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTcc mht_7(mht_7_v, 334, "", "./tensorflow/stream_executor/device_description.cc", "CalculateDimensionality");

  *threads_per_block = device_description.threads_per_block_limit();
  *block_count = port::MathUtil::CeilOfRatio(element_count, *threads_per_block);
  if (*block_count == 1) {
    CHECK_LE(element_count, *threads_per_block);
    *threads_per_block = element_count;
  }
}

}  // namespace stream_executor
