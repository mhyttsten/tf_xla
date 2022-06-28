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

// Describes the underlying platform for a StreamExecutor; e.g. OpenCL or CUDA
// device and platform properties. Also contains convenience functions for
// checking/calculating launch dimensionality based on device properties.

#ifndef TENSORFLOW_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
#define TENSORFLOW_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh() {
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
#include <memory>
#include <set>
#include <vector>

#include "absl/base/macros.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {
namespace internal {
class DeviceDescriptionBuilder;
}  // namespace internal

// CUDA compute capability, as reported by the device description.
struct CudaComputeCapability {
  int major = 0;
  int minor = 0;

  // MSVC does not like "PASCAL" symbol.
  enum CudaComputeCapabilities { PASCAL_ = 6, VOLTA = 7, AMPERE = 8 };

  CudaComputeCapability() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_0(mht_0_v, 217, "", "./tensorflow/stream_executor/device_description.h", "CudaComputeCapability");
}
  CudaComputeCapability(int major, int minor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_1(mht_1_v, 221, "", "./tensorflow/stream_executor/device_description.h", "CudaComputeCapability");

    this->major = major;
    this->minor = minor;
  }

  bool IsAtLeast(int other_major, int other_minor = 0) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_2(mht_2_v, 229, "", "./tensorflow/stream_executor/device_description.h", "IsAtLeast");

    return !(*this < CudaComputeCapability{other_major, other_minor});
  }

  bool operator<(const CudaComputeCapability &other) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_3(mht_3_v, 236, "", "./tensorflow/stream_executor/device_description.h", "operator<");

    return ToPair() < other.ToPair();
  }

  bool operator==(const CudaComputeCapability &other) const {
    return ToPair() == other.ToPair();
  }

  bool operator!=(const CudaComputeCapability &other) const {
    return !(*this == other);
  }

  // Maximum resident blocks per multiprocessor, values taken from
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities.
  int GetMaxResidentBlocksPerSM() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_4(mht_4_v, 253, "", "./tensorflow/stream_executor/device_description.h", "GetMaxResidentBlocksPerSM");

    if (IsAtLeast(8, 6)) {
      return 16;
    } else if (IsAtLeast(8)) {
      return 32;
    } else if (IsAtLeast(7, 5)) {
      return 16;
    }
    return 32;
  }

  // Maximum resident warps per multiprocessor, values taken from
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities.
  int GetMaxResidentWarpsPerSM() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_5(mht_5_v, 269, "", "./tensorflow/stream_executor/device_description.h", "GetMaxResidentWarpsPerSM");

    if (IsAtLeast(8, 6)) {
      return 48;
    } else if (IsAtLeast(8)) {
      return 64;
    } else if (IsAtLeast(7, 5)) {
      return 32;
    }
    return 64;
  }

  std::string ToString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_6(mht_6_v, 283, "", "./tensorflow/stream_executor/device_description.h", "ToString");
 return absl::StrCat(major, ".", minor); }

  std::pair<int, int> ToPair() const { return std::make_pair(major, minor); }
};

// ROCm compute capability, as reported by the device description.
class RocmComputeCapability {
 public:
  // gcn_arch_name example --  gfx90a:sramecc+:xnack-
  // gfx_version is the "gfx90a" part of the gcn_arch_name
  explicit RocmComputeCapability(const std::string &gcn_arch_name)
      : gcn_arch_name_(gcn_arch_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("gcn_arch_name: \"" + gcn_arch_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_7(mht_7_v, 298, "", "./tensorflow/stream_executor/device_description.h", "RocmComputeCapability");
}

  ~RocmComputeCapability() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_8(mht_8_v, 303, "", "./tensorflow/stream_executor/device_description.h", "~RocmComputeCapability");
}

  std::string gcn_arch_name() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_9(mht_9_v, 308, "", "./tensorflow/stream_executor/device_description.h", "gcn_arch_name");
 return gcn_arch_name_; }

  std::string gfx_version() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_10(mht_10_v, 313, "", "./tensorflow/stream_executor/device_description.h", "gfx_version");

    std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name_, ':');
    return tokens[0];
  }

  bool is_supported_gfx_version() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_11(mht_11_v, 321, "", "./tensorflow/stream_executor/device_description.h", "is_supported_gfx_version");

    return supported_gfx_versions().count(gfx_version()) != 0;
  }

  std::string supported_gfx_versions_str() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_12(mht_12_v, 328, "", "./tensorflow/stream_executor/device_description.h", "supported_gfx_versions_str");

    return absl::StrJoin(supported_gfx_versions(), ", ");
  }

  bool has_nhwc_layout_support() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_13(mht_13_v, 335, "", "./tensorflow/stream_executor/device_description.h", "has_nhwc_layout_support");

    return gfx_versions_with_nhwc_layout_support().count(gfx_version()) != 0;
  }

  bool has_bf16_dtype_support() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_14(mht_14_v, 342, "", "./tensorflow/stream_executor/device_description.h", "has_bf16_dtype_support");

    return gfx_versions_with_fast_bf16_support().count(gfx_version()) != 0;
  }

  bool has_fast_fp16_support() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_15(mht_15_v, 349, "", "./tensorflow/stream_executor/device_description.h", "has_fast_fp16_support");

    return gfx_versions_with_fast_fp16_support().count(gfx_version()) != 0;
  }

  bool has_mfma_instr_support() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_16(mht_16_v, 356, "", "./tensorflow/stream_executor/device_description.h", "has_mfma_instr_support");

    return gfx_versions_with_mfma_instr_support().count(gfx_version()) != 0;
  }

  bool has_fp16_atomics_support() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_17(mht_17_v, 363, "", "./tensorflow/stream_executor/device_description.h", "has_fp16_atomics_support");

    return gfx_versions_with_fp16_atomics_support().count(gfx_version()) != 0;
  }

 private:
  std::string gcn_arch_name_;
  std::set<std::string> supported_gfx_versions() {
    return {
        "gfx900",  // MI25
        "gfx906",  // MI50 / MI60
        "gfx908",  // MI100
        "gfx90a",  // MI200
        "gfx1030"  // Navi21
    };
  }
  std::set<std::string> gfx_versions_with_nhwc_layout_support() {
    return {"gfx908", "gfx90a"};
  }
  std::set<std::string> gfx_versions_with_fast_bf16_support() {
    return {"gfx908", "gfx90a"};
  }
  std::set<std::string> gfx_versions_with_fast_fp16_support() {
    return {"gfx906", "gfx908", "gfx90a", "gfx1030"};
  }
  std::set<std::string> gfx_versions_with_mfma_instr_support() {
    return {"gfx908", "gfx90a"};
  }
  std::set<std::string> gfx_versions_with_fp16_atomics_support() {
    return {"gfx90a"};
  }
};

// Data that describes the execution target of the StreamExecutor, in terms of
// important logical parameters. These include dimensionality limits and
// physical parameters of interest, such as number of cores present on the
// device.
//
// Thread-safe: immutable post-initialization.
class DeviceDescription {
 public:
  // Returns the platform being run on; this value is primarily intended for
  // printing, and comes out something like "OpenCL 1.2" or "Compute Capability
  // 3.5".
  const std::string &platform_version() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_18(mht_18_v, 409, "", "./tensorflow/stream_executor/device_description.h", "platform_version");
 return platform_version_; }

  // Returns the driver version interfacing with the underlying platform. Vendor
  // dependent format.
  const std::string &driver_version() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_19(mht_19_v, 416, "", "./tensorflow/stream_executor/device_description.h", "driver_version");
 return driver_version_; }

  // Return the runtime version, if one is provided by the underlying platform.
  // Vendor dependent format / usefulness.
  const std::string &runtime_version() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_20(mht_20_v, 423, "", "./tensorflow/stream_executor/device_description.h", "runtime_version");
 return runtime_version_; }

  // Returns the name that the device reports. Vendor dependent.
  const std::string &name() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_21(mht_21_v, 429, "", "./tensorflow/stream_executor/device_description.h", "name");
 return name_; }

  // Returns the PCI bus identifier for this device, of the form
  // [domain]:[bus]:[device].[function]
  const std::string &pci_bus_id() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_22(mht_22_v, 436, "", "./tensorflow/stream_executor/device_description.h", "pci_bus_id");
 return pci_bus_id_; }

  // Returns the NUMA node associated with this device, for use in
  // determining socket locality. If the NUMA node could not be determined, -1
  // is returned.
  int numa_node() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_23(mht_23_v, 444, "", "./tensorflow/stream_executor/device_description.h", "numa_node");
 return numa_node_; }

  // Number of cores (traditional notion of core; i.e. an SM on an NVIDIA device
  // or an AMD Compute Unit.
  int core_count() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_24(mht_24_v, 451, "", "./tensorflow/stream_executor/device_description.h", "core_count");
 return core_count_; }

  // Returns the limit on the thread dimensionality values in each of the
  // respective dimensions. These limits affect what constitutes a legitimate
  // kernel launch request.
  const ThreadDim &thread_dim_limit() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_25(mht_25_v, 459, "", "./tensorflow/stream_executor/device_description.h", "thread_dim_limit");
 return thread_dim_limit_; }

  // Returns the limit on the block dimensionality values in each of the
  // respective dimensions. These limits may affect what constitutes a
  // legitimate kernel launch request.
  const BlockDim &block_dim_limit() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_26(mht_26_v, 467, "", "./tensorflow/stream_executor/device_description.h", "block_dim_limit");
 return block_dim_limit_; }

  // Returns the limit on the total number of threads that can be launched in a
  // single block; i.e. the limit on x * y * z dimensions of a ThreadDim.
  // This limit affects what constitutes a legitimate kernel launch request.
  const int64_t &threads_per_block_limit() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_27(mht_27_v, 475, "", "./tensorflow/stream_executor/device_description.h", "threads_per_block_limit");

    return threads_per_block_limit_;
  }

  // Returns the limit on the total number of threads that can be simultaneously
  // launched on a given multiprocessor.
  const int64_t &threads_per_core_limit() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_28(mht_28_v, 484, "", "./tensorflow/stream_executor/device_description.h", "threads_per_core_limit");

    return threads_per_core_limit_;
  }

  // Returns the number of threads per warp/wavefront.
  const int64_t &threads_per_warp() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_29(mht_29_v, 492, "", "./tensorflow/stream_executor/device_description.h", "threads_per_warp");
 return threads_per_warp_; }

  // Returns the limit on the total number of registers per core.
  const int64_t &registers_per_core_limit() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_30(mht_30_v, 498, "", "./tensorflow/stream_executor/device_description.h", "registers_per_core_limit");

    return registers_per_core_limit_;
  }

  // Returns the limit on the total number of registers that can be
  // simultaneously used by a block.
  const int64_t &registers_per_block_limit() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_31(mht_31_v, 507, "", "./tensorflow/stream_executor/device_description.h", "registers_per_block_limit");

    return registers_per_block_limit_;
  }

  // Returns the number of address bits available to kernel code running on the
  // platform. This affects things like the maximum allocation size and perhaps
  // types used in kernel code such as size_t.
  const int64_t &device_address_bits() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_32(mht_32_v, 517, "", "./tensorflow/stream_executor/device_description.h", "device_address_bits");
 return device_address_bits_; }

  // Returns the device memory size in bytes.
  int64_t device_memory_size() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_33(mht_33_v, 523, "", "./tensorflow/stream_executor/device_description.h", "device_memory_size");
 return device_memory_size_; }

  // Returns the device's memory bandwidth in bytes/sec.  (This is for
  // reads/writes to/from the device's own memory, not for transfers between the
  // host and device.)
  int64_t memory_bandwidth() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_34(mht_34_v, 531, "", "./tensorflow/stream_executor/device_description.h", "memory_bandwidth");
 return memory_bandwidth_; }

  // Returns the device's core clock rate in GHz.
  float clock_rate_ghz() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_35(mht_35_v, 537, "", "./tensorflow/stream_executor/device_description.h", "clock_rate_ghz");
 return clock_rate_ghz_; }

  // Returns whether ECC is enabled.
  bool ecc_enabled() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_36(mht_36_v, 543, "", "./tensorflow/stream_executor/device_description.h", "ecc_enabled");
 return ecc_enabled_; }

  // Returns the device vendor string, e.g., "NVIDIA Corporation", "Advanced
  // Micro Devices, Inc.", or "GenuineIntel".
  const std::string &device_vendor() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_37(mht_37_v, 550, "", "./tensorflow/stream_executor/device_description.h", "device_vendor");
 return device_vendor_; }

  // Returns the CUDA compute capability if we're running on the CUDA platform.
  // If a CUDA compute capability is not available, the major version will be
  // zero.
  CudaComputeCapability cuda_compute_capability() const;

  // Returns the ROCm compute capability if we're running on the ROCm platform.
  // If a ROCm compute capability is not available, the default gfx_arch will
  // be "gfx000" (which is an invalid gfx arch).
  RocmComputeCapability rocm_compute_capability() const;

  // Returns the maximum amount of shared memory present on a single core
  // (i.e. Streaming Multiprocessor on NVIDIA GPUs; Compute Unit for OpenCL
  // devices). Note that some devices, such as NVIDIA's have a configurable
  // partitioning between shared memory and L1 cache.
  int64_t shared_memory_per_core() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_38(mht_38_v, 569, "", "./tensorflow/stream_executor/device_description.h", "shared_memory_per_core");
 return shared_memory_per_core_; }

  // Returns the maximum amount of shared memory available for a single block.
  int64_t shared_memory_per_block() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_39(mht_39_v, 575, "", "./tensorflow/stream_executor/device_description.h", "shared_memory_per_block");
 return shared_memory_per_block_; }

  // TODO(leary): resident blocks per core will be useful.

  // Convenience typedef for the string-based DeviceDescription mapping.
  typedef std::map<std::string, std::string> Map;

  // Returns a mapping from readable names to readable values that describe the
  // device. This is useful for things like printing.
  std::unique_ptr<Map> ToMap() const;

  // For string values that are not available via the underlying platform, this
  // value will be provided.
  static const char *kUndefinedString;

 private:
  friend class internal::DeviceDescriptionBuilder;

  DeviceDescription();

  // For description of the following members, see the corresponding accessor
  // above.
  //
  // N.B. If another field is added, update ToMap() above.
  std::string device_vendor_;
  std::string platform_version_;
  std::string driver_version_;
  std::string runtime_version_;
  std::string pci_bus_id_;
  std::string name_;

  ThreadDim thread_dim_limit_;
  BlockDim block_dim_limit_;

  int64_t threads_per_core_limit_;
  int64_t threads_per_block_limit_;
  int64_t threads_per_warp_;

  int64_t registers_per_core_limit_;
  int64_t registers_per_block_limit_;

  int64_t device_address_bits_;
  int64_t device_memory_size_;
  int64_t memory_bandwidth_;

  // Shared memory limits on a given device.
  int64_t shared_memory_per_core_;
  int64_t shared_memory_per_block_;

  float clock_rate_ghz_;

  // CUDA "CC" major value, -1 if not available.
  CudaComputeCapability cuda_compute_capability_{-1, -1};

  // ROCm gfx arch,  "gfx000" if not available.
  RocmComputeCapability rocm_compute_capability_{"gfx000"};

  int numa_node_;
  int core_count_;
  bool ecc_enabled_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescription);
};

namespace internal {

// Helper class the builds a device description, given that it has a large
// number of fields that would be easily confused in constructor form.
class DeviceDescriptionBuilder {
 public:
  DeviceDescriptionBuilder();

  // For descriptions of the following fields, see comments on the corresponding
  // DeviceDescription::* accessors above.

  void set_device_vendor(const std::string &value) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_40(mht_40_v, 654, "", "./tensorflow/stream_executor/device_description.h", "set_device_vendor");

    device_description_->device_vendor_ = value;
  }
  void set_platform_version(const std::string &value) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_41(mht_41_v, 661, "", "./tensorflow/stream_executor/device_description.h", "set_platform_version");

    device_description_->platform_version_ = value;
  }
  void set_driver_version(const std::string &value) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_42(mht_42_v, 668, "", "./tensorflow/stream_executor/device_description.h", "set_driver_version");

    device_description_->driver_version_ = value;
  }
  void set_runtime_version(const std::string &value) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_43(mht_43_v, 675, "", "./tensorflow/stream_executor/device_description.h", "set_runtime_version");

    device_description_->runtime_version_ = value;
  }
  void set_pci_bus_id(const std::string &value) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_44(mht_44_v, 682, "", "./tensorflow/stream_executor/device_description.h", "set_pci_bus_id");

    device_description_->pci_bus_id_ = value;
  }
  void set_name(const std::string &value) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_45(mht_45_v, 689, "", "./tensorflow/stream_executor/device_description.h", "set_name");

    device_description_->name_ = value;
  }

  void set_thread_dim_limit(const ThreadDim &value) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_46(mht_46_v, 696, "", "./tensorflow/stream_executor/device_description.h", "set_thread_dim_limit");

    device_description_->thread_dim_limit_ = value;
  }
  void set_block_dim_limit(const BlockDim &value) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_47(mht_47_v, 702, "", "./tensorflow/stream_executor/device_description.h", "set_block_dim_limit");

    device_description_->block_dim_limit_ = value;
  }

  void set_threads_per_core_limit(int64_t value) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_48(mht_48_v, 709, "", "./tensorflow/stream_executor/device_description.h", "set_threads_per_core_limit");

    device_description_->threads_per_core_limit_ = value;
  }
  void set_threads_per_block_limit(int64_t value) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_49(mht_49_v, 715, "", "./tensorflow/stream_executor/device_description.h", "set_threads_per_block_limit");

    device_description_->threads_per_block_limit_ = value;
  }
  void set_threads_per_warp(int64_t value) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_50(mht_50_v, 721, "", "./tensorflow/stream_executor/device_description.h", "set_threads_per_warp");

    device_description_->threads_per_warp_ = value;
  }

  void set_registers_per_core_limit(int64_t value) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_51(mht_51_v, 728, "", "./tensorflow/stream_executor/device_description.h", "set_registers_per_core_limit");

    device_description_->registers_per_core_limit_ = value;
  }
  void set_registers_per_block_limit(int64_t value) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_52(mht_52_v, 734, "", "./tensorflow/stream_executor/device_description.h", "set_registers_per_block_limit");

    device_description_->registers_per_block_limit_ = value;
  }

  void set_device_address_bits(int64_t value) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_53(mht_53_v, 741, "", "./tensorflow/stream_executor/device_description.h", "set_device_address_bits");

    device_description_->device_address_bits_ = value;
  }
  void set_device_memory_size(int64_t value) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_54(mht_54_v, 747, "", "./tensorflow/stream_executor/device_description.h", "set_device_memory_size");

    device_description_->device_memory_size_ = value;
  }
  void set_memory_bandwidth(int64_t value) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_55(mht_55_v, 753, "", "./tensorflow/stream_executor/device_description.h", "set_memory_bandwidth");

    device_description_->memory_bandwidth_ = value;
  }

  void set_shared_memory_per_core(int64_t value) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_56(mht_56_v, 760, "", "./tensorflow/stream_executor/device_description.h", "set_shared_memory_per_core");

    device_description_->shared_memory_per_core_ = value;
  }
  void set_shared_memory_per_block(int64_t value) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_57(mht_57_v, 766, "", "./tensorflow/stream_executor/device_description.h", "set_shared_memory_per_block");

    device_description_->shared_memory_per_block_ = value;
  }

  void set_clock_rate_ghz(float value) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_58(mht_58_v, 773, "", "./tensorflow/stream_executor/device_description.h", "set_clock_rate_ghz");

    device_description_->clock_rate_ghz_ = value;
  }

  void set_cuda_compute_capability(int major, int minor) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_59(mht_59_v, 780, "", "./tensorflow/stream_executor/device_description.h", "set_cuda_compute_capability");

    device_description_->cuda_compute_capability_ =
        CudaComputeCapability{major, minor};
  }

  void set_rocm_compute_capability(std::string gcn_arch_name) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("gcn_arch_name: \"" + gcn_arch_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_60(mht_60_v, 789, "", "./tensorflow/stream_executor/device_description.h", "set_rocm_compute_capability");

    device_description_->rocm_compute_capability_ =
        RocmComputeCapability(gcn_arch_name);
  }

  void set_numa_node(int value) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_61(mht_61_v, 797, "", "./tensorflow/stream_executor/device_description.h", "set_numa_node");
 device_description_->numa_node_ = value; }
  void set_core_count(int value) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_62(mht_62_v, 801, "", "./tensorflow/stream_executor/device_description.h", "set_core_count");
 device_description_->core_count_ = value; }
  void set_ecc_enabled(bool value) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_descriptionDTh mht_63(mht_63_v, 805, "", "./tensorflow/stream_executor/device_description.h", "set_ecc_enabled");

    device_description_->ecc_enabled_ = value;
  }

  // Returns a built DeviceDescription with ownership transferred to the
  // caller. There are currently no restrictions on which fields must be set in
  // order to build the descriptor.
  //
  // Once the description is built, this builder object should be discarded.
  std::unique_ptr<DeviceDescription> Build() {
    return std::move(device_description_);
  }

 private:
  std::unique_ptr<DeviceDescription> device_description_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescriptionBuilder);
};

}  // namespace internal

// Returns whether the given thread_dim is acceptable given the limits described
// in device_description. For detailed reasons for failing the predicate, enable
// VLOG(2) for this module.
bool ThreadDimOk(const DeviceDescription &device_description,
                 const ThreadDim &thread_dim);

// Equivalent to ceil(double(element_count) / threads_per_block).
ABSL_DEPRECATED("Use MathUtil::CeilOfRatio directly instead.")
int64_t DivideCeil(int64_t x, int64_t y);

// Calculate the number of threads/blocks required to process element_count
// elements. Note that you can still end up with more threads than
// element_count due to rounding, so kernels often start with an "is this
// thread id in the element_count range?" test.
void CalculateDimensionality(const DeviceDescription &device_description,
                             int64_t element_count, int64_t *threads_per_block,
                             int64_t *block_count);

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
