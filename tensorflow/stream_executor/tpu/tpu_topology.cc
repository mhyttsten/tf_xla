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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc() {
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

#include "tensorflow/stream_executor/tpu/tpu_topology.h"

#include "tensorflow/core/tpu/tpu_api.h"

namespace tensorflow {
namespace tpu {

TpuDimensionsExternal TpuCoreLocationExternal::chip_coordinates() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_0(mht_0_v, 192, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuCoreLocationExternal::chip_coordinates");

  int x, y, z;
  tpu::ExecutorApiFn()->TpuCoreLocation_ChipCoordinatesFn(core_location_, &x,
                                                          &y, &z);
  return {x, y, z};
}

TpuDimensionsExternal TpuCoreLocationExternal::host_coordinates() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_1(mht_1_v, 202, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuCoreLocationExternal::host_coordinates");

  int x, y, z;
  tpu::ExecutorApiFn()->TpuCoreLocation_HostCoordinatesFn(core_location_, &x,
                                                          &y, &z);
  return {x, y, z};
}

int32 TpuCoreLocationExternal::index() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_2(mht_2_v, 212, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuCoreLocationExternal::index");

  return tpu::ExecutorApiFn()->TpuCoreLocation_IndexFn(core_location_);
}

int32 TpuCoreLocationExternal::Id() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_3(mht_3_v, 219, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuCoreLocationExternal::Id");

  return tpu::ExecutorApiFn()->TpuCoreLocation_IdFn(core_location_);
}

int32 TpuHostLocationExternal::Id() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_4(mht_4_v, 226, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuHostLocationExternal::Id");

  return tpu::ExecutorApiFn()->TpuHostLocation_IdFn(host_location_);
}

std::vector<TpuCoreLocationExternal> TpuHostLocationExternal::Cores(
    TpuCoreTypeEnum core_type) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_5(mht_5_v, 234, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuHostLocationExternal::Cores");

  int num_cores = tpu::ExecutorApiFn()->TpuHostLocation_NumCoresFn(
      host_location_, core_type);
  std::vector<SE_TpuTopology_Core*> core_ptrs(num_cores);
  tpu::ExecutorApiFn()->TpuHostLocation_CoresFn(host_location_, core_type,
                                                core_ptrs.data());
  std::vector<TpuCoreLocationExternal> result;
  result.reserve(num_cores);
  for (SE_TpuTopology_Core* ptr : core_ptrs) {
    result.emplace_back(ptr);
  }
  return result;
}

int32 TpuTopologyExternal::LogicalDevicesPerHost(
    TpuCoreTypeEnum core_type) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_6(mht_6_v, 252, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::LogicalDevicesPerHost");

  return tpu::ExecutorApiFn()->TpuTopology_LogicalDevicesPerHostFn(topology_,
                                                                   core_type);
}

int32 TpuTopologyExternal::LogicalDevicesPerChip(
    TpuCoreTypeEnum core_type) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_7(mht_7_v, 261, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::LogicalDevicesPerChip");

  return tpu::ExecutorApiFn()->TpuTopology_LogicalDevicesPerChipFn(topology_,
                                                                   core_type);
}

int32 TpuTopologyExternal::HostCount() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_8(mht_8_v, 269, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::HostCount");

  return tpu::ExecutorApiFn()->TpuTopology_HostCountFn(topology_);
}

int32 TpuTopologyExternal::ChipsPerHost() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_9(mht_9_v, 276, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::ChipsPerHost");

  return tpu::ExecutorApiFn()->TpuTopology_ChipsPerHostFn(topology_);
}

TpuTopologyChipBoundsExternal TpuTopologyExternal::chip_bounds() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_10(mht_10_v, 283, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::chip_bounds");

  return {tpu::ExecutorApiFn()->TpuTopology_ChipBounds_XFn(topology_),
          tpu::ExecutorApiFn()->TpuTopology_ChipBounds_YFn(topology_),
          tpu::ExecutorApiFn()->TpuTopology_ChipBounds_ZFn(topology_)};
}

bool TpuTopologyExternal::HasChip(int x, int y, int z) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_11(mht_11_v, 292, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::HasChip");

  return tpu::ExecutorApiFn()->TpuTopology_HasChipFn(topology_, x, y, z);
}

TpuCoreLocationExternal TpuTopologyExternal::CoreForId(
    TpuCoreTypeEnum core_type, int id) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_12(mht_12_v, 300, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::CoreForId");

  return TpuCoreLocationExternal(
      tpu::ExecutorApiFn()->TpuTopology_CoreForIdFn(topology_, core_type, id));
}

TpuCoreLocationExternal TpuTopologyExternal::Core(TpuCoreTypeEnum core_type,
                                                  int x, int y, int z,
                                                  int index) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_13(mht_13_v, 310, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::Core");

  return TpuCoreLocationExternal(tpu::ExecutorApiFn()->TpuTopology_CoreFn(
      topology_, core_type, x, y, z, index));
}

std::vector<TpuCoreLocationExternal> TpuTopologyExternal::cores(
    TpuCoreTypeEnum core_type) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_14(mht_14_v, 319, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::cores");

  int num_cores =
      tpu::ExecutorApiFn()->TpuTopology_NumCoresFn(topology_, core_type);
  std::vector<SE_TpuTopology_Core*> core_ptrs(num_cores);
  tpu::ExecutorApiFn()->TpuTopology_CoresFn(topology_, core_type,
                                            core_ptrs.data());
  std::vector<TpuCoreLocationExternal> result;
  result.reserve(num_cores);
  for (SE_TpuTopology_Core* ptr : core_ptrs) {
    result.emplace_back(ptr);
  }
  return result;
}

int TpuTopologyExternal::IdForHost(TpuDimensionsExternal host) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_15(mht_15_v, 336, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::IdForHost");

  return tpu::ExecutorApiFn()->TpuTopology_IdForHostFn(topology_, host.x,
                                                       host.y, host.z);
}

TpuVersionEnum TpuTopologyExternal::version() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_16(mht_16_v, 344, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuTopologyExternal::version");

  return tpu::ExecutorApiFn()->TpuTopology_VersionFn(topology_);
}

std::string TpuVersionEnumToString(TpuVersionEnum version) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_topologyDTcc mht_17(mht_17_v, 351, "", "./tensorflow/stream_executor/tpu/tpu_topology.cc", "TpuVersionEnumToString");

  switch (version) {
    case kUnknownTpuVersion:
      return "Unknown TPU version";
    case kTpuV2:
      return "TPU v2";
    case kTpuV3:
      return "TPU v3";
    case kTpuV4:
      return "TPU v4";
  }
}

}  // namespace tpu
}  // namespace tensorflow
