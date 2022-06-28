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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh() {
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

#include "absl/hash/hash.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/context_types.h"

namespace tensorflow {
namespace profiler {

// Name of XPlane that contains TraceMe events.
TF_CONST_INIT extern const absl::string_view kHostThreadsPlaneName;
// Name prefix of XPlane that contains GPU events.
TF_CONST_INIT extern const absl::string_view kGpuPlanePrefix;
// Name prefix of XPlane that contains TPU events.
TF_CONST_INIT extern const absl::string_view kTpuPlanePrefix;
// Name prefix of XPlane that contains custom device events.
TF_CONST_INIT extern const absl::string_view kCustomPlanePrefix;
// Name prefix of XPlane that contains TPU runtime events.
TF_CONST_INIT extern const absl::string_view kTpuRuntimePlaneName;
// Name of XPlane that contains CUPTI driver API generated events.
TF_CONST_INIT extern const absl::string_view kCuptiDriverApiPlaneName;
// Name of XPlane that contains Roctracer API generated events.
TF_CONST_INIT extern const absl::string_view kRoctracerApiPlaneName;
// Name of XPlane that contains profile metadata such as XLA debug info.
TF_CONST_INIT extern const absl::string_view kMetadataPlaneName;
// Name of XPlane that contains kpi related metrics.
TF_CONST_INIT extern const absl::string_view kTFStreamzPlaneName;
// Name of XPlane that contains events from python tracer.
TF_CONST_INIT extern const absl::string_view kPythonTracerPlaneName;

// Names of XLines that contain ML-level events.
TF_CONST_INIT extern const absl::string_view kStepLineName;
TF_CONST_INIT extern const absl::string_view kTensorFlowNameScopeLineName;
TF_CONST_INIT extern const absl::string_view kTensorFlowOpLineName;
TF_CONST_INIT extern const absl::string_view kXlaModuleLineName;
TF_CONST_INIT extern const absl::string_view kXlaOpLineName;
TF_CONST_INIT extern const absl::string_view kKernelLaunchLineName;
TF_CONST_INIT extern const absl::string_view kSourceLineName;

// GPU device vendors.
TF_CONST_INIT extern const absl::string_view kDeviceVendorNvidia;
TF_CONST_INIT extern const absl::string_view kDeviceVendorAMD;

// Interesting event types (i.e., TraceMe names).
enum HostEventType {
  kFirstHostEventType = 0,
  kUnknownHostEventType = kFirstHostEventType,
  kTraceContext,
  kSessionRun,
  kFunctionRun,
  kRunGraph,
  kRunGraphDone,
  kTfOpRun,
  kEagerKernelExecute,
  kExecutorStateProcess,
  kExecutorDoneCallback,
  kMemoryAllocation,
  kMemoryDeallocation,
  // Performance counter related.
  kRemotePerf,
  // tf.data captured function events.
  kTfDataCapturedFunctionRun,
  kTfDataCapturedFunctionRunWithBorrowedArgs,
  kTfDataCapturedFunctionRunInstantiated,
  kTfDataCapturedFunctionRunAsync,
  // Functional ops.
  kCallOp,
  kParallelForOp,
  kForeverOp,
  kNumericalGradientOpEvalRight,
  kNumericalGradientOpEvalLeft,
  kSymbolicGradientOp,
  kRemoteCallOp,
  kIfOp,
  kCaseOp,
  kWhileOpEvalCond,
  kWhileOpStartBody,
  kForOp,
  kPartitionedCallOp,
  // tf.data related.
  kIteratorGetNextOp,
  kIteratorGetNextAsOptionalOp,
  kIterator,
  kDeviceInputPipelineSecondIterator,
  kPrefetchProduce,
  kPrefetchConsume,
  kParallelInterleaveProduce,
  kParallelInterleaveConsume,
  kParallelInterleaveInitializedInput,
  kParallelMapProduce,
  kParallelMapConsume,
  kMapAndBatchProduce,
  kMapAndBatchConsume,
  kParseExampleProduce,
  kParseExampleConsume,
  kParallelBatchProduce,
  kParallelBatchConsume,
  // Batching related.
  kBatchingSessionRun,
  kProcessBatch,
  kConcatInputTensors,
  kMergeInputTensors,
  kScheduleWithoutSplit,
  kScheduleWithSplit,
  kScheduleWithEagerSplit,
  kASBSQueueSchedule,
  // TFRT related.
  kTfrtModelRun,
  // JAX related.
  kExecuteOnLocalDevices,
  // GPU related.
  kKernelLaunch,
  kKernelExecute,
  // TPU related
  kEnqueueRequestLocked,
  kRunProgramRequest,
  kStartProgramRequest,
  kHostCallbackRequest,
  kTransferH2DRequest,
  kTransferPreprocessedH2DRequest,
  kTransferD2HRequest,
  kTransferD2DRequest,
  kTransferD2DRemoteRequest,
  kOnDeviceSendRequest,
  kOnDeviceRecvRequest,
  kOnDeviceSendRecvLocalRequest,
  kDoEnqueueProgram,
  kDoEnqueueContinuationProgram,
  kStartProgram,
  kWriteHbm,
  kReadHbm,
  kTpuExecuteOp,
  kCompleteCallbacks,
  kTransferToDeviceIssueEvent,
  kTransferToDeviceDone,
  kTransferFromDeviceIssueEvent,
  kTransferFromDeviceDone,
  kTpuSystemExecute,
  kTpuPartitionedCallOpInitializeVarOnTpu,
  kTpuPartitionedCallOpExecuteRemote,
  kTpuPartitionedCallOpExecuteLocal,
  kLinearize,
  kDelinearize,
  kTransferBufferFromDeviceFastPath,
  kLastHostEventType = kTransferBufferFromDeviceFastPath,
};

enum StatType {
  kFirstStatType = 0,
  kUnknownStatType = kFirstStatType,
  // TraceMe arguments.
  kStepId,
  kParentStepId,
  kFunctionStepId,
  kDeviceOrdinal,
  kChipOrdinal,
  kNodeOrdinal,
  kModelId,
  kQueueId,
  kQueueAddr,
  kRequestId,
  kRunId,
  kReplicaId,
  kGraphType,
  kStepNum,
  kIterNum,
  kIndexOnHost,
  kAllocatorName,
  kBytesReserved,
  kBytesAllocated,
  kBytesAvailable,
  kFragmentation,
  kPeakBytesInUse,
  kRequestedBytes,
  kAllocationBytes,
  kAddress,
  kRegionType,
  kDataType,
  kTensorShapes,
  kTensorLayout,
  kKpiName,
  kKpiValue,
  kElementId,
  kParentId,
  // XPlane semantics related.
  kProducerType,
  kConsumerType,
  kProducerId,
  kConsumerId,
  kIsRoot,
  kIsAsync,
  // Device trace arguments.
  kDeviceId,
  kContextId,
  kCorrelationId,
  // TODO(b/176137043): These "details" should differentiate between activity
  // and API event sources.
  kMemcpyDetails,
  kMemallocDetails,
  kMemFreeDetails,
  kMemsetDetails,
  kMemoryResidencyDetails,
  kNVTXRange,
  kKernelDetails,
  kStream,
  // Stats added when processing traces.
  kGroupId,
  kFlow,
  kStepName,
  kTfOp,
  kHloOp,
  kHloCategory,
  kHloModule,
  kProgramId,
  kEquation,
  kIsEager,
  kIsFunc,
  kTfFunctionCall,
  kTfFunctionTracingCount,
  kFlops,
  kBytesAccessed,
  kSelectedGroupIds,
  kSourceInfo,
  kModelName,
  kModelVersion,
  kBytesTransferred,
  kDmaQueue,
  // Performance counter related.
  kRawValue,
  kScaledValue,
  kThreadId,
  // XLA metadata map related.
  kHloProto,
  // Device capability related.
  kDevCapClockRateKHz,
  kDevCapCoreCount,
  kDevCapMemoryBandwidth,
  kDevCapMemorySize,
  kDevCapComputeCapMajor,
  kDevCapComputeCapMinor,
  kDevVendor,
  // Batching related.
  kBatchSizeAfterPadding,
  kPaddingAmount,
  kBatchingInputTaskSize,
  // GPU occupancy metrics
  kTheoreticalOccupancyPct,
  kOccupancyMinGridSize,
  kOccupancySuggestedBlockSize,
  kLastStatType = kOccupancySuggestedBlockSize,
};

inline std::string GpuPlaneName(int32_t device_ordinal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_0(mht_0_v, 447, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "GpuPlaneName");

  return absl::StrCat(kGpuPlanePrefix, device_ordinal);
}

absl::string_view GetHostEventTypeStr(HostEventType event_type);

bool IsHostEventType(HostEventType event_type, absl::string_view event_name);

inline bool IsHostEventType(HostEventType event_type,
                            absl::string_view event_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_1(mht_1_v, 460, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "IsHostEventType");

  return GetHostEventTypeStr(event_type) == event_name;
}

absl::optional<int64_t> FindHostEventType(absl::string_view event_name);

absl::optional<int64_t> FindTfOpEventType(absl::string_view event_name);

absl::string_view GetStatTypeStr(StatType stat_type);

bool IsStatType(StatType stat_type, absl::string_view stat_name);

inline bool IsStatType(StatType stat_type, absl::string_view stat_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("stat_name: \"" + std::string(stat_name.data(), stat_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_2(mht_2_v, 476, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "IsStatType");

  return GetStatTypeStr(stat_type) == stat_name;
}

absl::optional<int64_t> FindStatType(absl::string_view stat_name);

// Returns true if the given event shouldn't be shown in the trace viewer.
bool IsInternalEvent(absl::optional<int64_t> event_type);

// Returns true if the given stat shouldn't be shown in the trace viewer.
bool IsInternalStat(absl::optional<int64_t> stat_type);

// Support for flow events:
// This class enables encoding/decoding the flow id and direction, stored as
// XStat value. The flow id are limited to 56 bits.
class XFlow {
 public:
  enum FlowDirection {
    kFlowUnspecified = 0x0,
    kFlowIn = 0x1,
    kFlowOut = 0x2,
    kFlowInOut = 0x3,
  };

  XFlow(uint64_t flow_id, FlowDirection direction,
        ContextType category = ContextType::kGeneric) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_3(mht_3_v, 504, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "XFlow");

    DCHECK_NE(direction, kFlowUnspecified);
    encoded_.parts.direction = direction;
    encoded_.parts.flow_id = flow_id;
    encoded_.parts.category = static_cast<uint64_t>(category);
  }

  // Encoding
  uint64 ToStatValue() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_4(mht_4_v, 515, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "ToStatValue");
 return encoded_.whole; }

  // Decoding
  static XFlow FromStatValue(uint64_t encoded) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_5(mht_5_v, 521, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "FromStatValue");
 return XFlow(encoded); }

  /* NOTE: absl::HashOf is not consistent across processes (some process level
   * salt is added), even different executions of the same program.
   * However we are not tracking cross-host flows, i.e. A single flow's
   * participating events are from the same XSpace. On the other hand,
   * events from the same XSpace is always processed in the same profiler
   * process. Flows from different hosts are unlikely to collide because of
   * 2^56 hash space. Therefore, we can consider this is good for now. We should
   * revisit the hash function when cross-hosts flows became more popular.
   */
  template <typename... Args>
  static uint64_t GetFlowId(Args&&... args) {
    return absl::HashOf(std::forward<Args>(args)...) & kFlowMask;
  }

  uint64_t Id() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_6(mht_6_v, 540, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "Id");
 return encoded_.parts.flow_id; }
  ContextType Category() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_7(mht_7_v, 544, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "Category");

    return GetSafeContextType(encoded_.parts.category);
  }
  FlowDirection Direction() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_8(mht_8_v, 550, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "Direction");

    return FlowDirection(encoded_.parts.direction);
  }

 private:
  explicit XFlow(uint64_t encoded) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTh mht_9(mht_9_v, 558, "", "./tensorflow/core/profiler/utils/xplane_schema.h", "XFlow");
 encoded_.whole = encoded; }
  static constexpr uint64_t kFlowMask = (1ULL << 56) - 1;

  union {
    // Encoded representation.
    uint64_t whole;
    struct {
      uint64_t direction : 2;
      uint64_t flow_id : 56;
      uint64_t category : 6;
    } parts;
  } encoded_ ABSL_ATTRIBUTE_PACKED;

  static_assert(sizeof(encoded_) == sizeof(uint64_t), "Must be 64 bits.");
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_SCHEMA_H_
