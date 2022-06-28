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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc() {
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

#include "tensorflow/core/profiler/utils/xplane_schema.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kHostThreadsPlaneName = "/host:CPU";
const absl::string_view kGpuPlanePrefix = "/device:GPU:";
const absl::string_view kTpuPlanePrefix = "/device:TPU:";
// TODO(b/195582092): change it to /device:custom once all literals are
// migrated.
const absl::string_view kCustomPlanePrefix = "/device:CUSTOM:";

const absl::string_view kTpuRuntimePlaneName = "/host:TPU-runtime";
const absl::string_view kCuptiDriverApiPlaneName = "/host:CUPTI";
const absl::string_view kRoctracerApiPlaneName = "/host:ROCTRACER";
const absl::string_view kMetadataPlaneName = "/host:metadata";
const absl::string_view kTFStreamzPlaneName = "/host:tfstreamz";
const absl::string_view kPythonTracerPlaneName = "/host:python-tracer";

const absl::string_view kStepLineName = "Steps";
const absl::string_view kTensorFlowNameScopeLineName = "TensorFlow Name Scope";
const absl::string_view kTensorFlowOpLineName = "TensorFlow Ops";
const absl::string_view kXlaModuleLineName = "XLA Modules";
const absl::string_view kXlaOpLineName = "XLA Ops";
const absl::string_view kKernelLaunchLineName = "Launch Stats";
const absl::string_view kSourceLineName = "Source code";

const absl::string_view kDeviceVendorNvidia = "Nvidia";
const absl::string_view kDeviceVendorAMD = "AMD";

namespace {

constexpr int kNumHostEventTypes =
    HostEventType::kLastHostEventType - HostEventType::kFirstHostEventType + 1;

constexpr int kNumStatTypes =
    StatType::kLastStatType - StatType::kFirstStatType + 1;

using HostEventTypeMap = absl::flat_hash_map<absl::string_view, HostEventType>;
using HostEventTypeStrMap =
    absl::flat_hash_map<HostEventType, absl::string_view>;
using StatTypeMap = absl::flat_hash_map<absl::string_view, StatType>;
using StatTypeStrMap = absl::flat_hash_map<StatType, absl::string_view>;

const HostEventTypeMap& GetHostEventTypeMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "GetHostEventTypeMap");

  static auto* host_event_type_map = new HostEventTypeMap({
      {"UnknownHostEventType", kUnknownHostEventType},
      {"TraceContext", kTraceContext},
      {"SessionRun", kSessionRun},
      {"FunctionRun", kFunctionRun},
      {"RunGraph", kRunGraph},
      {"RunGraphDone", kRunGraphDone},
      {"TfOpRun", kTfOpRun},
      {"EagerExecute", kEagerKernelExecute},
      {"ExecutorState::Process", kExecutorStateProcess},
      {"ExecutorDoneCallback", kExecutorDoneCallback},
      {"MemoryAllocation", kMemoryAllocation},
      {"MemoryDeallocation", kMemoryDeallocation},
      // Performance counter related.
      {"RemotePerfCounter", kRemotePerf},
      // tf data captured function events.
      {"InstantiatedCapturedFunction::Run", kTfDataCapturedFunctionRun},
      {"InstantiatedCapturedFunction::RunWithBorrowedArgs",
       kTfDataCapturedFunctionRunWithBorrowedArgs},
      {"InstantiatedCapturedFunction::RunInstantiated",
       kTfDataCapturedFunctionRunInstantiated},
      {"InstantiatedCapturedFunction::RunAsync",
       kTfDataCapturedFunctionRunAsync},
      // Functional ops.
      {"CallOp", kCallOp},
      {"ParallelForOp", kParallelForOp},
      {"ForeverOp", kForeverOp},
      {"NumericalGradientOp-EvalRight", kNumericalGradientOpEvalRight},
      {"NumericalGradientOp-EvalLeft", kNumericalGradientOpEvalLeft},
      {"SymbolicGradientOp", kSymbolicGradientOp},
      {"RemoteCallOp", kRemoteCallOp},
      {"IfOp", kIfOp},
      {"CaseOp", kCaseOp},
      {"WhileOp-EvalCond", kWhileOpEvalCond},
      {"WhileOp-StartBody", kWhileOpStartBody},
      {"ForOp", kForOp},
      {"PartitionedCallOp", kPartitionedCallOp},
      // tf.data related.
      {"IteratorGetNextOp::DoCompute", kIteratorGetNextOp},
      {"IteratorGetNextAsOptionalOp::DoCompute", kIteratorGetNextAsOptionalOp},
      {"Iterator", kIterator},
      {"Iterator::Prefetch::Generator", kDeviceInputPipelineSecondIterator},
      {"PrefetchProduce", kPrefetchProduce},
      {"PrefetchConsume", kPrefetchConsume},
      {"ParallelInterleaveProduce", kParallelInterleaveProduce},
      {"ParallelInterleaveConsume", kParallelInterleaveConsume},
      {"ParallelInterleaveInitializeInput",
       kParallelInterleaveInitializedInput},
      {"ParallelMapProduce", kParallelMapProduce},
      {"ParallelMapConsume", kParallelMapConsume},
      {"MapAndBatchProduce", kMapAndBatchProduce},
      {"MapAndBatchConsume", kMapAndBatchConsume},
      {"ParseExampleProduce", kParseExampleProduce},
      {"ParseExampleConsume", kParseExampleConsume},
      {"ParallelBatchProduce", kParallelBatchProduce},
      {"ParallelBatchConsume", kParallelBatchConsume},
      // Batching related.
      {"BatchingSessionRun", kBatchingSessionRun},
      {"ProcessBatch", kProcessBatch},
      {"ConcatInputTensors", kConcatInputTensors},
      {"MergeInputTensors", kMergeInputTensors},
      {"ScheduleWithoutSplit", kScheduleWithoutSplit},
      {"ScheduleWithSplit", kScheduleWithSplit},
      {"ScheduleWithEagerSplit", kScheduleWithEagerSplit},
      {"ASBSQueue::Schedule", kASBSQueueSchedule},
      // TFRT related.
      {"TfrtModelRun", kTfrtModelRun},
      // JAX related.
      {"LocalExecutable::ExecuteOnLocalDevices", kExecuteOnLocalDevices},
      // GPU related.
      {"KernelLaunch", kKernelLaunch},
      {"KernelExecute", kKernelExecute},
      // TPU related.
      {"EnqueueRequestLocked", kEnqueueRequestLocked},
      {"RunProgramRequest", kRunProgramRequest},
      {"StartProgramRequest", kStartProgramRequest},
      {"HostCallbackRequest", kHostCallbackRequest},
      {"TransferH2DRequest", kTransferH2DRequest},
      {"TransferPreprocessedH2DRequest", kTransferPreprocessedH2DRequest},
      {"TransferD2HRequest", kTransferD2HRequest},
      {"TransferD2DRequest", kTransferD2DRequest},
      {"TransferD2DRemoteRequest", kTransferD2DRemoteRequest},
      {"OnDeviceSendRequest", kOnDeviceSendRequest},
      {"OnDeviceRecvRequest", kOnDeviceRecvRequest},
      {"OnDeviceSendRecvLocalRequest", kOnDeviceSendRecvLocalRequest},
      {"DoEnqueueProgram", kDoEnqueueProgram},
      {"DoEnqueueContinuationProgram", kDoEnqueueContinuationProgram},
      {"StartProgram", kStartProgram},
      {"WriteHbm", kWriteHbm},
      {"ReadHbm", kReadHbm},
      {"TpuExecuteOp", kTpuExecuteOp},
      {"CompleteCallbacks", kCompleteCallbacks},
      {"TPUPartitionedCallOp-InitializeVarOnTPU",
       kTpuPartitionedCallOpInitializeVarOnTpu},
      {"TPUPartitionedCallOp-ExecuteRemote",
       kTpuPartitionedCallOpExecuteRemote},
      {"TPUPartitionedCallOp-ExecuteLocal", kTpuPartitionedCallOpExecuteLocal},
      {"Linearize", kLinearize},
      {"Delinearize", kDelinearize},
      {"TransferBufferFromDevice-FastPath", kTransferBufferFromDeviceFastPath},
      {"tpu::System::TransferToDevice=>IssueEvent",
       kTransferToDeviceIssueEvent},
      {"tpu::System::TransferToDevice=>IssueEvent=>Done",
       kTransferToDeviceDone},
      {"tpu::System::TransferFromDevice=>IssueEvent",
       kTransferFromDeviceIssueEvent},
      {"tpu::System::TransferFromDevice=>IssueEvent=>Done",
       kTransferFromDeviceDone},
      {"tpu::System::Execute", kTpuSystemExecute},
  });
  DCHECK_EQ(host_event_type_map->size(), kNumHostEventTypes);
  return *host_event_type_map;
}

const StatTypeMap& GetStatTypeMap() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_1(mht_1_v, 355, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "GetStatTypeMap");

  static auto* stat_type_map = new StatTypeMap({
      {"UnknownStatType", kUnknownStatType},
      // TraceMe arguments.
      {"id", kStepId},
      {"parent_step_id", kParentStepId},
      {"function_step_id", kFunctionStepId},
      {"device_ordinal", kDeviceOrdinal},
      {"chip_ordinal", kChipOrdinal},
      {"node_ordinal", kNodeOrdinal},
      {"model_id", kModelId},
      {"queue_addr", kQueueAddr},
      {"queue_id", kQueueId},
      {"request_id", kRequestId},
      {"run_id", kRunId},
      {"replica_id", kReplicaId},
      {"graph_type", kGraphType},
      {"step_num", kStepNum},
      {"iter_num", kIterNum},
      {"index_on_host", kIndexOnHost},
      {"allocator_name", kAllocatorName},
      {"bytes_reserved", kBytesReserved},
      {"bytes_allocated", kBytesAllocated},
      {"bytes_available", kBytesAvailable},
      {"fragmentation", kFragmentation},
      {"peak_bytes_in_use", kPeakBytesInUse},
      {"requested_bytes", kRequestedBytes},
      {"allocation_bytes", kAllocationBytes},
      {"addr", kAddress},
      {"region_type", kRegionType},
      {"data_type", kDataType},
      {"shape", kTensorShapes},
      {"layout", kTensorLayout},
      {"kpi_name", kKpiName},
      {"kpi_value", kKpiValue},
      {"element_id", kElementId},
      {"parent_id", kParentId},
      // XPlane semantics related.
      {"_pt", kProducerType},
      {"_ct", kConsumerType},
      {"_p", kProducerId},
      {"_c", kConsumerId},
      {"_r", kIsRoot},
      {"_a", kIsAsync},
      // Device trace arguments.
      {"device_id", kDeviceId},
      {"context_id", kContextId},
      {"correlation_id", kCorrelationId},
      {"memcpy_details", kMemcpyDetails},
      {"memalloc_details", kMemallocDetails},
      {"MemFree_details", kMemFreeDetails},
      {"Memset_details", kMemsetDetails},
      {"MemoryResidency_details", kMemoryResidencyDetails},
      {"kernel_details", kKernelDetails},
      {"nvtx_range", kNVTXRange},
      {"stream", kStream},
      // Stats added when processing traces.
      {"group_id", kGroupId},
      {"flow", kFlow},
      {"step_name", kStepName},
      {"tf_op", kTfOp},
      {"hlo_op", kHloOp},
      {"hlo_category", kHloCategory},
      {"hlo_module", kHloModule},
      {"program_id", kProgramId},
      {"equation", kEquation},
      {"is_eager", kIsEager},
      {"is_func", kIsFunc},
      {"tf_function_call", kTfFunctionCall},
      {"tracing_count", kTfFunctionTracingCount},
      {"flops", kFlops},
      {"bytes_accessed", kBytesAccessed},
      {"selected_group_ids", kSelectedGroupIds},
      {"source", kSourceInfo},
      {"model_name", kModelName},
      {"model_version", kModelVersion},
      {"bytes_transferred", kBytesTransferred},
      {"queue", kDmaQueue},
      // Performance counter related.
      {"Raw Value", kRawValue},
      {"Scaled Value", kScaledValue},
      {"Thread Id", kThreadId},
      // XLA metadata map related.
      {"Hlo Proto", kHloProto},
      // Device capability related.
      {"clock_rate", kDevCapClockRateKHz},
      {"core_count", kDevCapCoreCount},
      {"memory_bandwidth", kDevCapMemoryBandwidth},
      {"memory_size", kDevCapMemorySize},
      {"compute_cap_major", kDevCapComputeCapMajor},
      {"compute_cap_minor", kDevCapComputeCapMinor},
      {"device_vendor", kDevVendor},
      // Batching related.
      {"batch_size_after_padding", kBatchSizeAfterPadding},
      {"padding_amount", kPaddingAmount},
      {"batching_input_task_size", kBatchingInputTaskSize},
      // GPU related metrics.
      {"theoretical_occupancy_pct", kTheoreticalOccupancyPct},
      {"occupancy_min_grid_size", kOccupancyMinGridSize},
      {"occupancy_suggested_block_size", kOccupancySuggestedBlockSize},
  });
  DCHECK_EQ(stat_type_map->size(), kNumStatTypes);
  return *stat_type_map;
}

const HostEventTypeStrMap& GetHostEventTypeStrMap() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_2(mht_2_v, 463, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "GetHostEventTypeStrMap");

  static auto* host_event_type_str_map = new HostEventTypeStrMap(
      gtl::ReverseMap<HostEventTypeStrMap>(GetHostEventTypeMap()));
  return *host_event_type_str_map;
}

const StatTypeStrMap& GetStatTypeStrMap() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_3(mht_3_v, 472, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "GetStatTypeStrMap");

  static auto* stat_type_str_map =
      new StatTypeStrMap(gtl::ReverseMap<StatTypeStrMap>(GetStatTypeMap()));
  return *stat_type_str_map;
}

}  // namespace

absl::string_view GetHostEventTypeStr(HostEventType event_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_4(mht_4_v, 483, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "GetHostEventTypeStr");

  return GetHostEventTypeStrMap().at(event_type);
}

absl::optional<int64_t> FindHostEventType(absl::string_view event_name) {
  if (auto event_type = gtl::FindOrNull(GetHostEventTypeMap(), event_name)) {
    return *event_type;
  }
  return absl::nullopt;
}

absl::optional<int64_t> FindTfOpEventType(absl::string_view event_name) {
  // TF op names.
  Category category = ParseTfOpFullname(event_name).category;
  switch (category) {
    case Category::kTensorFlow:
      return HostEventType::kTfOpRun;
    case Category::kTfData:
      return HostEventType::kIterator;
    default:
      return absl::nullopt;
  }
}

absl::string_view GetStatTypeStr(StatType stat_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_5(mht_5_v, 510, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "GetStatTypeStr");

  return GetStatTypeStrMap().at(stat_type);
}

absl::optional<int64_t> FindStatType(absl::string_view stat_name) {
  if (auto stat_type = gtl::FindOrNull(GetStatTypeMap(), stat_name)) {
    return *stat_type;
  }
  return absl::nullopt;
}

bool IsInternalEvent(absl::optional<int64_t> event_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_6(mht_6_v, 524, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "IsInternalEvent");

  // TODO(b/162102421): Introduce a prefix for internal event names.
  if (!event_type.has_value()) return false;
  switch (*event_type) {
    case HostEventType::kMemoryAllocation:
    case HostEventType::kMemoryDeallocation:
    case HostEventType::kPrefetchProduce:
    case HostEventType::kPrefetchConsume:
    case HostEventType::kParallelInterleaveProduce:
    case HostEventType::kParallelInterleaveConsume:
    case HostEventType::kParallelInterleaveInitializedInput:
    case HostEventType::kParallelMapProduce:
    case HostEventType::kParallelMapConsume:
    case HostEventType::kMapAndBatchProduce:
    case HostEventType::kMapAndBatchConsume:
    case HostEventType::kParseExampleProduce:
    case HostEventType::kParseExampleConsume:
      return true;
    default:
      return false;
  }
}

bool IsInternalStat(absl::optional<int64_t> stat_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_schemaDTcc mht_7(mht_7_v, 550, "", "./tensorflow/core/profiler/utils/xplane_schema.cc", "IsInternalStat");

  // TODO(b/162102421): Introduce a prefix for internal stat names.
  if (!stat_type.has_value()) return false;
  switch (*stat_type) {
    case StatType::kKernelDetails:
    case StatType::kProducerType:
    case StatType::kProducerId:
    case StatType::kConsumerType:
    case StatType::kConsumerId:
    case StatType::kIsRoot:
    case StatType::kIsAsync:
    case StatType::kFlops:
    case StatType::kBytesAccessed:
      return true;
    default:
      return false;
  }
}

}  // namespace profiler
}  // namespace tensorflow
