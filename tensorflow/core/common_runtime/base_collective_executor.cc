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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc() {
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
#include "tensorflow/core/common_runtime/base_collective_executor.h"

#include <algorithm>
#include <functional>
#include <utility>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#define VALUE_IN_DEBUG_STRING false

namespace tensorflow {

namespace {
bool IsCancelled(CancellationManager* cancel_mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "IsCancelled");

  return cancel_mgr != nullptr &&
         (cancel_mgr->IsCancelled() || cancel_mgr->IsCancelling());
}
}  // namespace

/*static*/
int64_t CollectiveAdapter::AlignedChunkElts(int64_t elt_bytes,
                                            int64_t total_elts,
                                            int64_t num_chunks) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "CollectiveAdapter::AlignedChunkElts");

  DCHECK_GT(num_chunks, 0);
  int64_t base_chunk_elts = (total_elts + (num_chunks - 1)) / num_chunks;
  if (EIGEN_MAX_ALIGN_BYTES == 0) return base_chunk_elts;
  if (EIGEN_MAX_ALIGN_BYTES <= elt_bytes) {
    // Tolerate weird small values of EIGEN_MAX_ALIGN_BYTES
    DCHECK_EQ(0, elt_bytes % EIGEN_MAX_ALIGN_BYTES);
    return base_chunk_elts;
  }
  // elt_bytes < EIGEN_MAX_ALIGN_BYTES, which
  // must be a common multiple of the various atomic data types.
  DCHECK_EQ(0, EIGEN_MAX_ALIGN_BYTES % elt_bytes)
      << "total_elts=" << total_elts << " num_chunks=" << num_chunks
      << " EIGEN_MAX_ALIGN_BYTES=" << EIGEN_MAX_ALIGN_BYTES
      << " elt_bytes=" << elt_bytes;
  // Round bytes per chunk up to the next multiple of EIGEN_MAX_ALIGN_BYTES.
  int64_t chunk_bytes = base_chunk_elts * elt_bytes;
  int64_t diff =
      (chunk_bytes < EIGEN_MAX_ALIGN_BYTES)
          ? (EIGEN_MAX_ALIGN_BYTES - chunk_bytes)
          : (EIGEN_MAX_ALIGN_BYTES - (chunk_bytes % EIGEN_MAX_ALIGN_BYTES));
  DCHECK_EQ(0, diff % elt_bytes);
  base_chunk_elts += (diff / elt_bytes);
  DCHECK_EQ(0, ((base_chunk_elts * elt_bytes) % EIGEN_MAX_ALIGN_BYTES))
      << "total_elts=" << total_elts << " num_chunks=" << num_chunks
      << " EIGEN_MAX_ALIGN_BYTES=" << EIGEN_MAX_ALIGN_BYTES
      << " base_chunk_elts=" << base_chunk_elts << " elt_bytes=" << elt_bytes;
  return base_chunk_elts;
}

namespace {
template <typename T>
class CollectiveAdapterImpl : public CollectiveAdapter {
 public:
  // Takes ownership of output and prepares to properly alias its chunks.
  // Ownership is taken because the shape may temporarily change.
  CollectiveAdapterImpl(Tensor* output, int64_t num_chunks,
                        Allocator* allocator, bool align_chunks)
      : output_(std::move(*output)),
        dt_(output_.dtype()),
        old_shape_(output_.shape()),
        num_chunks_(num_chunks),
        allocator_(allocator),
        total_elts_(output_.NumElements()),
        chunk_elts_(align_chunks
                        ? AlignedChunkElts(sizeof(T), total_elts_, num_chunks_)
                        : total_elts_ / num_chunks_),
        data_start_(reinterpret_cast<T*>(DMAHelper::base(&output_))),
        data_end_(data_start_ + total_elts_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_2(mht_2_v, 281, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "CollectiveAdapterImpl");

    if (!align_chunks) {
      DCHECK_EQ(total_elts_, num_chunks_ * chunk_elts_);
    }
    DCHECK_GT(chunk_elts_, 0);
    Flatten();
  }

  ~CollectiveAdapterImpl() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_3(mht_3_v, 292, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "~CollectiveAdapterImpl");
}

  const Tensor& Value() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_4(mht_4_v, 297, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "Value");
 return output_; }

  // If necessary, flatten output.
  void Flatten() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_5(mht_5_v, 303, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "Flatten");

    if (old_shape_.dims() != 1) {
      TensorShape new_shape = TensorShape({old_shape_.num_elements()});
      DMAHelper::UnsafeSetShape(&output_, new_shape);
    }
  }

  void ConsumeFinalValue(Tensor* output) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_6(mht_6_v, 313, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "ConsumeFinalValue");

    if (old_shape_ != output_.shape()) {
      DMAHelper::UnsafeSetShape(&output_, old_shape_);
    }
    *output = std::move(output_);
  }

  // Number of T elements in a particular chunk.
  inline int64_t ChunkElts(int i) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_7(mht_7_v, 324, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "ChunkElts");

    DCHECK_LT(i, num_chunks_);
    const T* chunk_start = std::min(data_end_, data_start_ + i * chunk_elts_);
    const T* chunk_end = std::min(data_end_, chunk_start + chunk_elts_);
    return chunk_end - chunk_start;
  }

  int64_t ChunkBytes(int i) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_8(mht_8_v, 334, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "ChunkBytes");
 return sizeof(T) * ChunkElts(i); }

  // Returns a new Tensor that aliases the required chunk.
  Tensor ChunkAlias(int i) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_9(mht_9_v, 340, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "ChunkAlias");

    int64_t start = chunk_elts_ * i;
    int64_t num_elts = ChunkElts(i);
    // If this chunk is empty the prior chunk might also be short
    // so always take an empty slice from the front of the tensor
    // to avoid an illegal offset check failure somewhere.
    return (num_elts > 0) ? output_.Slice(start, start + num_elts)
                          : output_.Slice(0, 0);
  }

  Tensor TempChunk(int i) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_10(mht_10_v, 353, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "TempChunk");

    AllocationAttributes empty;
    profiler::ScopedMemoryDebugAnnotation op_annotation(
        "CollectiveAdapterImpl::TempChunk");
    return Tensor(allocator_, dt_, {ChunkElts(i)}, empty);
  }

  string DebugString() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_11(mht_11_v, 363, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "DebugString");

    return strings::StrCat(
        "base addr ", reinterpret_cast<int64_t>(DMAHelper::base(&output_)),
        " num_chunks ", num_chunks_, " total_elts ", total_elts_, " chunk_elts",
        chunk_elts_, " value ",
        VALUE_IN_DEBUG_STRING ? output_.SummarizeValue(1024) : "<hidden>");
  }

  string TBounds(const Tensor& t) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_12(mht_12_v, 374, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "TBounds");

    int64_t base_addr = reinterpret_cast<int64_t>(DMAHelper::base(&t));
    return strings::StrCat("(", base_addr, ", ", (base_addr + t.TotalBytes()),
                           ")");
  }

  Tensor Scalar(int v) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_13(mht_13_v, 383, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "Scalar");
 return Tensor(static_cast<T>(v)); }

  Tensor Scalar(Allocator* a, const AllocationAttributes& attr) const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_14(mht_14_v, 388, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "Scalar");

    Tensor t(a, dt_, TensorShape({}), attr);
    return t;
  }

  Tensor output_;
  const DataType dt_;
  const TensorShape old_shape_;
  const int64_t num_chunks_;
  Allocator* allocator_;
  const int64_t total_elts_;
  const int64_t chunk_elts_;
  const T* data_start_;
  const T* data_end_;
};

}  // namespace

CollectiveAdapter* MakeCollectiveAdapter(Tensor* output, int num_chunks,
                                         Allocator* allocator,
                                         bool align_chunks) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_15(mht_15_v, 411, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "MakeCollectiveAdapter");

  switch (output->dtype()) {
    case DT_BFLOAT16:
      return new CollectiveAdapterImpl<Eigen::bfloat16>(
          output, num_chunks, allocator, align_chunks);
      break;
    case DT_HALF:
      return new CollectiveAdapterImpl<Eigen::half>(output, num_chunks,
                                                    allocator, align_chunks);
      break;
    case DT_FLOAT:
      return new CollectiveAdapterImpl<float>(output, num_chunks, allocator,
                                              align_chunks);
      break;
    case DT_DOUBLE:
      return new CollectiveAdapterImpl<double>(output, num_chunks, allocator,
                                               align_chunks);
      break;
    case DT_INT32:
      return new CollectiveAdapterImpl<int32>(output, num_chunks, allocator,
                                              align_chunks);
      break;
    case DT_INT64:
      return new CollectiveAdapterImpl<int64_t>(output, num_chunks, allocator,
                                                align_chunks);
      break;
    default:
      LOG(FATAL) << "Unsupported type " << DataTypeString(output->dtype())
                 << " to MakeCollectiveAdapter";
      return nullptr;
  }
}

BaseCollectiveExecutor::~BaseCollectiveExecutor() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_16(mht_16_v, 447, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::~BaseCollectiveExecutor");
}

void BaseCollectiveExecutor::StartAbort(const Status& s) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_17(mht_17_v, 452, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::StartAbort");

  Status status;
  {
    mutex_lock l(status_mu_);
    if (!status_.ok()) {
      VLOG(2) << "BaseCollectiveExecutor already aborted, ignoring StartAbort: "
              << s;
      return;
    }
    status_ = StatusGroup::MakeDerived(Status(
        s.code(),
        absl::StrCat(
            "Collective ops is aborted by: ", s.error_message(),
            "\nThe error could be from a previous operation. Restart your "
            "program to reset.")));
    status = status_;
  }
  LOG(ERROR) << "BaseCollectiveExecutor::StartAbort " << s;
  cem_->GetParamResolver()->StartAbort(status);
  remote_access_->StartAbort(status);
  if (cem_->GetNcclCommunicator() != nullptr) {
    cem_->GetNcclCommunicator()->StartAbort(status);
  }
}

Status BaseCollectiveExecutor::GetStatus(const Status& s) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_18(mht_18_v, 480, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::GetStatus");

  if (s.ok()) return s;
  mutex_lock l(status_mu_);
  // If the collective executor is already aborted, use the aborted status
  // which is more likely the actual error instead of an artifact of an
  // abortion.
  if (!status_.ok()) {
    VLOG(2) << "Overriding status with collective ops executor status. "
               "Original status: "
            << s;
    return status_;
  }
  return s;
}

void BaseCollectiveExecutor::ExecuteAsync(OpKernelContext* ctx,
                                          const CollectiveParams* col_params,
                                          const string& exec_key,
                                          StatusCallback done) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("exec_key: \"" + exec_key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_19(mht_19_v, 502, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::ExecuteAsync");

  // See CompleteParamsAsync() how done() and the timeout callback interacts.
  const auto is_callback_called = std::make_shared<std::atomic<bool>>(false);
  auto done_safe = [this, done, ctx, is_callback_called](const Status& s) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_20(mht_20_v, 508, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "lambda");

    bool called = is_callback_called->exchange(true);
    if (!called) {
      if (!s.ok() && !IsCancelled(ctx->cancellation_manager())) {
        // This is a collective error. Abort CollectiveExecutor so that this
        // error can propagate to other workers.
        StartAbort(s);
      }
      done(GetStatus(s));
    }
  };
  auto timeout_microseconds = static_cast<int64_t>(
      col_params->instance.impl_details.timeout_seconds * 1'000'000);
  if (timeout_microseconds > 0) {
    // TODO(xldrx): Share the timeout watchdog thread among collectives.
    SchedNonBlockingClosureAfter(
        timeout_microseconds, [this, is_callback_called, done] {
          bool called = is_callback_called->exchange(true);
          if (!called) {
            Status status(error::DEADLINE_EXCEEDED,
                          "Collective has timed out during execution.");
            StartAbort(status);
            done(status);
          }
        });
  }

  Tensor* output = ctx->mutable_output(0);
  const Tensor* input = (col_params->instance.type == REDUCTION_COLLECTIVE ||
                         col_params->instance.type == GATHER_COLLECTIVE ||
                         col_params->instance.type == PERMUTE_COLLECTIVE ||
                         col_params->instance.type == ALL_TO_ALL_COLLECTIVE ||
                         (col_params->instance.type == BROADCAST_COLLECTIVE &&
                          col_params->is_source))
                            ? &ctx->input(0)
                            : nullptr;
  CollectiveImplementationInterface* col_impl = nullptr;
  Status status = CreateCollective(*col_params, &col_impl);
  if (!status.ok()) {
    done_safe(status);
    DCHECK_EQ(nullptr, col_impl);
    return;
  }
  core::ScopedUnref unref(col_impl);
  auto col_ctx = std::make_shared<CollectiveContext>(
      this, cem_->GetNcclCommunicator(), dev_mgr_, ctx, CtxParams(ctx),
      col_params, exec_key, step_id_, input, output);
  status = col_impl->InitializeCollectiveContext(col_ctx);
  if (!status.ok()) {
    done_safe(status);
    return;
  }
  // Run on an unbounded work queue that can handle blocking work so as to not
  // starve executor threads.
  col_impl->Ref();
  profiler::TraceMeProducer producer("BaseCollectiveExecutor::ExecuteAsync");
  RunClosure([col_impl, col_ctx, done_safe, ctx,
              context_id = producer.GetContextId()]() {
    core::ScopedUnref unref(col_impl);
    profiler::TraceMeConsumer consumer(
        [ctx] {
          string op = profiler::TraceMeOp(ctx->op_kernel().name_view(),
                                          ctx->op_kernel().type_string_view());
          return profiler::TraceMeEncode(std::move(op),
                                         {{"id", ctx->step_id()}});
        },
        context_id);
    col_impl->Ref();
    col_impl->Run([col_impl, col_ctx, done_safe](const Status& s) {
      core::ScopedUnref unref(col_impl);
      done_safe(s);
    });
  });
}

void BaseCollectiveExecutor::CompleteParamsAsync(
    const DeviceAttributes& device, CollectiveParams* cp,
    CancellationManager* cancel_mgr, StatusCallback done) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_21(mht_21_v, 588, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::CompleteParamsAsync");

  // We need to make sure that when the timeout callback executes,
  // CollectiveExecutor and CollectiveExecutorMgr are both alive. After done()
  // is called, CollectiveExecutorMgr may be destructed and we don't have a way
  // to keep it without making the ownerships more complicated. Therefore if the
  // timeout callback executes, done_safe will become a no-op and the timeout
  // callback is responsible for invoking done() at the end.
  const auto is_callback_called = std::make_shared<std::atomic<bool>>(false);
  auto trace_id =
      profiler::TraceMe::ActivityStart("CollectiveExecutor::CompleteParams");
  auto done_safe = [this, is_callback_called, cancel_mgr, trace_id,
                    done](const Status& s) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_22(mht_22_v, 602, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "lambda");

    profiler::TraceMe::ActivityEnd(trace_id);
    bool called = is_callback_called->exchange(true);
    if (!called) {
      if (!s.ok() && !IsCancelled(cancel_mgr)) {
        // This is a collective error. Abort CollectiveExecutor so that this
        // error can propagate to other workers.
        StartAbort(s);
      }
      done(GetStatus(s));
    }
  };
  auto timeout_microseconds = static_cast<int64_t>(
      cp->instance.impl_details.timeout_seconds * 1'000'000);
  if (timeout_microseconds > 0) {
    // TODO(xldrx): Share the timeout watchdog thread among collectives.
    SchedNonBlockingClosureAfter(
        timeout_microseconds, [this, is_callback_called, done]() {
          bool called = is_callback_called->exchange(true);
          if (!called) {
            Status status(
                error::DEADLINE_EXCEEDED,
                "Collective has timed out waiting for other workers.");
            StartAbort(status);
            done(status);
          }
        });
  }
  cem_->GetParamResolver()->CompleteParamsAsync(device, cp, cancel_mgr,
                                                done_safe);
}

Status BaseCollectiveExecutor::CreateCollective(
    const CollectiveParams& col_params,
    CollectiveImplementationInterface** col_impl) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_23(mht_23_v, 639, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::CreateCollective");

  VLOG(2) << "CreateCollective type "
          << DataTypeString(col_params.instance.data_type) << " name "
          << col_params.instance.impl_details.collective_name;
  *col_impl = nullptr;
  switch (col_params.instance.data_type) {
    case DT_BOOL:
      if (col_params.instance.type == BROADCAST_COLLECTIVE) {
        return CollectiveRegistry::Lookup(
            col_params.instance.impl_details.collective_name, col_impl);
      } else {
        return errors::Internal(
            "No collective other than broadcast supports DT_BOOL");
      }
    case DT_INT32:
      if (col_params.group.device_type == DEVICE_GPU &&
          col_params.instance.type == REDUCTION_COLLECTIVE) {
        // TODO(b/139421603): enable int32 all-reduce on GPU.
        return errors::Internal(
            "Collective all-reduce does not support datatype DT_INT32 on "
            "DEVICE_GPU");
      } else {
        return CollectiveRegistry::Lookup(
            col_params.instance.impl_details.collective_name, col_impl);
      }
    case DT_BFLOAT16:
      if (col_params.group.device_type == DEVICE_GPU &&
          col_params.instance.type == REDUCTION_COLLECTIVE) {
        return errors::Internal(
            "Collective all-reduce does not support datatype DT_BFLOAT16 on "
            "DEVICE_GPU");
      } else {
        return CollectiveRegistry::Lookup(
            col_params.instance.impl_details.collective_name, col_impl);
      }
    case DT_HALF:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT64: {
      return CollectiveRegistry::Lookup(
          col_params.instance.impl_details.collective_name, col_impl);
    }
    default:
      return errors::Internal(
          "CollectiveImplementation does not support datatype ",
          DataTypeString(col_params.instance.data_type));
  }
}

bool BaseCollectiveExecutor::CheckDependencies(
    const CollectiveParams& col_params) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_24(mht_24_v, 692, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::CheckDependencies");

  for (int32_t instance : col_params.instance.impl_details.dependencies) {
    auto find_iter = launched_.find(instance);
    if (find_iter == launched_.end() || find_iter->second != 0) {
      VLOG(1) << "Collective " << col_params.ToString()
              << " blocked by instance " << instance;
      return false;
    }
  }
  return true;
}

void BaseCollectiveExecutor::WaitForDependencies(
    const CollectiveParams& col_params) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_25(mht_25_v, 708, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::WaitForDependencies");

  mutex_lock l(launch_mu_);
  while (!CheckDependencies(col_params)) {
    launch_cv_.wait(l);
  }
  VLOG(1) << "Unblocking collective " << col_params.ToString();
}

void BaseCollectiveExecutor::UnblockDependencies(
    const CollectiveParams& col_params) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbase_collective_executorDTcc mht_26(mht_26_v, 720, "", "./tensorflow/core/common_runtime/base_collective_executor.cc", "BaseCollectiveExecutor::UnblockDependencies");

  mutex_lock l(launch_mu_);
  if (launched_.find(col_params.instance.instance_key) == launched_.end()) {
    const string& task_name =
        col_params.group.members[col_params.default_rank].task;
    const int32_t num_devices =
        col_params.group.num_devices_per_task.at(task_name);
    launched_[col_params.instance.instance_key] = num_devices;
  }
  if (--launched_[col_params.instance.instance_key] == 0) {
    VLOG(1) << "Unblocking dependencies for collective instance "
            << col_params.instance.instance_key;
    launch_cv_.notify_all();
  }
}

}  // namespace tensorflow
