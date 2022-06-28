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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/iterator_ops.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/finalization_utils.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/optional_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.

const char kAnonymousIterator[] = "AnonymousIterator";
const char kAnonymousIteratorV2[] = "AnonymousIteratorV2";
const char kAnonymousIteratorV3[] = "AnonymousIteratorV3";
const char kIteratorVariantTypeName[] = "tensorflow::Iterator";
const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

// Safely subtracts x from y avoiding underflow.
inline uint64 safe_sub(uint64 x, uint64 y) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_0(mht_0_v, 249, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "safe_sub");
 return x >= y ? x - y : 0; }

}  // namespace

/* static */ constexpr const char* const
    SerializeIteratorOp::kExternalStatePolicy;

IteratorResource::IteratorResource(
    Env* env, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes,
    std::unique_ptr<DeviceMgr> device_mgr,
    std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* flr)
    : unbounded_thread_pool_(env, "tf_data_iterator_resource"),
      device_mgr_(std::move(device_mgr)),
      iterator_state_(std::make_shared<State>(std::move(flib_def),
                                              std::move(pflr), flr,
                                              /*iterator=*/nullptr)),
      output_dtypes_(output_dtypes),
      output_shapes_(output_shapes),
      // We do not collect iterator resource metrics for non-CPU devices. This
      // is a heuristic to avoid collecting metrics for device-side iterators
      // created by the multi-device iterator mechanism.
      collect_metrics_(flr->device()->device_type() == DEVICE_CPU) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_1(mht_1_v, 276, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorResource::IteratorResource");

  VLOG(2) << "creating iterator resource";
}

IteratorResource::~IteratorResource() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_2(mht_2_v, 283, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorResource::~IteratorResource");

  VLOG(2) << "destroying iterator resource";
}

Status IteratorResource::GetNext(OpKernelContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_3(mht_3_v, 292, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorResource::GetNext");

  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  if (!captured_state->iterator()) {
    return errors::FailedPrecondition(
        "GetNext() failed because the iterator has not been initialized. "
        "Ensure that you have run the initializer operation for this iterator "
        "before getting the next element.");
  }
  IteratorContext::Params params(ctx);
  params.flr = captured_state->flr();
  params.function_handle_cache = captured_state->function_handle_cache();
  params.resource_mgr = captured_state->resource_mgr();
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.cancellation_manager = captured_state->cancellation_manager();
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
  const uint64 start_time_us = ctx->env()->NowMicros();
  if (collect_metrics_) {
    mutex_lock l(mu_);
    if (get_next_end_time_us_ == 0) {
      // We initialize `get_next_end_time_us_` to the start time of the first
      // request to make it possible to use the delta between
      // `get_next_end_time_us_` and subsequent `GetNext()` end time to
      // incrementally collect the duration of the iterator's lifetime.
      get_next_end_time_us_ = start_time_us;
    }
    uint64 gap_time_us = 0;
    if (num_get_next_calls_ == 0) {
      get_next_start_time_us_ = start_time_us;
      gap_time_us = safe_sub(start_time_us, get_next_end_time_us_);
    }
    metrics::RecordTFDataIteratorGap(gap_time_us);
    num_get_next_calls_++;
  }
  auto iterator_ = captured_state->iterator();
  auto status = iterator_->GetNext(IteratorContext(std::move(params)),
                                   out_tensors, end_of_sequence);
  if (collect_metrics_) {
    const uint64 end_time_us = ctx->env()->NowMicros();
    AddLatencySample(safe_sub(end_time_us, start_time_us));
    IncrementThroughput(GetTotalBytes(*out_tensors));
    mutex_lock l(mu_);
    metrics::RecordTFDataIteratorLifetime(
        safe_sub(end_time_us, get_next_end_time_us_));
    get_next_end_time_us_ = std::max(get_next_end_time_us_, end_time_us);
    num_get_next_calls_--;
    if (num_get_next_calls_ == 0) {
      metrics::RecordTFDataIteratorBusy(
          safe_sub(get_next_end_time_us_, get_next_start_time_us_));
    }
  }
  return status;
}

Status IteratorResource::Save(SerializationContext* ctx,
                              IteratorStateWriter* writer) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_4(mht_4_v, 359, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorResource::Save");

  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  auto iterator_ = captured_state->iterator();
  if (iterator_) {
    return iterator_->Save(ctx, writer);
  }
  return errors::FailedPrecondition(
      "Save() failed because the iterator has not been initialized. Ensure "
      "that you have run the initializer operation for this iterator before "
      "saving it.");
}

Status IteratorResource::Restore(OpKernelContext* ctx,
                                 IteratorStateReader* reader) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_5(mht_5_v, 379, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorResource::Restore");

  const DatasetBase* dataset;
  std::shared_ptr<State> new_state;
  const DatasetBase* input_dataset;
  {
    tf_shared_lock l(mu_);
    if (!iterator_state_->iterator()) {
      return errors::FailedPrecondition(
          "Restore() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before restoring it.");
    }
    auto iterator_ = iterator_state_->iterator();
    dataset = iterator_->dataset();
    // Hang onto a reference until we've created the new iterator, which will
    // then hold its own reference to keep the dataset alive.
    dataset->Ref();
    new_state =
        std::make_shared<State>(iterator_state_->flib_def(),
                                iterator_state_->pflr(), iterator_state_->flr(),
                                /*iterator=*/nullptr);
    input_dataset = iterator_state_->dataset();
  }
  core::ScopedUnref scoped_unref(dataset);
  IteratorContext::Params params(ctx);
  params.flr = new_state->flr();
  params.function_handle_cache = new_state->function_handle_cache();
  params.resource_mgr = new_state->resource_mgr();
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.cancellation_manager = new_state->cancellation_manager();
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
  std::unique_ptr<IteratorBase> iterator_base;
  TF_RETURN_IF_ERROR(dataset->MakeIteratorFromCheckpoint(
      IteratorContext(std::move(params)), "Iterator", reader, &iterator_base));
  new_state->DowncastAndSetIteratorAndDataset(std::move(iterator_base),
                                              input_dataset);

  mutex_lock l(mu_);
  std::swap(iterator_state_, new_state);
  return Status::OK();
}

Status IteratorResource::SetIteratorFromDataset(OpKernelContext* ctx,
                                                const DatasetBase* dataset) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_6(mht_6_v, 431, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorResource::SetIteratorFromDataset");

  std::shared_ptr<State> new_state;
  {
    tf_shared_lock l(mu_);
    new_state =
        std::make_shared<State>(iterator_state_->flib_def(),
                                iterator_state_->pflr(), iterator_state_->flr(),
                                /*iterator=*/nullptr);
  }

  // Create new iterator.
  IteratorContext::Params params(ctx);
  params.flr = new_state->flr();
  params.function_handle_cache = new_state->function_handle_cache();
  params.resource_mgr = new_state->resource_mgr();
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.cancellation_manager = new_state->cancellation_manager();
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));

  std::unique_ptr<IteratorBase> iterator;
  if (ctx->function_library()->device()->device_type() == DEVICE_CPU) {
    DatasetBase* finalized_dataset;
    TF_ASSIGN_OR_RETURN(finalized_dataset, GetFinalizedDataset(ctx, dataset));
    TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(
        IteratorContext(std::move(params)),
        /*parent=*/nullptr, "Iterator", &iterator));
  } else {
    TF_RETURN_IF_ERROR(dataset->MakeIterator(IteratorContext(std::move(params)),
                                             /*parent=*/nullptr, "Iterator",
                                             &iterator));
  }
  TF_RETURN_IF_ERROR(
      VerifyTypesMatch(output_dtypes_, iterator->output_dtypes()));
  TF_RETURN_IF_ERROR(
      VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));

  new_state->DowncastAndSetIteratorAndDataset(std::move(iterator), dataset);

  mutex_lock l(mu_);
  std::swap(iterator_state_, new_state);
  return Status::OK();
}

namespace {

// Wrapper for encoding/decoding the iterator state stored in a Variant tensor.
// The get() method returns an VariantTensorData object which contains all the
// state needed to restore a single iterator.
//
// Usage example:
//
// Encoding:
//
//   Tensor t(DT_VARIANT, TensorShape({}));
//   t->scalar<Variant>()() = IteratorStateVariant();
//
// Encode() sets the type_name of the VariantTensorData object to
// IteratorStateVariant::TypeName().
//
// Decoding:
//
//   Variant v = <VariantTensorDataProto object>;
//   DecodeUnaryVariant(&v);
//   IteratorStateVariant* wrapper = v.get<IteratorStateVariant>();
//   IteratorStateReader reader({wrapper->GetData()});
//   iterator_resource->Restore(ctx, &reader);
//
// The type_name of the VariantTensorData object to be decoded must
// match IteratorStateVariant::TypeName().
class IteratorStateVariant {
 public:
  IteratorStateVariant() : data_(nullptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_7(mht_7_v, 511, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorStateVariant");
}
  IteratorStateVariant(const IteratorStateVariant& other) : data_(nullptr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_8(mht_8_v, 515, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorStateVariant");

    if (other.data_) {
      Decode(*other.data_);
    }
  }
  IteratorStateVariant& operator=(IteratorStateVariant&& other) = default;
  IteratorStateVariant& operator=(const IteratorStateVariant& other) = delete;

  // Initializes `this` from a VariantTensorData object.
  Status InitializeFromVariantData(std::unique_ptr<VariantTensorData> d) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_9(mht_9_v, 527, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "InitializeFromVariantData");

    data_ = std::move(d);
    return Status::OK();
  }

  string TypeName() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_10(mht_10_v, 535, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "TypeName");
 return kIteratorVariantTypeName; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_11(mht_11_v, 539, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "Encode");
 *data = *data_; }
  bool Decode(VariantTensorData data) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_12(mht_12_v, 543, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "Decode");

    if (data.type_name() != TypeName()) {
      return false;
    }
    auto tensor_data = absl::make_unique<VariantTensorData>();
    std::swap(*tensor_data, data);
    data_ = std::move(tensor_data);
    return true;
  }

  // Returns a borrowed pointer to the underlying VariantTensorData.
  const VariantTensorData* GetData() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_13(mht_13_v, 557, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "GetData");
 return data_.get(); }

  string DebugString() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_14(mht_14_v, 562, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "DebugString");

    if (data_) {
      return strings::StrCat("IteratorStateVariant<", data_->DebugString(),
                             ">");
    } else {
      return strings::StrCat("IteratorStateVariant<empty>");
    }
  }

 private:
  std::unique_ptr<VariantTensorData> data_;
};

// Register the reader class in the global variant decode_fn registry
// so that a Variant containing a serialized representation of iterator state
// can be decoded using DecodeUnaryVariant. If we don't do this we will need
// to manually decode the returned Variant using MaybeDecodeAndCopy in
// DeserializeIteratorOp which is not recommended.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(IteratorStateVariant,
                                       kIteratorVariantTypeName);

// A helper class that uses a list of IteratorStateVariant objects to represent
// the state for an iterator resource. It exposes methods that help with
// saving and restoring of this state. Sample usage
// Saving:
//   IteratorVariantSerializer serializer;
//   serializer.InitializeFromIterator(iterator_resource);
//   Tensor serialized_t;
//   serializer.Serialize(&serialized_t);
//
// Restoring:
//   IteratorVariantSerializer serializer;
//   serializer.InitFromTensor(ctx->input(0));
//   IteratorStateReader* reader = serializer.GetReader();
//   iterator_resource->Restore(ctx, reader);
class IteratorVariantSerializer {
 public:
  IteratorVariantSerializer() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_15(mht_15_v, 602, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorVariantSerializer");
}

  // Calls `Save` on the iterator_resource to build up the list of
  // IteratorStateVariant objects.
  Status InitializeFromIterator(SerializationContext* serialization_ctx,
                                IteratorResource* iterator_resource) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_16(mht_16_v, 610, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "InitializeFromIterator");

    VariantTensorDataWriter writer;
    TF_RETURN_IF_ERROR(iterator_resource->Save(serialization_ctx, &writer));
    std::vector<std::unique_ptr<VariantTensorData>> data;
    writer.ReleaseData(&data);
    variants_.clear();
    variants_.reserve(data.size());
    for (auto& it : data) {
      IteratorStateVariant v;
      TF_RETURN_IF_ERROR(v.InitializeFromVariantData(std::move(it)));
      variants_.push_back(v);
    }
    num_tensors_ = variants_.size();
    can_serialize_ = true;
    return Status::OK();
  }

  // Initializes `this` from `serialized_t` while restoring the iterator state.
  Status InitFromTensor(const Tensor* serialized_t) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_17(mht_17_v, 631, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "InitFromTensor");

    int64_t num_tensors = serialized_t->dim_size(0);
    auto serialized_vec = serialized_t->vec<Variant>();
    std::vector<const VariantTensorData*> data;
    data.reserve(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      auto* w = serialized_vec(i).get<IteratorStateVariant>();
      if (!w) {
        return errors::Internal(
            "Cannot initialize an iterator from tensor ",
            serialized_vec(i).DebugString(),
            ". Expected a variant tensor of type IteratorStateVariant");
      }
      data.push_back(w->GetData());
    }
    reader_ = absl::make_unique<VariantTensorDataReader>(data);
    num_tensors_ = data.size();
    return Status::OK();
  }

  int64_t NumTensors() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_18(mht_18_v, 654, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "NumTensors");
 return num_tensors_; }

  // Stores the IteratorStateVariant list into a pre-allocated tensor. Expects
  // that InitializeFromIterator was called before.
  Status Serialize(Tensor* serialized) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_19(mht_19_v, 661, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "Serialize");

    if (!can_serialize_) {
      return errors::InvalidArgument(
          "Please call InitializeFromIterator before calling Serialize.");
    }
    int64_t size = variants_.size();
    for (int64_t i = 0; i < size; ++i) {
      if (variants_[i].GetData() == nullptr) {
        return errors::Internal(
            "Cannot serialize an empty IteratorStateVariant");
      }
      serialized->vec<Variant>()(i) = variants_[i];
    }
    return Status::OK();
  }

  // Returns an IteratorStateReader to restore iterator state. Expects that
  // InitFromTensor was called before.
  IteratorStateReader* GetReader() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_20(mht_20_v, 682, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "GetReader");
 return reader_.get(); }

 private:
  bool can_serialize_ = false;
  int64_t num_tensors_;
  std::vector<IteratorStateVariant> variants_;
  std::unique_ptr<IteratorStateReader> reader_;
};

}  // namespace

// Note that IteratorHandleOp holds a reference to the resource it creates. If
// cleaning up resources with DestroyResourceOp is important, consider creating
// resource containers with AnonymousIteratorHandleOp instead.
IteratorHandleOp::IteratorHandleOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_21(mht_21_v, 700, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorHandleOp::IteratorHandleOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
}

// The resource is deleted from the resource manager only when it is private
// to kernel. Ideally the resource should be deleted when it is no longer held
// by anyone, but it would break backward compatibility.
IteratorHandleOp::~IteratorHandleOp() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_22(mht_22_v, 712, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorHandleOp::~IteratorHandleOp");

  if (resource_ != nullptr) {
    resource_->Unref();
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<IteratorResource>(cinfo_.container(),
                                                   cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }
}

void IteratorHandleOp::Compute(OpKernelContext* context)
    TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(mu_);
    if (resource_ == nullptr) {
      FunctionLibraryRuntime* flr;
      std::unique_ptr<DeviceMgr> device_mgr(nullptr);
      std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
      // If the iterator is shared then we construct a new FLR, and pass that
      // in. NOTE(mrry,rohanj): In this case it is not possible to call remote
      // functions from the iterator. We may add this functionality if there
      // is sufficient demand, but it will require a significant refactoring.
      if (!name_.empty()) {
        flr = CreatePrivateFLR(context, &device_mgr, &flib_def, &pflr);
      } else {
        OP_REQUIRES_OK(context, context->function_library()->Clone(
                                    &flib_def, &pflr, &flr, true));
      }

      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

      IteratorResource* resource;
      OP_REQUIRES_OK(
          context,
          mgr->LookupOrCreate<IteratorResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [context, flr, &device_mgr, &flib_def, &pflr,
               this](IteratorResource** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                *ret = new IteratorResource(
                    context->env(), output_dtypes_, output_shapes_,
                    std::move(device_mgr), std::move(flib_def), std::move(pflr),
                    flr);
                return Status::OK();
              }));

      Status s = VerifyResource(resource);
      if (TF_PREDICT_FALSE(!s.ok())) {
        resource->Unref();
        context->SetStatus(s);
        return;
      }

      resource_ = resource;
    }
  }
  OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                              context, 0, cinfo_.container(), cinfo_.name(),
                              TypeIndex::Make<IteratorResource>()));
}

Status IteratorHandleOp::VerifyResource(IteratorResource* resource) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_23(mht_23_v, 781, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorHandleOp::VerifyResource");

  TF_RETURN_IF_ERROR(
      VerifyTypesMatch(output_dtypes_, resource->output_dtypes()));
  TF_RETURN_IF_ERROR(
      VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
  return Status::OK();
}

FunctionLibraryRuntime* IteratorHandleOp::CreatePrivateFLR(
    OpKernelContext* ctx, std::unique_ptr<DeviceMgr>* device_mgr,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* pflr) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_24(mht_24_v, 795, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorHandleOp::CreatePrivateFLR");

  // Wrap the existing device in order to see any captured resources
  // in its resource manager. The existing device will outlive the
  // IteratorResource, because we are storing the IteratorResource
  // in that device's resource manager.

  *device_mgr =
      absl::make_unique<StaticDeviceMgr>(RenamedDevice::NewRenamedDevice(
          ctx->device()->name(), down_cast<Device*>(ctx->device()),
          false /* owns_underlying */, false /* isolate_session_state */));
  *flib_def = absl::make_unique<FunctionLibraryDefinition>(
      *ctx->function_library()->GetFunctionLibraryDefinition());
  const auto* config = ctx->function_library()->config_proto();
  *pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr->get(), ctx->env(),
      /*config=*/config, graph_def_version_, flib_def->get(),
      config->graph_options().optimizer_options());

  return (*pflr)->GetFLR(ctx->device()->name());
}

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
AnonymousIteratorHandleOp::AnonymousIteratorHandleOp(
    OpKernelConstruction* context)
    : AnonymousResourceOp<IteratorResource>(
          context,
          /* ref_counting */
          // Only enable this for V2 (via Python's iter protocol),
          // AnonymousIteratorV1 requires IteratorToStringHandle, which is
          // undefined on Refcounting ResourceHandle.
          context->def().op() == kAnonymousIteratorV2 ||
              context->def().op() == kAnonymousIteratorV3,
          // V1 does not return a deleter.
          /* return_deleter */
          context->def().op() == kAnonymousIteratorV2),
      graph_def_version_(context->graph_def_version()) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_25(mht_25_v, 836, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "AnonymousIteratorHandleOp::AnonymousIteratorHandleOp");

  OP_REQUIRES_OK(context, context->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(context, context->GetAttr(kOutputShapes, &output_shapes_));
}

string AnonymousIteratorHandleOp::name() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_26(mht_26_v, 844, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "AnonymousIteratorHandleOp::name");
 return kAnonymousIterator; }

Status AnonymousIteratorHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, IteratorResource** resource) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_27(mht_27_v, 852, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "AnonymousIteratorHandleOp::CreateResource");

  std::unique_ptr<DeviceMgr> device_mgr(nullptr);
  *resource = new IteratorResource(ctx->env(), output_dtypes_, output_shapes_,
                                   std::move(device_mgr), std::move(flib_def),
                                   std::move(pflr), lib);
  return Status::OK();
}

HybridAsyncOpKernel::HybridAsyncOpKernel(OpKernelConstruction* ctx,
                                         const char* background_worker_name)
    : AsyncOpKernel(ctx),
      background_worker_(ctx->env(), background_worker_name) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("background_worker_name: \"" + (background_worker_name == nullptr ? std::string("nullptr") : std::string((char*)background_worker_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_28(mht_28_v, 867, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "HybridAsyncOpKernel::HybridAsyncOpKernel");
}

void HybridAsyncOpKernel::ComputeAsync(OpKernelContext* ctx,
                                       DoneCallback done) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_29(mht_29_v, 873, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "HybridAsyncOpKernel::ComputeAsync");

  background_worker_.Schedule([this, ctx, done = std::move(done)]() {
    ctx->SetStatus(DoCompute(ctx));
    done();
  });
}

void HybridAsyncOpKernel::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_30(mht_30_v, 883, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "HybridAsyncOpKernel::Compute");

  ctx->SetStatus(DoCompute(ctx));
}

Status MakeIteratorOp::DoCompute(OpKernelContext* ctx) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_31(mht_31_v, 890, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "MakeIteratorOp::DoCompute");

  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  IteratorResource* iterator_resource;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  return iterator_resource->SetIteratorFromDataset(ctx, dataset);
}

Status DeleteIteratorOp::DoCompute(OpKernelContext* ctx) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_32(mht_32_v, 905, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "DeleteIteratorOp::DoCompute");

  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  const ResourceHandle& handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The iterator resource is guaranteed to exist because the variant tensor
  // wrapping the deleter is provided as an unused input to this op, which
  // guarantees that it has not run yet.
  return DeleteResource(ctx, handle);
}

namespace {

class ToSingleElementOp : public AsyncOpKernel {
 public:
  explicit ToSingleElementOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        unbounded_threadpool_(ctx->env(), "tf_data_to_single_element") {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_33(mht_33_v, 924, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "ToSingleElementOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_34(mht_34_v, 932, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "ComputeAsync");

    unbounded_threadpool_.Schedule([this, ctx, done = std::move(done)]() {
      ctx->SetStatus(DoCompute(ctx));
      done();
    });
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_35(mht_35_v, 942, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "Compute");

    ctx->SetStatus(DoCompute(ctx));
  }

 private:
  Status DoCompute(OpKernelContext* ctx) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_36(mht_36_v, 950, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "DoCompute");

    profiler::TraceMe traceme(
        [&] {
          return profiler::TraceMeEncode("ToSingleElementOp::DoCompute",
                                         {{"id", ctx->step_id()}});
        },
        profiler::kInfo);
    tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                   ctx->op_kernel().type_string());
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    IteratorContext::Params params(ctx);
    ResourceMgr resource_mgr;
    params.resource_mgr = &resource_mgr;
    CancellationManager cancellation_manager(ctx->cancellation_manager());
    params.cancellation_manager = &cancellation_manager;

    IteratorContext iter_ctx(std::move(params));
    std::unique_ptr<IteratorBase> iterator;
    TF_RETURN_IF_ERROR(dataset->MakeIterator(
        &iter_ctx, /*parent=*/nullptr, "SingleElementIterator", &iterator));

    std::vector<Tensor> components;
    components.reserve(dataset->output_dtypes().size());
    bool end_of_sequence = false;

    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &components, &end_of_sequence));

    if (end_of_sequence) {
      return errors::InvalidArgument("Dataset was empty.");
    }
    TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, components));
    TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, components));
    for (int i = 0; i < components.size(); ++i) {
      ctx->set_output(i, components[i]);
    }

    components.clear();
    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &components, &end_of_sequence));
    if (!end_of_sequence) {
      return errors::InvalidArgument("Dataset had more than one element.");
    }
    return Status::OK();
  }

  UnboundedThreadPool unbounded_threadpool_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class OneShotIteratorOp : public AsyncOpKernel {
 public:
  explicit OneShotIteratorOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_one_shot_iterator"),
        graph_def_version_(ctx->graph_def_version())

  {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_37(mht_37_v, 1013, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "OneShotIteratorOp");

    string shared_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name));
    OP_REQUIRES(ctx, shared_name.empty(),
                errors::InvalidArgument("OneShotIteratorOp does not currently "
                                        "support the 'shared_name' attr."));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("dataset_factory", &dataset_factory_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

  ~OneShotIteratorOp() override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_38(mht_38_v, 1028, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "~OneShotIteratorOp");

    if (iterator_resource_ != nullptr) {
      iterator_resource_->Unref();
      if (!cinfo_.resource_manager()
               ->Delete<IteratorResource>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  // NOTE(mrry): This is based on `ResourceOpKernel<T>::Compute()`,
  // but due to the fact that `ResourceOpKernel<T>::CreateResource()`
  // does not provide access to the `OpKernelContext*` and we need
  // this to invoke the factory function, it's not possible to
  // implement this kernel by implementing `CreateResource()`.
  // Furthermore, due to the fact that this kernel might block when
  // running the initialization function, we must implement this
  // kernel as an async kernel.
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_39(mht_39_v, 1050, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "ComputeAsync");

    tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                   ctx->op_kernel().type_string());
    {
      mutex_lock l(mu_);
      if (iterator_resource_ == nullptr && initialization_status_.ok()) {
        // The initialization thread will call `done`.
        if (!initialization_started_) {
          // TODO(mrry): Convert the initialization code to use
          // callbacks instead of wasting a thread.
          background_worker_.Schedule([this, ctx, done]() { Init(ctx, done); });
          initialization_started_ = true;
        } else {
          done_callbacks_.emplace_back(ctx, std::move(done));
        }
        return;
      }
    }
    ProduceOutput(ctx, done);
  }

 private:
  void Init(OpKernelContext* ctx, const DoneCallback& done) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_40(mht_40_v, 1075, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "Init");

    IteratorResource* iterator = nullptr;
    ContainerInfo cinfo;
    Status s = TryInit(ctx, &iterator, &cinfo);

    std::vector<std::pair<OpKernelContext*, DoneCallback>> callbacks_to_run;
    {
      mutex_lock l(mu_);
      if (s.ok()) {
        iterator_resource_ = iterator;
        cinfo_ = cinfo;
      }
      initialization_status_ = s;
      std::swap(done_callbacks_, callbacks_to_run);
    }

    for (auto&& ctx_done : callbacks_to_run) {
      ProduceOutput(ctx_done.first, ctx_done.second);
    }
    ProduceOutput(ctx, done);
  }

  Status TryInit(OpKernelContext* ctx, IteratorResource** iterator,
                 ContainerInfo* cinfo) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_41(mht_41_v, 1101, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "TryInit");

    TF_RETURN_IF_ERROR(cinfo->Init(ctx->resource_manager(), def()));

    FunctionLibraryRuntime* flr;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    TF_RETURN_IF_ERROR(
        ctx->function_library()->Clone(&flib_def, &pflr, &flr, true));

    // Create an IteratorResource that will hold the iterator for this op.
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()->LookupOrCreate<IteratorResource>(
            cinfo->container(), cinfo->name(), iterator,
            [ctx, flr, this, &flib_def, &pflr](IteratorResource** ret)
                TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                  *ret = new IteratorResource(
                      ctx->env(), output_dtypes_, output_shapes_,
                      /*device_mgr=*/nullptr, std::move(flib_def),
                      std::move(pflr), flr);
                  return Status::OK();
                }));

    core::ScopedUnref unref_iterator(*iterator);

    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, (*iterator)->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, (*iterator)->output_shapes()));

    // Call the dataset_factory_func_ to create a new dataset,
    // over which this op will iterate.
    FunctionLibraryRuntime::Handle f_handle;
    TF_RETURN_IF_ERROR(ctx->function_library()->Instantiate(
        dataset_factory_func_.name(), AttrSlice(&dataset_factory_func_.attr()),
        &f_handle));
    FunctionLibraryRuntime::Options opts;
    opts.cancellation_manager = ctx->cancellation_manager();
    ScopedStepContainer step_container(opts.step_id, [ctx](const string& name) {
      ctx->resource_manager()->Cleanup(name).IgnoreError();
    });
    opts.step_container = &step_container;
    opts.runner = ctx->runner();
    opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
    std::vector<Tensor> return_values;
    TF_RETURN_IF_ERROR(ctx->function_library()->RunSync(
        std::move(opts), f_handle, {}, &return_values));
    if (return_values.size() != 1 || return_values[0].dtype() != DT_VARIANT ||
        !TensorShapeUtils::IsScalar(return_values[0].shape())) {
      return errors::InvalidArgument(
          "The `dataset_factory` function must return "
          "a single scalar of dtype DT_VARIANT.");
    }

    // Create an iterator for the dataset that was created in the
    // factory function.
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(return_values[0], &dataset));
    TF_RETURN_IF_ERROR((*iterator)->SetIteratorFromDataset(ctx, dataset));
    (*iterator)->Ref();
    return Status::OK();
  }

  void ProduceOutput(OpKernelContext* ctx, const DoneCallback& done) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_42(mht_42_v, 1166, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "ProduceOutput");

    Tensor* handle;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &handle),
                         done);
    Status s;
    {
      mutex_lock l(mu_);
      s = initialization_status_;
      if (s.ok()) {
        handle->scalar<ResourceHandle>()() =
            MakeResourceHandle<IteratorResource>(ctx, cinfo_.container(),
                                                 cinfo_.name());
      }
    }
    OP_REQUIRES_OK_ASYNC(ctx, s, done);
    done();
  }

  NameAttrList dataset_factory_func_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;

  BackgroundWorker background_worker_;

  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  IteratorResource* iterator_resource_ TF_GUARDED_BY(mu_) = nullptr;

  bool initialization_started_ TF_GUARDED_BY(mu_) = false;
  Status initialization_status_ TF_GUARDED_BY(mu_);
  std::vector<std::pair<OpKernelContext*, DoneCallback>> done_callbacks_
      TF_GUARDED_BY(mu_);
  const int graph_def_version_;
};

}  // namespace

AsyncOpKernel* IteratorGetNextOp::AsAsync() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_43(mht_43_v, 1206, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorGetNextOp::AsAsync");

  return type_string() == "IteratorGetNextSync" ? nullptr : this;
}

void RecordElementSize(const std::vector<Tensor> element,
                       profiler::TraceMe* traceme) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_44(mht_44_v, 1214, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "RecordElementSize");

  traceme->AppendMetadata([&]() {
    int64_t element_size = 0;
    for (const auto& component : element) {
      element_size += component.TotalBytes();
    }
    return profiler::TraceMeEncode({{"element_size", element_size}});
  });
}

Status IteratorGetNextOp::DoCompute(OpKernelContext* ctx) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_45(mht_45_v, 1227, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorGetNextOp::DoCompute");

  VLOG(3) << "IteratorGetNextOp enter. iter_id=" << ctx->frame_iter().iter_id;
  auto cleanup = gtl::MakeCleanup([ctx] {
    VLOG(3) << "IteratorGetNextOp exit. iter_id=" << ctx->frame_iter().iter_id;
  });
  profiler::TraceMe traceme(
      [&] {
        return profiler::TraceMeEncode(
            "IteratorGetNextOp::DoCompute",
            {{"id", ctx->step_id()}, {"iter_num", ctx->frame_iter().iter_id}});
      },
      profiler::kInfo);
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  IteratorResource* iterator;
  TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
  core::ScopedUnref unref_iterator(iterator);
  std::vector<Tensor> components;
  bool end_of_sequence = false;

  TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &components, &end_of_sequence));
  if (end_of_sequence) {
    return errors::OutOfRange("End of sequence");
  }
  TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, components));
  TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, components));
  RecordElementSize(components, &traceme);
  for (int i = 0; i < components.size(); ++i) {
    ctx->set_output(i, components[i]);
  }
  return Status::OK();
}

Status IteratorGetNextAsOptionalOp::DoCompute(OpKernelContext* ctx) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_46(mht_46_v, 1263, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorGetNextAsOptionalOp::DoCompute");

  VLOG(3) << "IteratorGetNextAsOptionalOp exit. iter_id="
          << ctx->frame_iter().iter_id;
  auto cleanup = gtl::MakeCleanup([ctx] {
    VLOG(3) << "IteratorGetNextAsOptionalOp exit. iter_id="
            << ctx->frame_iter().iter_id;
  });
  profiler::TraceMe traceme(
      [&] {
        return profiler::TraceMeEncode(
            "IteratorGetNextAsOptionalOp::DoCompute",
            {{"id", ctx->step_id()}, {"iter_num", ctx->frame_iter().iter_id}});
      },
      profiler::kInfo);
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  IteratorResource* iterator;
  TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
  core::ScopedUnref unref_iterator(iterator);
  std::vector<Tensor> components;
  bool end_of_sequence = false;

  TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &components, &end_of_sequence));

  if (end_of_sequence) {
    return WriteOptionalNoneToOutput(ctx, 0);
  } else {
    RecordElementSize(components, &traceme);
    for (int i = 0; i < components.size(); ++i) {
      if (components[i].dtype() != output_types_[i]) {
        return errors::InvalidArgument(
            "The given optional does not match the expected type for "
            "component ",
            i, ". Expected: ", DataTypeString(output_types_[i]),
            ". Actual: ", DataTypeString(components[i].dtype()), ".");
      }
      if (!output_shapes_[i].IsCompatibleWith(components[i].shape())) {
        return errors::InvalidArgument(
            "The given optional does not match the expected shape "
            "for component ",
            i, ". Expected: ", output_shapes_[i].DebugString(),
            ". Actual: ", components[i].shape().DebugString(), ".");
      }
    }
    return WriteOptionalWithValueToOutput(ctx, 0, std::move(components));
  }
}

void IteratorToStringHandleOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_47(mht_47_v, 1314, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorToStringHandleOp::Compute");

  const Tensor& resource_handle_t = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
              errors::InvalidArgument("resource_handle must be a scalar"));

  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
  iterator_resource->Unref();

  Tensor* string_handle_t;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({}), &string_handle_t));
  string_handle_t->scalar<tstring>()() =
      resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
}

IteratorFromStringHandleOp::IteratorFromStringHandleOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_48(mht_48_v, 1338, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorFromStringHandleOp::IteratorFromStringHandleOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES(
      ctx,
      output_dtypes_.empty() || output_shapes_.empty() ||
          output_dtypes_.size() == output_shapes_.size(),
      errors::InvalidArgument("If both 'output_types' and 'output_shapes' "
                              "are set, they must have the same length."));
}

void IteratorFromStringHandleOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_49(mht_49_v, 1352, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "IteratorFromStringHandleOp::Compute");

  const Tensor& string_handle_t = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
              errors::InvalidArgument("string_handle must be a scalar"));

  ResourceHandle resource_handle;
  OP_REQUIRES(
      ctx, resource_handle.ParseFromString(string_handle_t.scalar<tstring>()()),
      errors::InvalidArgument(
          "Could not parse string_handle as a valid ResourceHandle"));

  OP_REQUIRES(
      ctx, resource_handle.device() == ctx->device()->attributes().name(),
      errors::InvalidArgument("Attempted create an iterator on device \"",
                              ctx->device()->attributes().name(),
                              "\" from handle defined on device \"",
                              resource_handle.device(), "\""));

  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, resource_handle, &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  if (!output_dtypes_.empty()) {
    OP_REQUIRES_OK(ctx, VerifyTypesMatch(output_dtypes_,
                                         iterator_resource->output_dtypes()));
  }
  if (!output_shapes_.empty()) {
    OP_REQUIRES_OK(ctx,
                   VerifyShapesCompatible(output_shapes_,
                                          iterator_resource->output_shapes()));
  }

  Tensor* resource_handle_t;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
  resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
}

SerializeIteratorOp::SerializeIteratorOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_50(mht_50_v, 1395, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "SerializeIteratorOp::SerializeIteratorOp");

  if (ctx->HasAttr(kExternalStatePolicy)) {
    int64_t state_change_option;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr(kExternalStatePolicy, &state_change_option));
    external_state_policy_ =
        SerializationContext::ExternalStatePolicy(state_change_option);
  }
}

void SerializeIteratorOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_51(mht_51_v, 1408, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "SerializeIteratorOp::Compute");

  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  const Tensor& resource_handle_t = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
              errors::InvalidArgument("resource_handle must be a scalar"));
  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  IteratorVariantSerializer serializer;
  SerializationContext::Params params(ctx);
  params.external_state_policy = external_state_policy_;
  SerializationContext serialization_ctx(params);
  OP_REQUIRES_OK(ctx, serializer.InitializeFromIterator(&serialization_ctx,
                                                        iterator_resource));
  Tensor* serialized_t;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({serializer.NumTensors()}),
                                      &serialized_t));
  OP_REQUIRES_OK(ctx, serializer.Serialize(serialized_t));
}

void DeserializeIteratorOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSiterator_opsDTcc mht_52(mht_52_v, 1436, "", "./tensorflow/core/kernels/data/iterator_ops.cc", "DeserializeIteratorOp::Compute");

  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  const Tensor* serialized_t;
  OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized_t));
  IteratorVariantSerializer serializer;
  OP_REQUIRES_OK(ctx, serializer.InitFromTensor(serialized_t));
  Status s = iterator_resource->Restore(ctx, serializer.GetReader());
  if (!s.ok()) {
    OP_REQUIRES_OK(
        ctx,
        errors::CreateWithUpdatedMessage(
            s, absl::StrCat(
                   "Failed to restore dataset iterator from checkpoint: ",
                   s.error_message(),
                   ". Make sure the dataset definition has not changed between "
                   "the process that saved the checkpoint and the process that "
                   "is restoring it.")));
  }
}

namespace {

REGISTER_KERNEL_BUILDER(Name("Iterator").Device(DEVICE_CPU), IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorV2").Device(DEVICE_CPU).Priority(2),
                        IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorV2").Device(DEVICE_GPU).Priority(1),
                        IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU).Priority(2),
                        MakeIteratorOp);
REGISTER_KERNEL_BUILDER(
    Name("MakeIterator").Device(DEVICE_GPU).Priority(1).HostMemory("dataset"),
    MakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeleteIterator").Device(DEVICE_CPU).Priority(2),
                        DeleteIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeleteIterator").Device(DEVICE_GPU).Priority(1),
                        DeleteIteratorOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIterator").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIterator").Device(DEVICE_GPU).Priority(1),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV2").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV2").Device(DEVICE_GPU).Priority(1),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV3").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV3").Device(DEVICE_GPU).Priority(1),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("DatasetToSingleElement").Device(DEVICE_CPU),
                        ToSingleElementOp);
REGISTER_KERNEL_BUILDER(Name("OneShotIterator").Device(DEVICE_CPU),
                        OneShotIteratorOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_CPU).Priority(2),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_GPU).Priority(1),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextSync").Device(DEVICE_CPU).Priority(2),
    IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextSync").Device(DEVICE_GPU).Priority(1),
    IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextAsOptional").Device(DEVICE_CPU).Priority(2),
    IteratorGetNextAsOptionalOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextAsOptional").Device(DEVICE_GPU).Priority(1),
    IteratorGetNextAsOptionalOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorToStringHandle").Device(DEVICE_CPU).Priority(2),
    IteratorToStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorToStringHandle")
                            .Device(DEVICE_GPU)
                            .HostMemory("string_handle")
                            .Priority(1),
                        IteratorToStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandle").Device(DEVICE_CPU),
                        IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorFromStringHandleV2").Device(DEVICE_CPU).Priority(2),
    IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandleV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("string_handle")
                            .Priority(1),
                        IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("SerializeIterator").Device(DEVICE_CPU),
                        SerializeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeserializeIterator").Device(DEVICE_CPU),
                        DeserializeIteratorOp);

}  // namespace

}  // namespace data
}  // namespace tensorflow
