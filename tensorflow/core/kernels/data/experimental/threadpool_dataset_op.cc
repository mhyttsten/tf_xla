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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/threadpool_dataset_op.h"

#include <memory>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const
    MaxIntraOpParallelismDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    MaxIntraOpParallelismDatasetOp::kDatasetOp;
/* static */ constexpr const char* const
    PrivateThreadPoolDatasetOp::kDatasetType;
/* static */ constexpr const char* const PrivateThreadPoolDatasetOp::kDatasetOp;

namespace {
// To prevent integer overflow issues when allocating threadpool memory for an
// unreasonable number of threads.
constexpr int kThreadLimit = 65536;

Status ValidateNumThreads(int32_t num_threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "ValidateNumThreads");

  if (num_threads < 0) {
    return errors::InvalidArgument("`num_threads` must be >= 0");
  }
  if (num_threads >= kThreadLimit) {
    return errors::InvalidArgument("`num_threads` must be < ", kThreadLimit);
  }
  return Status::OK();
}
}  // namespace

class ThreadPoolResource : public ResourceBase {
 public:
  ThreadPoolResource(Env* env, const ThreadOptions& thread_options,
                     const string& name, int num_threads, bool low_latency_hint,
                     int max_intra_op_parallelism)
      : thread_pool_(env, thread_options, name, num_threads, low_latency_hint),
        max_intra_op_parallelism_(max_intra_op_parallelism) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "ThreadPoolResource");
}

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Schedule");

    if (max_intra_op_parallelism_ < 0) {
      thread_pool_.Schedule(std::move(fn));
    } else {
      thread_pool_.Schedule(std::bind(
          [this](std::function<void()> bound_fn) {
            // TODO(mrry): Consider moving this thread-local configuration to
            // the threads themselves.
            ScopedPerThreadMaxParallelism scope(max_intra_op_parallelism_);
            bound_fn();
          },
          std::move(fn)));
    }
  }

  int32 NumThreads() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "NumThreads");
 return thread_pool_.NumThreads(); }

  string DebugString() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "DebugString");
 return "ThreadPoolResource"; }

 private:
  thread::ThreadPool thread_pool_;
  const int max_intra_op_parallelism_;
};

// Creates a handle to a ThreadPool resource. Note that we don't use
// ResourceOpKernel here because the ThreadPoolResource constructor requires
// access to `OpKernelContext::env()`, which isn't provided by
// `ResourceOpKernel<T>::CreateResource()`.
class ThreadPoolHandleOp : public OpKernel {
 public:
  explicit ThreadPoolHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "ThreadPoolHandleOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("display_name", &display_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_intra_op_parallelism",
                                     &max_intra_op_parallelism_));
    OP_REQUIRES_OK(ctx, ValidateNumThreads(num_threads_));
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~ThreadPoolHandleOp() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_6(mht_6_v, 296, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "~ThreadPoolHandleOp");

    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<ThreadPoolResource>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));
      ThreadPoolResource* resource;
      OP_REQUIRES_OK(ctx, mgr->LookupOrCreate<ThreadPoolResource>(
                              cinfo_.container(), cinfo_.name(), &resource,
                              [this, ctx](ThreadPoolResource** ret)
                                  TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                    *ret = new ThreadPoolResource(
                                        ctx->env(), {}, display_name_,
                                        num_threads_,
                                        /*low_latency_hint=*/false,
                                        max_intra_op_parallelism_);
                                    return Status::OK();
                                  }));
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            TypeIndex::Make<ThreadPoolResource>()));
  }

 private:
  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  bool initialized_ TF_GUARDED_BY(mu_) = false;
  string display_name_;
  int num_threads_;
  int max_intra_op_parallelism_;
};

class ThreadPoolDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ThreadPoolDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_7(mht_7_v, 345, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "ThreadPoolDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_8(mht_8_v, 351, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "MakeDataset");

    core::RefCountPtr<ThreadPoolResource> threadpool_resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1),
                                       &threadpool_resource));
    *output = new Dataset(ctx, input, ctx->input(1), threadpool_resource.get());
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const Tensor& resource_handle, ThreadPoolResource* threadpool)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          resource_handle_(resource_handle),
          threadpool_(threadpool) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_9(mht_9_v, 369, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Dataset");

      input_->Ref();
      threadpool_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_10(mht_10_v, 377, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "~Dataset");

      input_->Unref();
      threadpool_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::ThreadPool")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_11(mht_11_v, 391, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "output_dtypes");

      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_12(mht_12_v, 397, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "output_shapes");

      return input_->output_shapes();
    }

    string DebugString() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_13(mht_13_v, 404, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "DebugString");

      return "ThreadPoolDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_14(mht_14_v, 411, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CardinalityInternal");

      return input_->Cardinality();
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_15(mht_15_v, 419, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_16(mht_16_v, 427, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_17(mht_17_v, 437, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* resource_handle_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, resource_handle_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_18(mht_18_v, 454, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_19(mht_19_v, 459, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Initialize");

        return dataset()->input_->MakeIterator(
            IteratorContext(CreateParams(ctx)), this, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_20(mht_20_v, 469, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "GetNextInternal");

        return input_impl_->GetNext(IteratorContext(CreateParams(ctx)),
                                    out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_21(mht_21_v, 485, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "SaveInternal");

        DCHECK(input_impl_ != nullptr);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_22(mht_22_v, 495, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "RestoreInternal");

        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      IteratorContext::Params CreateParams(IteratorContext* ctx) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_23(mht_23_v, 504, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CreateParams");

        ThreadPoolResource* pool = dataset()->threadpool_;
        IteratorContext::Params params(ctx);
        params.runner = [pool](std::function<void()> c) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_24(mht_24_v, 510, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "lambda");

          pool->Schedule(std::move(c));
        };
        params.runner_threadpool_size = pool->NumThreads();
        return params;
      }

      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    const Tensor resource_handle_;
    ThreadPoolResource* const threadpool_;
  };
};

class MaxIntraOpParallelismDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int64_t max_intra_op_parallelism)
      : Dataset(DatasetContext(ctx), input, max_intra_op_parallelism) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_25(mht_25_v, 533, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Dataset");
}

  Dataset(DatasetContext&& ctx, const DatasetBase* input,
          int64_t max_intra_op_parallelism)
      : DatasetBase(std::move(ctx)),
        input_(input),
        max_intra_op_parallelism_(max_intra_op_parallelism),
        traceme_metadata_(
            {{"parallelism",
              strings::Printf("%lld", static_cast<long long>(
                                          max_intra_op_parallelism_))}}) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_26(mht_26_v, 546, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Dataset");

    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_27(mht_27_v, 553, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, strings::StrCat(prefix, "::MaxIntraOpParallelism")});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_28(mht_28_v, 564, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_29(mht_29_v, 570, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_30(mht_30_v, 577, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "DebugString");

    return "MaxIntraOpParallelismDatasetOp::Dataset";
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_31(mht_31_v, 584, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CardinalityInternal");
 return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_32(mht_32_v, 589, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "InputDatasets");

    inputs->clear();
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_33(mht_33_v, 598, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_34(mht_34_v, 608, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* max_intra_op_parallelism_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(max_intra_op_parallelism_,
                                    &max_intra_op_parallelism_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, max_intra_op_parallelism_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_35(mht_35_v, 626, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_36(mht_36_v, 631, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Initialize");

      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_37(mht_37_v, 640, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "GetNextInternal");

      IteratorContext::Params params(ctx);
      auto max_parallelism = dataset()->max_intra_op_parallelism_;
      params.runner = RunnerWithMaxParallelism(*ctx->runner(), max_parallelism);
      return input_impl_->GetNext(IteratorContext{std::move(params)},
                                  out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_38(mht_38_v, 658, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "SaveInternal");

      DCHECK(input_impl_ != nullptr);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_39(mht_39_v, 668, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "RestoreInternal");

      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_40(mht_40_v, 676, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "GetTraceMeMetadata");

      return dataset()->traceme_metadata_;
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* const input_;
  const int64_t max_intra_op_parallelism_;
  const TraceMeMetadata traceme_metadata_;
};

/* static */
void MaxIntraOpParallelismDatasetOp::MakeDatasetFromOptions(
    OpKernelContext* ctx, DatasetBase* input, int32_t max_intra_op_parallelism,
    DatasetBase** output) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_41(mht_41_v, 695, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "MaxIntraOpParallelismDatasetOp::MakeDatasetFromOptions");

  OP_REQUIRES(
      ctx, max_intra_op_parallelism >= 0,
      errors::InvalidArgument("`max_intra_op_parallelism` must be >= 0"));
  *output = new Dataset(DatasetContext(DatasetContext::Params(
                            {MaxIntraOpParallelismDatasetOp::kDatasetType,
                             MaxIntraOpParallelismDatasetOp::kDatasetOp})),
                        input, max_intra_op_parallelism);
}

void MaxIntraOpParallelismDatasetOp::MakeDataset(OpKernelContext* ctx,
                                                 DatasetBase* input,
                                                 DatasetBase** output) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_42(mht_42_v, 710, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "MaxIntraOpParallelismDatasetOp::MakeDataset");

  int64_t max_intra_op_parallelism;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, "max_intra_op_parallelism",
                                              &max_intra_op_parallelism));
  OP_REQUIRES(
      ctx, max_intra_op_parallelism >= 0,
      errors::InvalidArgument("`max_intra_op_parallelism` must be >= 0"));
  *output = new Dataset(ctx, input, max_intra_op_parallelism);
}

class PrivateThreadPoolDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int num_threads)
      : Dataset(ctx, DatasetContext(ctx), input, num_threads) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_43(mht_43_v, 727, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Dataset");
}

  Dataset(OpKernelContext* ctx, DatasetContext&& dataset_ctx,
          const DatasetBase* input, int num_threads)
      : DatasetBase(std::move(dataset_ctx)),
        input_(input),
        num_threads_(num_threads == 0 ? port::MaxParallelism() : num_threads),
        traceme_metadata_(
            {{"num_threads",
              strings::Printf("%lld", static_cast<long long>(num_threads_))}}) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_44(mht_44_v, 739, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Dataset");

    thread_pool_ = absl::make_unique<thread::ThreadPool>(
        ctx->env(), ThreadOptions{}, "data_private_threadpool", num_threads_);
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_45(mht_45_v, 748, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::PrivateThreadPool")});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_46(mht_46_v, 759, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_47(mht_47_v, 765, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_48(mht_48_v, 772, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "DebugString");

    return "PrivateThreadPoolDatasetOp::Dataset";
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_49(mht_49_v, 779, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CardinalityInternal");
 return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_50(mht_50_v, 784, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "InputDatasets");

    inputs->clear();
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_51(mht_51_v, 793, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_52(mht_52_v, 803, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* num_threads_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_threads_, &num_threads_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, num_threads_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_53(mht_53_v, 820, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_54(mht_54_v, 825, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "Initialize");

      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_55(mht_55_v, 834, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "GetNextInternal");

      thread::ThreadPool* pool = dataset()->thread_pool_.get();
      IteratorContext::Params params(ctx);
      params.runner = [pool](std::function<void()> c) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_56(mht_56_v, 840, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "lambda");

        pool->Schedule(std::move(c));
      };
      params.runner_threadpool_size = dataset()->num_threads_;
      return input_impl_->GetNext(IteratorContext{std::move(params)},
                                  out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_57(mht_57_v, 858, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "SaveInternal");

      DCHECK(input_impl_ != nullptr);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_58(mht_58_v, 868, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "RestoreInternal");

      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_59(mht_59_v, 876, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "GetTraceMeMetadata");

      return dataset()->traceme_metadata_;
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* const input_;
  const int64_t num_threads_;
  const TraceMeMetadata traceme_metadata_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

/* static */
void PrivateThreadPoolDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                                        DatasetBase* input,
                                                        int32_t num_threads,
                                                        DatasetBase** output) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_60(mht_60_v, 897, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "PrivateThreadPoolDatasetOp::MakeDatasetFromOptions");

  OP_REQUIRES_OK(ctx, ValidateNumThreads(num_threads));
  *output = new Dataset(ctx,
                        DatasetContext(DatasetContext::Params(
                            {PrivateThreadPoolDatasetOp::kDatasetType,
                             PrivateThreadPoolDatasetOp::kDatasetOp})),
                        input, num_threads);
}

void PrivateThreadPoolDatasetOp::MakeDataset(OpKernelContext* ctx,
                                             DatasetBase* input,
                                             DatasetBase** output) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSthreadpool_dataset_opDTcc mht_61(mht_61_v, 911, "", "./tensorflow/core/kernels/data/experimental/threadpool_dataset_op.cc", "PrivateThreadPoolDatasetOp::MakeDataset");

  int64_t num_threads = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<int64_t>(ctx, "num_threads", &num_threads));
  OP_REQUIRES_OK(ctx, ValidateNumThreads(num_threads));
  *output = new Dataset(ctx, input, num_threads);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("MaxIntraOpParallelismDataset").Device(DEVICE_CPU),
                        MaxIntraOpParallelismDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalMaxIntraOpParallelismDataset").Device(DEVICE_CPU),
    MaxIntraOpParallelismDatasetOp);

REGISTER_KERNEL_BUILDER(Name("PrivateThreadPoolDataset").Device(DEVICE_CPU),
                        PrivateThreadPoolDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalPrivateThreadPoolDataset").Device(DEVICE_CPU),
    PrivateThreadPoolDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ThreadPoolHandle").Device(DEVICE_CPU),
                        ThreadPoolHandleOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalThreadPoolHandle").Device(DEVICE_CPU),
                        ThreadPoolHandleOp);

REGISTER_KERNEL_BUILDER(Name("ThreadPoolDataset").Device(DEVICE_CPU),
                        ThreadPoolDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalThreadPoolDataset").Device(DEVICE_CPU),
    ThreadPoolDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
