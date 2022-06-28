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
#ifndef TENSORFLOW_CORE_DATA_DATASET_UTILS_H_
#define TENSORFLOW_CORE_DATA_DATASET_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh() {
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


#include <functional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// Constant used for indicating that the argument of tf.data.Dataset.shard
// should be supplied by the auto-sharding rewrite.
constexpr int kShardHint = -1;

// The initial parallelism value before Autotune has a chance to optimize.
constexpr int kAutotuneDefaultParallelism = 16;

// Creates a resource handle with a unique name for the given resource where
// the resource is managed by the Resource Manager.
template <typename T>
Status CreateWeakHandle(OpKernelContext* ctx, T* resource,
                        const string& container_name, ResourceHandle* handle) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("container_name: \"" + container_name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/data/dataset_utils.h", "CreateWeakHandle");

  static std::atomic<int64_t> resource_id_counter(0);
  string unique_name =
      strings::StrCat(container_name, resource_id_counter.fetch_add(1));
  ResourceMgr* mgr = ctx->resource_manager();
  TF_RETURN_IF_ERROR(mgr->Create<T>(container_name, unique_name, resource));

  *handle = MakeResourceHandle(container_name, unique_name, *ctx->device(),
                               TypeIndex::Make<T>());
  return Status::OK();
}

// Creates a ref-counting resource handle for the given resource, where the
// resource is owned by the handle.
template <typename T>
Status CreateHandle(OpKernelContext* ctx, T* resource, ResourceHandle* handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/data/dataset_utils.h", "CreateHandle");

  ResourceMgr* mgr = ctx->resource_manager();
  *handle =
      ResourceHandle::MakeRefCountingHandle(resource, ctx->device()->name());
  TF_RETURN_IF_ERROR(
      mgr->CreateUnowned<T>(handle->container(), handle->name(), resource));
  return Status::OK();
}

// TODO(b/198162355): Merge this class with ResourceOpKernel.
template <typename T>
class AnonymousResourceOp : public OpKernel {
 public:
  // Creates an AnonymousResourceOp.
  // ref_counting: Determines if the Op returns a ref-counting ResourceHandle.
  // ResourceHandle. See go/tf-resource-handle-ref-count.
  // return_deleter: Determines if the Op outputs a deleter tensor in addition
  // to the resource handle tensor.
  // If the resource handle is ref-counting, a no-op deleter is returned.
  explicit AnonymousResourceOp(OpKernelConstruction* context, bool ref_counting,
                               bool return_deleter)
      : OpKernel(context),
        ref_counting_(ref_counting),
        return_deleter_(return_deleter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_2(mht_2_v, 257, "", "./tensorflow/core/data/dataset_utils.h", "AnonymousResourceOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_3(mht_3_v, 262, "", "./tensorflow/core/data/dataset_utils.h", "Compute");

    FunctionLibraryRuntime* lib;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    OP_REQUIRES_OK(
        ctx, ctx->function_library()->Clone(&flib_def, &pflr, &lib, true));
    T* resource;
    OP_REQUIRES_OK(ctx, CreateResource(ctx, std::move(flib_def),
                                       std::move(pflr), lib, &resource));

    ResourceHandle handle;
    if (ref_counting_) {
      OP_REQUIRES_OK(ctx, CreateHandle(ctx, resource, &handle));
    } else {
      OP_REQUIRES_OK(ctx, CreateWeakHandle(ctx, resource, name(), &handle));
    }
    Tensor* handle_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle_t));
    handle_t->scalar<ResourceHandle>()() = handle;

    if (return_deleter_) {
      Tensor* deleter_t;
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(1, TensorShape({}), &deleter_t, attr));
      // TODO(feyu): Consider returning an OptionalVariant.
      if (!ref_counting_) {
        // A deleter output that deletes the resource when destroyed.
        deleter_t->scalar<Variant>()() =
            ResourceDeleter(handle, ctx->resource_manager());
      }
    }
  }

 protected:
  virtual string name() = 0;

  virtual Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, T** resource) = 0;

 private:
  const bool ref_counting_;
  const bool return_deleter_;
};

// Returns Status::OK() if `expected` and `received` types match,
// errors::InvalidArgument otherwise.
Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received);

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const std::vector<Tensor>& received);

// Returns Status::OK() if `expected` and `received` shapes are compatible,
// errors::InvalidArgument otherwise.
Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received);

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<Tensor>& received);

// Dataset op level determinism policy.
class DeterminismPolicy {
 public:
  enum class Type : int {
    // The op must produce elements deterministically.
    kDeterministic,
    // The op may relax determinism to improve performance.
    kNondeterministic,
    // The determinism policy is not specified at the op level. In this case we
    // use the experimental_deterministic dataset option to determine the
    // determinism policy.
    kDefault,
  };
  static constexpr const char* const kDeterministic = "true";
  static constexpr const char* const kNondeterministic = "false";
  static constexpr const char* const kDefault = "default";

  DeterminismPolicy() : determinism_(Type::kDefault) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_4(mht_4_v, 346, "", "./tensorflow/core/data/dataset_utils.h", "DeterminismPolicy");
}
  explicit DeterminismPolicy(Type determinism) : determinism_(determinism) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_5(mht_5_v, 350, "", "./tensorflow/core/data/dataset_utils.h", "DeterminismPolicy");
}
  // Creates a DeterminismPolicy with Type kDeterministic or
  // kNondeterministic, depending on the values of `is_deterministic`.
  explicit DeterminismPolicy(bool is_deterministic);

  static Status FromString(const std::string& s, DeterminismPolicy* out);

  // Returns the string representing the determinism policy. This will be one of
  // the string constants defined above.
  std::string String() const;

  /// Convenience methods for checking the DeterminismPolicy::Type.
  bool IsDeterministic() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_6(mht_6_v, 365, "", "./tensorflow/core/data/dataset_utils.h", "IsDeterministic");
 return determinism_ == Type::kDeterministic; }
  bool IsNondeterministic() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_7(mht_7_v, 369, "", "./tensorflow/core/data/dataset_utils.h", "IsNondeterministic");

    return determinism_ == Type::kNondeterministic;
  }
  bool IsDefault() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_8(mht_8_v, 375, "", "./tensorflow/core/data/dataset_utils.h", "IsDefault");
 return determinism_ == Type::kDefault; }

 private:
  Type determinism_;
};

// Resolves non-deterministic seeds if necessary, returning either the original
// seeds or the resolved seeds.
//
// By TensorFlow convention, if both seeds are 0, they should be replaced with
// non-deterministically chosen seeds.
std::pair<int64_t, int64_t> MaybeOverrideSeeds(
    std::pair<int64_t, int64_t> seeds);

// Adds the functions in `to_add` to `base`. If a function with a matching
// signature already exists in `base`, replaces it with the function from
// `to_add`.
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add);
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add);

// Determines whether the given function is stateful.
Status IsFunctionStateful(const FunctionLibraryDefinition& library,
                          const FunctionDef& function_def);

// Determines whether the given node is stateful.
Status IsNodeStateful(const FunctionLibraryDefinition& library,
                      const NodeDef& node);

// Creates a runner that runs functions with limited parallelism.
std::function<void(std::function<void()>)> RunnerWithMaxParallelism(
    std::function<void(std::function<void()>)> runner, int max_parallelism);

// Op for creating a typed dummy resource.
//
// This op is used to provide a resource "placeholder" for ops such as
// `CacheDatasetV2` or `ShuffleDatasetV2` that expects a resource input.
// Originally, the lifetime of the resources passed into these ops was managed
// externally. After the implementation changed to manage the lifetime of the
// resources (including creation) by the ops themselves, the resource input is
// only needed to pass a resource handle through graph rewrites. When they are
// invoked from user code, the implementation passes in a dummy resource.
template <typename ResourceType>
class DummyResourceOp : public OpKernel {
 public:
  explicit DummyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_9(mht_9_v, 424, "", "./tensorflow/core/data/dataset_utils.h", "DummyResourceOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_10(mht_10_v, 429, "", "./tensorflow/core/data/dataset_utils.h", "Compute");

    Tensor* tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &tensor));
    tensor->scalar<ResourceHandle>()() = MakeResourceHandle<ResourceType>(
        ctx, /*container=*/"", /*name=*/"dummy_resource");
  }
};

// Given an op prefix and an op to match, returns whether the op to match
// is a match for any version of the op prefix. For example,
// MatchesAnyVersion("BatchDataset", "BatchDataset") == true
// MatchesAnyVersion("BatchDataset", "BatchDatasetV2") == true
// MatchesAnyVersion("BatchDataset", "BatchDatasetV3") == true
// MatchesAnyVersion("PaddedBatchDataset", "BatchDataset") == false
bool MatchesAnyVersion(StringPiece op_prefix, StringPiece op_to_match);

// Returns the index-th slice of a given tensor. If the index-th slice of
// the tensor is not aligned, returns a deep copy of the tensor.
Tensor MaybeCopySubSlice(const Tensor& tensor, int64 index);

// Removes device placements from the ops of all functions in `library`.
void StripDevicePlacement(FunctionDefLibrary* library);

// Copies partial of the batch output.
Status CopyPartialBatch(int64_t num_elements, const Tensor& value,
                        Tensor* output);

// Reads a batch when restoring the iterator.
Status ReadBatch(IteratorContext* ctx, IteratorStateReader* reader,
                 int64_t batch_size, const string& iterator_prefix,
                 const string& batch_prefix, std::vector<Tensor>* batch);

// Writes a batch when saving the iterator.
Status WriteBatch(int64_t batch_size, int64_t num_elements,
                  const string& iterator_prefix, const string& batch_prefix,
                  IteratorStateWriter* writer, std::vector<Tensor>* batch);

// Reads a status when restoring the iterator.
Status ReadStatus(const string& iterator_prefix, const string& prefix,
                  IteratorStateReader* reader, Status* status);

// Writes a status when saving the iterator.
Status WriteStatus(const string& iterator_prefix, const string& prefix,
                   const Status& status, IteratorStateWriter* writer);

// Processes a batch to output. In the case a partial batch is encountered, copy
// only partial of the batch.
Status ProcessBatch(int64_t batch_size, int64_t num_elements,
                    bool drop_remainder, const Status& status,
                    IteratorContext* ctx, std::vector<Tensor>* output,
                    bool* end_of_sequence, std::vector<Tensor>* batch);

// Constructs and stores the parameters for the CopyBatch function.
struct CopyBatchParams {
  Allocator* allocator;
  std::function<void(std::function<void()>)>* runner;
  int64 runner_threadpool_size;

  explicit CopyBatchParams(IteratorContext* ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_11(mht_11_v, 490, "", "./tensorflow/core/data/dataset_utils.h", "CopyBatchParams");

    allocator = ctx->allocator({});
    runner = ctx->runner();
    runner_threadpool_size = ctx->runner_threadpool_size();
  }

  explicit CopyBatchParams(OpKernelContext* ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_12(mht_12_v, 499, "", "./tensorflow/core/data/dataset_utils.h", "CopyBatchParams");

    allocator = ctx->get_allocator({});
    runner = ctx->runner();
    runner_threadpool_size = GetRunnerThreadpoolSizeFromOpKernelContext(ctx);
  }
};

// Copies the input elements to a batch.
//
// The `batch_elements` argument contains the individual elements to copy into a
// batch. The `parallel_copy` argument indicates whether to parallelize the
// copy. The `allocation_callback` argument can be used to pass a callback to
// invoke upon successful allocation of the memory for the batch. The
// `out_tensors` argument will be used to store the resulting batch (one for
// each component of the input).
Status CopyBatch(CopyBatchParams params,
                 const std::vector<std::vector<Tensor>>& batch_elements,
                 bool parallel_copy,
                 std::function<Status()> allocation_callback,
                 std::vector<Tensor>* out_tensors);

// Computes the set of experiments to apply based on the job name, rollout
// percentage of registered experiments, and the TF_DATA_EXPERIMENT_OPT_IN and
// TF_DATA_EXPERIMENT_OPT_OUT environment variables.
absl::flat_hash_set<string> GetExperiments();
absl::flat_hash_set<string> GetExperiments(
    const string& job_name, std::function<uint64(const string&)> hash_func);

// Logs and records the experiments that will be applied.
void LogAndRecordExperiments(const absl::flat_hash_set<string>& experiments);

// Computes the set of enabled, disabled, and default optimizations based on the
// given options. An optimization must be a graph optimizer name that has been
// registered with Grappler.
void GetOptimizations(const Options& options,
                      absl::flat_hash_set<tstring>* optimizations_enabled,
                      absl::flat_hash_set<tstring>* optimizations_disabled,
                      absl::flat_hash_set<tstring>* optimizations_default);

// Creates graph rewrite configs based on the given options. The configs will
// only be used if their corresponding optimizers registered with Grappler are
// enabled.
// A config is a string with the following format:
//   <optimizer name>:<attribute name>:<attribute value>
absl::flat_hash_set<tstring> CreateGraphRewriteConfigs(const Options& options);

// Determines whether max intra-op parallelism should be configured.
bool ShouldConfigureMaxIntraOpParallelism(const Options& options);

// Determines whether private threadpool should be used.
bool ShouldUsePrivateThreadPool(const Options& options);

// Determines whether autotuning should be used.
bool ShouldUseAutotuning(const Options& options);

// Determines whether optimizations should be applied.
bool ShouldApplyOptimizations(
    const Options& options,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_default);

// Returns the default CPU budget.
inline int GetCpuBudget() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_13(mht_13_v, 564, "", "./tensorflow/core/data/dataset_utils.h", "GetCpuBudget");

  static bool in_experiment = GetExperiments().contains("tune_cpu_budget");
  return (in_experiment ? 1.2 : 1.0) * port::NumSchedulableCPUs();
}

// Returns the initial value for parallelism parameter before the first Autotune
// optimization.
int64 GetAutotuneDefaultParallelism(IteratorContext* ctx);

// Registry of tf.data experiments.
class DatasetExperimentRegistry {
 public:
  // Registers the experiment.
  static void Register(const string& experiment, int64_t rollout_pct);

  // Returns all registered experiments.
  static absl::flat_hash_map<string, int64_t> Experiments();
};

// Helper class to register a dataset experiment.
class DatasetExperimentRegistrar {
 public:
  explicit DatasetExperimentRegistrar(const string& experiment,
                                      int64_t rollout_pct) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("experiment: \"" + experiment + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSdataset_utilsDTh mht_14(mht_14_v, 591, "", "./tensorflow/core/data/dataset_utils.h", "DatasetExperimentRegistrar");

    DatasetExperimentRegistry::Register(experiment, rollout_pct);
  }
};

// Macro that can be used to register a dataset experiment.
#define REGISTER_DATASET_EXPERIMENT(experiment, rollout_pct) \
  REGISTER_DATASET_OP_NAME_UNIQ_HELPER(__COUNTER__, experiment, rollout_pct)

#define REGISTER_DATASET_OP_NAME_UNIQ_HELPER(ctr, experiment, rollout_pct) \
  REGISTER_DATASET_OP_NAME_UNIQ(ctr, experiment, rollout_pct)

#define REGISTER_DATASET_OP_NAME_UNIQ(ctr, experiment, rollout_pct) \
  static ::tensorflow::data::DatasetExperimentRegistrar             \
      registrar__body__##ctr##__object(experiment, rollout_pct)

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_DATASET_UTILS_H_
