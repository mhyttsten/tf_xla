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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

using InitializerSerializer =
    ::tensorflow::lookup::InitializableLookupTable::InitializerSerializer;

class DatasetIterator
    : public lookup::InitializableLookupTable::InitTableIterator {
 public:
  explicit DatasetIterator(data::DatasetBase* dataset) : dataset_(dataset) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "DatasetIterator");
}

  ~DatasetIterator() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "~DatasetIterator");
}

  Status Init(OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "Init");

    data::IteratorContext::Params params(ctx);
    function_handle_cache_ = absl::make_unique<FunctionHandleCache>(params.flr);
    params.function_handle_cache = function_handle_cache_.get();
    params.resource_mgr = &resource_mgr_;
    cancellation_manager_ =
        absl::make_unique<CancellationManager>(ctx->cancellation_manager());
    params.cancellation_manager = cancellation_manager_.get();
    iterator_ctx_ = absl::make_unique<data::IteratorContext>(std::move(params));

    DatasetBase* finalized_dataset;
    TF_RETURN_IF_ERROR(
        data::FinalizeDataset(ctx, dataset_, &finalized_dataset));
    TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(
        iterator_ctx_.get(), nullptr, "LookupTable", &iterator_));
    core::ScopedUnref unref(finalized_dataset);
    Next();
    return Status::OK();
  }

  void Next() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "Next");

    bool end_of_input;
    tensors_.clear();
    status_ = iterator_->GetNext(iterator_ctx_.get(), &tensors_, &end_of_input);
    if (status_.ok() && end_of_input) {
      status_ = errors::OutOfRange("end of iterator");
    }
  }

  bool Valid() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "Valid");
 return status_.ok(); }

  const Tensor& keys() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "keys");
 return tensors_[0]; }

  const Tensor& values() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_6(mht_6_v, 276, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "values");
 return tensors_[1]; }

  Status status() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_7(mht_7_v, 281, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "status");
 return status_; }

  int64_t total_size() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_8(mht_8_v, 286, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "total_size");

    int64_t size = dataset_->Cardinality();
    if (size < 0) {
      return 0;
    }
    return size;
  }

 private:
  data::DatasetBase* dataset_;  // owned.
  std::unique_ptr<data::IteratorContext> iterator_ctx_;
  std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  ResourceMgr resource_mgr_;
  std::unique_ptr<CancellationManager> cancellation_manager_;
  std::unique_ptr<data::IteratorBase> iterator_;
  std::vector<Tensor> tensors_;
  Status status_;
};

std::unique_ptr<InitializerSerializer> MakeDatasetInitializerSerializer(
    OpKernelContext* ctx, data::DatasetBase* dataset) {
  dataset->Ref();
  auto unref_dataset = [dataset] { dataset->Unref(); };
  return absl::make_unique<InitializerSerializer>(
      [dataset, resource_manager = ctx->resource_manager(),
       device_name = ctx->device()->attributes().name()](
          GraphDefBuilder* builder, Node* table, Node** out) {
        data::DatasetBase::DatasetGraphDefBuilder db(builder);
        data::SerializationContext::Params params;
        params.resource_mgr = resource_manager;
        params.device_name = device_name;
        data::SerializationContext serialization_ctx(params);
        Node* dataset_node;
        TF_RETURN_IF_ERROR(
            db.AddInputDataset(&serialization_ctx, dataset, &dataset_node));
        *out = ops::BinaryOp("InitializeTableFromDataset", table, dataset_node,
                             builder->opts());
        if (*out == nullptr) {
          return errors::Internal(
              "Failed to create InitializeTableFromDataset op: ",
              builder->opts().StatusToString());
        }
        return Status::OK();
      },
      /*cleanup=*/std::move(unref_dataset));
}

void InitializeTableFromDataset(OpKernelContext* ctx,
                                data::DatasetBase* dataset,
                                lookup::InitializableLookupTable* table,
                                AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_9(mht_9_v, 339, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "InitializeTableFromDataset");

  // Construct the cleanup before `iter` below so that `iter` is destroyed
  // before calling `done`.
  auto cleanup = gtl::MakeCleanup([done = std::move(done)]() { done(); });
  // Assert that the dataset types match up to that expected in the table.
  const auto& dataset_types = dataset->output_dtypes();
  OP_REQUIRES(
      ctx, dataset_types.size() == 2,
      errors::InvalidArgument("Dataset should have two output types only"));
  OP_REQUIRES(ctx, dataset_types[0] == table->key_dtype(),
              errors::InvalidArgument(
                  "Key dtype expected: ", table->key_dtype(),
                  " but obtained: ", dataset_types[0], " from the dataset"));
  OP_REQUIRES(ctx, dataset_types[1] == table->value_dtype(),
              errors::InvalidArgument(
                  "Value dtype expected: ", table->value_dtype(),
                  " but obtained: ", dataset_types[1], " from the dataset"));
  // Assert that the dataset output shapes are scalars.
  const auto& dataset_shapes = dataset->output_shapes();
  OP_REQUIRES(
      ctx, dataset_shapes.size() == 2,
      errors::InvalidArgument("Dataset should have two output shapes only"));
  OP_REQUIRES(ctx, dataset_shapes[0].IsCompatibleWith(PartialTensorShape({})),
              errors::InvalidArgument("Expected scalar for key. Obtained: ",
                                      dataset_shapes[0].DebugString()));
  OP_REQUIRES(ctx, dataset_shapes[1].IsCompatibleWith(PartialTensorShape({})),
              errors::InvalidArgument("Expected scalar for key. Obtained: ",
                                      dataset_shapes[1].DebugString()));
  DatasetIterator iter(dataset);
  OP_REQUIRES_OK(ctx, iter.Init(ctx));
  Status s =
      table->Initialize(iter, MakeDatasetInitializerSerializer(ctx, dataset));
  if (errors::IsFailedPrecondition(s) && table->is_initialized()) {
    LOG(INFO) << "Table already initialized from dataset.";
    return;
  }
  ctx->SetStatus(s);
}

class InitializeTableFromDatasetOp : public AsyncOpKernel {
 public:
  explicit InitializeTableFromDatasetOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "initialize_table_from_dataset") {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_10(mht_10_v, 385, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "InitializeTableFromDatasetOp");
}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlookup_opsDTcc mht_11(mht_11_v, 390, "", "./tensorflow/core/kernels/data/experimental/lookup_ops.cc", "ComputeAsync");

    lookup::InitializableLookupTable* table;
    OP_REQUIRES_OK_ASYNC(
        ctx, GetInitializableLookupTable("table_handle", ctx, &table), done);
    core::ScopedUnref unref_me(table);
    data::DatasetBase* dataset;
    OP_REQUIRES_OK_ASYNC(
        ctx, GetDatasetFromVariantTensor(ctx->input(1), &dataset), done);
    background_worker_.Schedule([ctx, dataset, table, done]() {
      InitializeTableFromDataset(ctx, dataset, table, done);
    });
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(InitializeTableFromDatasetOp);

  data::BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(Name("InitializeTableFromDataset").Device(DEVICE_CPU),
                        InitializeTableFromDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
