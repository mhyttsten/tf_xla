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
class MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc() {
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

#include "tensorflow/core/data/standalone.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace standalone {

namespace {

OpKernelContext::Params CreateParams(
    ProcessFunctionLibraryRuntime* pflr, DeviceMgr* device_mgr,
    std::function<void(std::function<void()>)>* runner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/data/standalone.cc", "CreateParams");

  OpKernelContext::Params params;
  params.function_library = pflr->GetFLR("/device:CPU:0");
  params.device = device_mgr->ListDevices()[0];
  params.runner = runner;
  return params;
}

}  // namespace

Status Iterator::GetNext(std::vector<Tensor>* outputs, bool* end_of_input) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/data/standalone.cc", "Iterator::GetNext");

  return iterator_->GetNext(ctx_.get(), outputs, end_of_input);
}

Iterator::Iterator(IteratorBase* iterator, IteratorContext* ctx)
    : iterator_(iterator), ctx_(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/data/standalone.cc", "Iterator::Iterator");
}

Status Dataset::FromGraph(Params params, const GraphDef& graph_def,
                          std::unique_ptr<Dataset>* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/data/standalone.cc", "Dataset::FromGraph");

  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));

  // Instantiate enough of the TF runtime to run `graph` on a single CPU device.
  auto device_mgr = absl::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
      "CPU", params.session_options, "/job:localhost/replica:0/task:0"));
  Device* device = device_mgr->ListDevices()[0];
  // Create a copy of the `FunctionLibraryDefinition` to extend lifetime beyond
  // the lifetime of `graph`.
  auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), graph_def.library());
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, flib_def.get(), OptimizerOptions{},
      /*thread_pool=*/nullptr, /*parent=*/nullptr,
      /*session_metadata=*/nullptr,
      Rendezvous::Factory{
          [](const int64_t, const DeviceMgr* device_mgr, Rendezvous** r) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/data/standalone.cc", "lambda");

            *r = new IntraProcessRendezvous(device_mgr);
            return Status::OK();
          }});

  string fetch_node = "";
  for (const auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      fetch_node = node.input(0);
    }
  }
  if (fetch_node.empty()) {
    return errors::NotFound("Failed to find a _Retval op in the given dataset");
  }

  // Run graph up to `output_node` and extract the `DatasetBase` stored in the
  // DT_VARIANT output tensor.
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(device);
  TF_RETURN_IF_ERROR(graph_runner.Run(&graph, pflr->GetFLR("/device:CPU:0"), {},
                                      {fetch_node}, &outputs));
  data::DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], &dataset));

  data::DatasetBase* finalized_dataset;
  std::unique_ptr<thread::ThreadPool> pool(
      NewThreadPoolFromSessionOptions(params.session_options));
  std::function<void(std::function<void()>)> runner =
      [&pool](std::function<void()> c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_5(mht_5_v, 297, "", "./tensorflow/core/data/standalone.cc", "lambda");
 pool->Schedule(std::move(c)); };
  OpKernelContext::Params op_params =
      CreateParams(pflr.get(), device_mgr.get(), &runner);
  OpKernelContext ctx(&op_params, /*num_outputs=*/0);
  TF_RETURN_IF_ERROR(data::FinalizeDataset(&ctx, dataset, &finalized_dataset));
  core::ScopedUnref unref(finalized_dataset);
  *result = WrapUnique(new Dataset(
      finalized_dataset, dataset, device_mgr.release(), pflr.release(),
      flib_def.release(), pool.release(), std::move(runner)));
  return Status::OK();
}  // static

Status Dataset::MakeIterator(
    std::vector<std::unique_ptr<SplitProvider>> split_providers,
    std::unique_ptr<Iterator>* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_6(mht_6_v, 314, "", "./tensorflow/core/data/standalone.cc", "Dataset::MakeIterator");

  // Create an `IteratorContext`, which bundles together the necessary runtime
  // support to create and get elements from an iterator.
  std::unique_ptr<IteratorContext> ctx;
  // NOTE(mrry): In the current API, an `IteratorContext` is always initially
  // created from an `OpKernelContext*`, so we need to create `OpKernelContext`
  // with a valid subset of parameters.
  OpKernelContext::Params op_params =
      CreateParams(pflr_.get(), device_mgr_.get(), &runner_);
  OpKernelContext op_ctx(&op_params, /*num_outputs=*/0);
  IteratorContext::Params params(&op_ctx);
  params.cancellation_manager = &cancellation_manager_;
  params.function_handle_cache = function_handle_cache_.get();
  params.resource_mgr = &resource_mgr_;
  std::move(split_providers.begin(), split_providers.end(),
            std::back_inserter(params.split_providers));
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  ctx = absl::make_unique<IteratorContext>(std::move(params));

  // Create the iterator from the dataset.
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(finalized_dataset_->MakeIterator(
      ctx.get(), /*parent=*/nullptr, "Iterator", &iterator));
  *result = WrapUnique(new Iterator(iterator.release(), ctx.release()));

  return Status::OK();
}

Status Dataset::MakeIterator(std::unique_ptr<Iterator>* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_7(mht_7_v, 346, "", "./tensorflow/core/data/standalone.cc", "Dataset::MakeIterator");

  return MakeIterator(/*split_providers=*/{}, result);
}

Status Dataset::MakeSplitProviders(
    std::vector<std::unique_ptr<SplitProvider>>* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_8(mht_8_v, 354, "", "./tensorflow/core/data/standalone.cc", "Dataset::MakeSplitProviders");

  return finalized_dataset_->MakeSplitProviders(result);
}

const DatasetBase* Dataset::Get() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_9(mht_9_v, 361, "", "./tensorflow/core/data/standalone.cc", "Dataset::Get");
 return finalized_dataset_; }

Dataset::Dataset(DatasetBase* finalized_dataset, DatasetBase* original_dataset,
                 DeviceMgr* device_mgr, ProcessFunctionLibraryRuntime* pflr,
                 FunctionLibraryDefinition* flib_def, thread::ThreadPool* pool,
                 std::function<void(std::function<void()>)> runner)
    : finalized_dataset_(finalized_dataset),
      original_dataset_(original_dataset),
      device_mgr_(device_mgr),
      flib_def_(flib_def),
      pflr_(pflr),
      interop_threadpool_(pool),
      runner_(std::move(runner)),
      unbounded_thread_pool_(Env::Default(), "tf_data_standalone") {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_10(mht_10_v, 377, "", "./tensorflow/core/data/standalone.cc", "Dataset::Dataset");

  finalized_dataset_->Ref();
  original_dataset_->Ref();
  function_handle_cache_ =
      absl::make_unique<FunctionHandleCache>(pflr_->GetFLR("/device:CPU:0"));
}

Dataset::~Dataset() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSstandaloneDTcc mht_11(mht_11_v, 387, "", "./tensorflow/core/data/standalone.cc", "Dataset::~Dataset");

  finalized_dataset_->Unref();
  original_dataset_->Unref();
}

}  // namespace standalone
}  // namespace data
}  // namespace tensorflow
