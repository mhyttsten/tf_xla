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
#ifndef TENSORFLOW_CORE_DATA_CAPTURED_FUNCTION_H_
#define TENSORFLOW_CORE_DATA_CAPTURED_FUNCTION_H_
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
class MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh() {
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


#include <memory>
#include <vector>

#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class Device;
class OpKernelContext;
class ResourceMgr;

namespace data {

class CapturedFunction;
class InstantiatedCapturedFunction;

// Creates an iterator for a dataset which is created by applying the given
// function to the given input element.
Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const IteratorBase* parent,
    const std::vector<Tensor>& input_element, int64_t thread_index,
    const InstantiatedCapturedFunction& inst_captured_func, StringPiece prefix,
    std::unique_ptr<IteratorBase>* out_iterator);

// Creates an iterator for a dataset which is created by applying the given
// function to the given input element. Pass non-null `node` to record
// processing time for modeling Iterator's GetNext() resource usage.
Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const IteratorBase* parent,
    const std::vector<Tensor>& input_element, int64_t thread_index,
    const InstantiatedCapturedFunction& inst_captured_func, StringPiece prefix,
    std::unique_ptr<IteratorBase>* out_iterator,
    const std::shared_ptr<model::Node>& node);

// Creates an iterator context appropriate for a nested dataset's iterator. A
// nested dataset is a dataset created within another dataset, e.g. by the
// function passed to `interleave` or `flat_map`.
IteratorContext MakeNestedIteratorContext(IteratorContext* ctx);

struct ShortCircuitInfo {
  std::vector<int> indices;
  std::vector<bool> can_move;
};

// Metadata shared across all captures of the same function.
class FunctionMetadata {
 public:
  struct Params {
    bool use_inter_op_parallelism = true;
    bool use_default_device = true;
  };

  // Creates a new instance of the `FunctionMetadata` class, fetching function
  // from a context argument.
  static Status Create(tensorflow::OpKernelConstruction* ctx,
                       const string& func_name, Params params,
                       std::shared_ptr<FunctionMetadata>* out_metadata);

  // Creates a new instance of the `FunctionMetadata` class, using the provided
  // function.
  static Status Create(tensorflow::OpKernelConstruction* ctx,
                       NameAttrList&& func, Params params,
                       std::shared_ptr<FunctionMetadata>* out_metadata);

  // Returns the named list of function arguments.
  const NameAttrList& func() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_0(mht_0_v, 261, "", "./tensorflow/core/data/captured_function.h", "func");
 return func_; }

  // Returns a borrowed pointer to the function library that contains the
  // transitive closure of definitions used by the function.
  const FunctionLibraryDefinition* lib_def() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_1(mht_1_v, 268, "", "./tensorflow/core/data/captured_function.h", "lib_def");
 return lib_def_.get(); }

  // Returns short-circuit information.
  const ShortCircuitInfo& short_circuit_info() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_2(mht_2_v, 274, "", "./tensorflow/core/data/captured_function.h", "short_circuit_info");

    return short_circuit_info_;
  }

  // Indicates whether a default device should be used for executing function
  // ops.
  bool use_default_device() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_3(mht_3_v, 283, "", "./tensorflow/core/data/captured_function.h", "use_default_device");
 return use_default_device_; }

  // Indicates whether to use inter-op parallelism for execution of the
  // function.
  bool use_inter_op_parallelism() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_4(mht_4_v, 290, "", "./tensorflow/core/data/captured_function.h", "use_inter_op_parallelism");
 return use_inter_op_parallelism_; }

  // Indicates whether the function should a multi-device function backend.
  bool use_multi_device_function() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_5(mht_5_v, 296, "", "./tensorflow/core/data/captured_function.h", "use_multi_device_function");
 return use_multi_device_function_; }

 private:
  FunctionMetadata(NameAttrList&& func, Params params)
      : func_(std::move(func)),
        use_default_device_(params.use_default_device),
        use_inter_op_parallelism_(params.use_inter_op_parallelism) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_6(mht_6_v, 305, "", "./tensorflow/core/data/captured_function.h", "FunctionMetadata");
}

  NameAttrList func_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_ = nullptr;
  ShortCircuitInfo short_circuit_info_;
  bool use_default_device_ = true;
  bool use_inter_op_parallelism_ = true;
  bool use_multi_device_function_ = true;
};

// Constructs and stores the parameters for the CapturedFunction Instantiate
// function.
struct InstantiateCapturedFunctionParams {
  explicit InstantiateCapturedFunctionParams(IteratorContext* ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_7(mht_7_v, 321, "", "./tensorflow/core/data/captured_function.h", "InstantiateCapturedFunctionParams");

    flr = ctx->flr();
    function_handle_cache = ctx->function_handle_cache();
    runner = ctx->runner();
  }

  explicit InstantiateCapturedFunctionParams(OpKernelContext* ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_8(mht_8_v, 330, "", "./tensorflow/core/data/captured_function.h", "InstantiateCapturedFunctionParams");

    flr = ctx->function_library();
    function_handle_cache = nullptr;
    runner = ctx->runner();
  }

  FunctionLibraryRuntime* flr;
  FunctionHandleCache* function_handle_cache;
  std::function<void(std::function<void()>)>* runner;
};

// A `CapturedFunction` encapsulates a TensorFlow function, plus any "captured"
// arguments that it closed over in the user program.
class CapturedFunction {
 public:
  // Creates a new instance using a list of named attributes, fetching captured
  // inputs from a context argument.
  static Status Create(OpKernelContext* ctx,
                       std::shared_ptr<const FunctionMetadata> metadata,
                       const string& argument_name,
                       std::unique_ptr<CapturedFunction>* out_function);

  // Creates a new instance using a list of named attributes, using provided
  // captured inputs.
  static Status Create(OpKernelContext* ctx,
                       std::shared_ptr<const FunctionMetadata> metadata,
                       std::vector<Tensor>&& captured_inputs,
                       std::unique_ptr<CapturedFunction>* out_function);

  // Adds the definition of this captured function into the given graph,
  // returning its captured inputs and types through the respective output
  // arguments.
  Status AddToGraph(SerializationContext* ctx,
                    DatasetBase::DatasetGraphDefBuilder* b,
                    std::vector<Node*>* other_arguments,
                    DataTypeVector* other_arguments_types) const;

  // Instantiates this function for use in the given context, providing an
  // InstantiatedCapturedFunction that can be used to execute functions.
  Status Instantiate(IteratorContext* ctx,
                     std::unique_ptr<InstantiatedCapturedFunction>*
                         instantiated_captured_function);

  Status Instantiate(InstantiateCapturedFunctionParams params,
                     std::unique_ptr<InstantiatedCapturedFunction>*
                         instantiated_captured_function);

  // Determines whether the captured function is stateful.
  Status CheckExternalState() const;

  // Returns the additional captured inputs that will be passed to the function.
  const std::vector<Tensor>& captured_inputs() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_9(mht_9_v, 384, "", "./tensorflow/core/data/captured_function.h", "captured_inputs");

    return captured_inputs_;
  }

  // Returns the named list of function arguments.
  const NameAttrList& func() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_10(mht_10_v, 392, "", "./tensorflow/core/data/captured_function.h", "func");
 return metadata_->func(); }

  // Returns the transitive set of function definition required to instantiate
  // this function.
  const FunctionLibraryDefinition* lib_def() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_11(mht_11_v, 399, "", "./tensorflow/core/data/captured_function.h", "lib_def");

    return metadata_->lib_def();
  }

  // If every function output corresponds to one of its inputs, the method
  // returns the mapping from output indices to input indices. Otherwise, it
  // returns an empty list.
  const ShortCircuitInfo& short_circuit_info() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_12(mht_12_v, 409, "", "./tensorflow/core/data/captured_function.h", "short_circuit_info");

    return metadata_->short_circuit_info();
  }

  // Indicates whether the function should use inter op parallelism.
  bool use_inter_op_parallelism() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPScaptured_functionDTh mht_13(mht_13_v, 417, "", "./tensorflow/core/data/captured_function.h", "use_inter_op_parallelism");

    return metadata_->use_inter_op_parallelism();
  }

 private:
  CapturedFunction(std::shared_ptr<const FunctionMetadata> metadata,
                   std::vector<Tensor> captured_inputs);

  Status IsMultiDevice(FunctionLibraryRuntime* flr,
                       bool* is_multi_device) const;

  const std::shared_ptr<const FunctionMetadata> metadata_;
  const std::vector<Tensor> captured_inputs_;

  TF_DISALLOW_COPY_AND_ASSIGN(CapturedFunction);
};

// `InstantiatedCapturedFunction` encapsulates all the runtime support needed
// to execute a tensorflow function.
//
// While `CapturedFunction` encapsulates constant attributes of the function,
// such as its name and captured arguments, `InstantiatedCapturedFunction`
// encapsulates runtime aspects, such as `FunctionLibraryRuntime` and function
// handle.
//
// The `Iterator` related classes use `InstantiatedCapturedFunction` to execute
// functions outside of the normal `OpKernel::Compute()` context.
class InstantiatedCapturedFunction {
 public:
  // Runs the instantiated captured function. This method takes ownership of
  // the tensors in `args`, in order to be able to deallocate them as early as
  // possible. Use `RunWithBorrowedArgs()` if the caller needs to retain
  // ownership of the `args`.
  Status Run(IteratorContext* ctx, std::vector<Tensor>&& args,
             std::vector<Tensor>* rets) const;

  // Runs the instantiated captured function. This method takes ownership of
  // the tensors in `args`, in order to be able to deallocate them as early as
  // possible. Use `RunWithBorrowedArgs()` if the caller needs to retain
  // ownership of the `args`. Pass non-null `node` to record processing time
  // for modeling Iterator's GetNext() resource usage. When non-null node is
  // provided, the pre-requisite is that the calling thread has previously
  // called `DatasetBaseIterator::RecordStart().
  Status Run(IteratorContext* ctx, std::vector<Tensor>&& args,
             std::vector<Tensor>* rets,
             const std::shared_ptr<model::Node>& node) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible.
  Status RunWithBorrowedArgs(IteratorContext* ctx,
                             const std::vector<Tensor>& args,
                             std::vector<Tensor>* rets) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible. Pass non-null `node` to record processing time for modeling
  // Iterator's GetNext() resource usage. When non-null node is provided, the
  // pre-requisite is that the calling thread has previously called
  // `DatasetBaseIterator::RecordStart().
  Status RunWithBorrowedArgs(IteratorContext* ctx,
                             const std::vector<Tensor>& args,
                             std::vector<Tensor>* rets,
                             const std::shared_ptr<model::Node>& node) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible. This can be useful for calling a captured function in cases where
  // an `IteratorContext*` is not available (such as a destructor).
  //
  // TODO(b/144278100): Avoid running functions without IteratorContext.
  Status RunInstantiated(const std::vector<Tensor>& args,
                         std::vector<Tensor>* rets);

  // Asynchronously runs the captured function on the given `args`, stores the
  // results in `*rets`, and calls the given `done` callback when the function
  // returns. This method takes ownership of the tensors in `args`, in order to
  // be able to deallocate them as early as possible. Pass non-null `node` to
  // record processing time for modeling Iterator's GetNext() resource usage.
  // When non-null node is provided, the pre-requisite is that the calling
  // thread has previously called `DatasetBaseIterator::RecordStart().
  void RunAsync(IteratorContext* ctx, std::vector<Tensor>&& args,
                std::vector<Tensor>* rets,
                FunctionLibraryRuntime::DoneCallback done,
                const std::shared_ptr<model::Node>& node) const;

 private:
  friend class CapturedFunction;

  InstantiatedCapturedFunction(
      FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
      DataTypeVector ret_types,
      std::function<void(std::function<void()>)> runner,
      CapturedFunction* captured_func, bool is_multi_device);

  // Determines whether a rendezvous object should be created when running the
  // instantiated function.
  bool ShouldCreateRendezvous() const;

  FunctionLibraryRuntime* const lib_;  // Not owned.
  const FunctionLibraryRuntime::Handle f_handle_;
  const DataTypeVector ret_types_;
  // Note: We capture the runner at function instantiation time to be able to
  // run the function without `IteratorContext` via `RunInstantiated`.
  std::function<void(std::function<void()>)> captured_runner_;
  CapturedFunction* const captured_func_;  // Not owned.
  const bool is_multi_device_;

  TF_DISALLOW_COPY_AND_ASSIGN(InstantiatedCapturedFunction);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_CAPTURED_FUNCTION_H_
