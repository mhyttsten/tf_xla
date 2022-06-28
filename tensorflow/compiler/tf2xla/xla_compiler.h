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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilerDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilerDTh() {
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


#include <stack>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class XlaContext;

// The XlaCompiler class is responsible for compilation of a self-contained
// subgraph of a TensorFlow computation using the XLA linear algebra runtime.
// It does a symbolic execution of the graph starting from specific input
// shapes, using a JIT device to convert operators into XLA computations.
//
// XlaCompiler is typically invoked from an `XlaLaunch` operator once the
// shapes of all input parameters to the computation are known. This is
// because the symbolic execution requires known shapes for all operations.
//
// XlaCompiler compiles Tensorflow graphs that received inputs via _Arg nodes,
// and return outputs via _Retval nodes.
//
// The XlaCompiler requires one Argument struct for each _Arg index, that
// describes each argument. Arguments can be compile-time constants
// (kind kConstant), run-time parameters (kind kParameter), or resources
// (kind kResource).
//
// Only kParameter and initialized kResource arguments become runtime parameters
// to the generated XLA computation.
//
// The run-time outputs of the XLA computation are arranged in the following
// order:
//   +------------------+-----------------------------------------+
//   |  _Retval values  |  Updated values of kResource arguments  |
//   +------------------+-----------------------------------------+
// _Retval values are ordered by _Retval index, whereas kResource values are
// ordered by the original _Arg position of the variable.
//
// If a shape representation function is provided as part of
// XlaCompiler::CompileOptions, kParameter arguments and return values to an
// entry computation will be reshaped in accordance to the shape function.
// Arguments and return values to a non-entry computation are not reshaped.
// Variable resource arguments are passed and returned in reshaped form, even
// for non-entry computations. This feature allows TensorFlow to keep on-device
// tensors with a different shape to their representation inside the XLA
// computation.
//
// In computation outputs, updated kResource values are placed the end. When
// emitting While loop bodies, we must ensure that the loop body has
// identical input and output signatures. By passing variable values
// at the end of the argument list and using the
// `return_updated_values_for_all_variables` option, we can ensure that the
// input and output values of resources appear at the same positions.
//
// Resources are passed as parameters or returned as resource updates in
// "packed" form.
// kStack resources are packed as (array, size of stack) XLA tuples.
// kTensorArray resources without gradients are packed as the array that
// backs the TensorArray. If gradients are present (`tensor_array_gradients`),
// the packed representation is a (array, gradient0, gradient1, ...) tuple,
// where gradient_k is the value of the k-th gradient in the
// `tensor_array_gradients` ordered set.
class XlaCompiler {
 public:
  using Argument = ::tensorflow::XlaArgument;

  // Options pertaining to an individual call to CompileGraph() or
  // CompileFunction().
  struct CompileOptions {
    // If `use_tuple_arg` is true, a single tuple parameter will be used for all
    // arguments; if false, each argument gets its own parameter.
    bool use_tuple_arg = false;

    // If 'return_updated_values_for_all_resources' is true, then updated
    // values of all resource arguments will be included in the
    // 'resource_updates' of the computation, even if the resource was not
    // modified by the computation. Used when compiling loop bodies to ensure
    // the input and output signatures match.
    bool return_updated_values_for_all_resources = false;

    // If 'always_return_tuple' is true, then the output of a computation will
    // always be a tuple. Otherwise, a single-element output will not be wrapped
    // in a tuple.
    bool always_return_tuple = true;

    // True when compiling the entry computation, false for subcomputations
    // (while, call, etc.)
    bool is_entry_computation = true;

    // True when we should add XLA input & output to the graph/function.
    bool add_token_input_output = false;

    // Resource updates are converted into input / output of xla. The two
    // buffers are aliased with other if this option is true.
    bool alias_resource_update = false;
  };

  using OutputDescription = ::tensorflow::XlaOutputDescription;

  using ResourceUpdate = ::tensorflow::XlaResourceUpdate;

  using CompilationResult = ::tensorflow::XlaCompilationResult;

  struct Options {
    // Name of the compilation device to use. It must be set by the caller.
    // The default empty value is invalid.
    DeviceType device_type = DeviceType("");

    // The device to use during compilation to execute instructions on, for
    // example for auto-tuning.
    // Valid values are defined by `xla::Backend::devices_ordinal_supported()`.
    // -1 indicates the default device should be used.
    int device_ordinal = -1;

    xla::Client* client = nullptr;

    // Function library in which to find function definitions. Must be non-null.
    const FunctionLibraryDefinition* flib_def = nullptr;

    // The graph def version to be compiled.
    int graph_def_version = TF_GRAPH_DEF_VERSION;

    // If 'allow_cpu_custom_calls' is true, kernels may make use of CustomCall()
    // for CPU.
    bool allow_cpu_custom_calls = false;

    // A ShapeDeterminationFns (i.e., a bundle of LayoutSelectionFn and
    // ShapeRepresentationFn). Each bundle describes the XLA representation of
    // arguments represented to XLA as the shape given by this shape function.
    // Arguments are input activations or weights to an XLA entry computation.
    // Variables are reshaped to this shape on write, and reshaped to their
    // original shape on read.
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;

    // If not nullptr, populate_resource_manager is called with the
    // compilation device's resource manager when the compilation
    // device is created, and can be used to create metadata objects
    // that can be accessed by XLA op kernels.
    std::function<Status(ResourceMgr*)>* populate_resource_manager = nullptr;

    // If not nullptr, this memory allocator can be used by the compiler for
    // temporary allocations it might want to make during compilation.
    //
    // For example, the compiler may want to try out different algorithms and
    // choose the fastest one, and it might run those algorithms over buffers
    // created using this allocator.
    //
    // The compiler can function correctly without an explicit allocator given
    // here, but on some devices (notably, GPUs), TensorFlow tends to eagerly
    // allocate most or all available memory on the device, leaving none for the
    // compiler to access, unless it can use TensorFlow's allocator.
    // This must be a shared_ptr, as this is passed all the way down to the
    // cluster compilation. This allows asynchronous compilation to hold a
    // reference until the compilation is finished.
    std::shared_ptr<se::DeviceMemoryAllocator> device_allocator;

    // Alias input and output buffers for parameters that are passed-through XLA
    // modules without being changed.
    bool alias_passthrough_params = false;

    // Enable detailed logging of compilation metadata.
    bool detailed_logging = true;
  };

  explicit XlaCompiler(Options options);

  ~XlaCompiler();

  // Helper function to populate an XlaCompiler::Argument from XlaResource.
  static void PopulateArgumentFromResource(const XlaResource& resource,
                                           Argument* arg);

  Status CompileFunction(const CompileOptions& options,
                         const NameAttrList& fn_name_attrs,
                         absl::Span<const Argument> args,
                         CompilationResult* result);

  // Compiles a tensorflow::Graph into an xla::XlaComputation.
  // Similar to CompileFunction, but takes a Graph as input rather than a
  // function.
  Status CompileGraph(
      const CompileOptions& options, string const& name,
      std::unique_ptr<Graph> graph, absl::Span<const Argument> args,
      CompilationResult* result);

  // Returns the shape of the XLA parameter for an argument 'arg'.
  // See the class comment for more details about the argument passing
  // convention.
  Status XLAShapeForArgument(
      const Argument& arg, bool is_entry_computation,
      const absl::optional<xla::HloSharding>& arg_sharding,
      xla::Shape* xla_shape) const;

  // Retrieves the channel handle associated with `key`. Allocates
  // a new channel handle if none exists.
  // Channel handles can be used to communicate between different
  // computations. Computations that communicate should be compiled with the
  // same XlaCompiler.
  Status GetChannelHandle(const string& key, xla::ChannelHandle* channel);

  // Retrieves the host-to-device channel handle associated with `key`.
  // Allocates a new channel handle if none exists.
  Status GetHostToDeviceChannelHandle(const string& key,
                                      xla::ChannelHandle* channel);

  // Retrieves the device-to-host channel handle associated with `key`.
  // Allocates a new channel handle if none exists.
  Status GetDeviceToHostChannelHandle(const string& key,
                                      xla::ChannelHandle* channel);

  // Sets the shapes and types for the device to host transfer associated with
  // 'key'.
  Status SetDeviceToHostMetadata(const string& key,
                                 absl::Span<const DataType> types,
                                 absl::Span<const TensorShape> shapes);

  // Gets the shapes the device to host transfer associated with 'key'.
  Status GetDeviceToHostShapes(const string& key,
                               std::vector<TensorShape>* shapes) const;

  // Sets the shapes and types for the host to device transfer associated with
  // 'key'.
  Status SetHostToDeviceMetadata(const string& key,
                                 absl::Span<const DataType> types,
                                 absl::Span<const TensorShape> shapes);

  // In order to avoid deadlocks from dependencies in host computations, it can
  // be necessary to enforce a partial order on the execution of HostCompute
  // Ops. In particular it may be necessary to constrain the SendToHost for one
  // HostCompute to run before blocking on the RecvAtHost for another
  // HostCompute. The compiler maintains a mapping from 'host_compute_name' to
  // handle, where the handle is an 'output' of the HostCompute Op corresponding
  // to 'host_compute_name'. Another HostCompute Op that needs to be sequenced
  // later can add the handle as an 'input' to enforce the constraints.
  // 'host_compute_name' can be any string the client wishes to use to identify
  // a given HostCompute Op as long as the names are unique within the
  // compilation.
  Status GetHostComputeControlDependency(const string& host_compute_name,
                                         xla::XlaOp* handle);
  Status SetHostComputeControlDependency(const string& host_compute_name,
                                         const xla::XlaOp& handle);

  const Options& options() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilerDTh mht_0(mht_0_v, 450, "", "./tensorflow/compiler/tf2xla/xla_compiler.h", "options");
 return options_; }
  xla::Client* client() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilerDTh mht_1(mht_1_v, 454, "", "./tensorflow/compiler/tf2xla/xla_compiler.h", "client");
 return options_.client; }
  FunctionLibraryRuntime* flib_runtime() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compilerDTh mht_2(mht_2_v, 458, "", "./tensorflow/compiler/tf2xla/xla_compiler.h", "flib_runtime");
 return flib_runtime_; }

  void PushNodeTokenMapping();
  Status PopNodeTokenMapping();
  Status SetNodeToken(const string& node_name, const xla::XlaOp& op);
  StatusOr<xla::XlaOp> GetNodeToken(const string& node_name);

  // Sets the function body `fbody` to the one registered as `function`.
  Status FindFunctionBody(const NameAttrList& function,
                          const FunctionBody** fbody,
                          const ConfigProto** config_proto = nullptr);

 private:
  // Returns the optimized graph object in this function body.
  std::unique_ptr<Graph> GetGraph(const FunctionBody* fbody);

  // Builds XLA computations for each of the arguments to the computation.
  // `args` are the arguments to the computation.
  Status BuildArguments(const Graph& graph,
                        const std::vector<XlaCompiler::Argument>& args,
                        bool use_tuple_arg, xla::XlaBuilder* builder,
                        XlaContext* context,
                        const std::map<int, xla::OpSharding>& arg_shardings,
                        std::vector<XlaExpression>* arg_expressions,
                        std::vector<int>* input_to_args,
                        std::vector<xla::Shape>* input_shapes,
                        bool is_entry_computation);

  // Graph compiler needs to know how to get an optimized graph from a function
  // body.
  friend class GraphCompiler;
  friend class XlaCompilerTest;

  Options options_;

  // Status set to non-OK in the constructor if initialization fails.
  Status initialization_status_;

  // Returns the next step sequence number.
  int64_t NextStepId();

  // Internal sequence number for steps executed on the compilation device.
  int64_t next_step_id_;

  XlaCompilationDevice* device_;  // Owned by device_mgr_
  StaticDeviceMgr device_mgr_;

  // To avoid copying the client's function library, use a local function
  // library and runtime for functions created as part of the functionalize
  // control flow transformation.
  std::unique_ptr<FunctionLibraryDefinition> local_flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> local_pflr_;

  FunctionLibraryRuntime* local_flib_runtime_;  // owned by local_pflr_.
  FunctionLibraryRuntime* flib_runtime_;        // owned by pflr_.

  struct SignatureHash {
    uint64 operator()(
        const std::pair<string, std::vector<Argument>>& signature) const;
  };

  std::unordered_map<std::pair<string, std::vector<Argument>>,
                     CompilationResult, SignatureHash>
      cache_;

  std::unordered_map<string, xla::ChannelHandle> channels_;

  std::unordered_map<string, tf2xla::HostTransferMetadata> host_compute_sends_;
  std::unordered_map<string, tf2xla::HostTransferMetadata> host_compute_recvs_;

  std::unordered_map<string, xla::XlaOp> host_compute_control_output_;

  // This is used to store <node name, token output> mapping. Side-effecting
  // ops call SetNodeToken() to record its token output, so later side-effecting
  // ops can use GetNodeToken() to get it and use it as token input.
  //
  // It's a stack because we need a mapping like this for each level of nested
  // CompileGraph() call. In CompileGraph(), we will push a new mapping to the
  // stack, and pop the mapping before returning.
  std::stack<std::map<string, xla::XlaOp>> node_token_mapping_stack_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompiler);
};


}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
