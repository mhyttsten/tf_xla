/*
  These intructions apply for a local macOS build.
  On master for repo https://github.com/mhyttsten/tf_xla, most of these operations have already been performed.
  So when using https://github.com/mhyttsten/tf_xla, the following steps are only required:
     - 0, 3, 4

  0a. Install TensorFlow source build requisites
  0b. $ git clone https://github.com/mhyttsten/tf_xla.git  # Based on tf 2.9 with function tracing enabled for source files
  0c. $ cd tf_xla
      $ ./configure  # To specify your Python virtualenv paths, use default options for all (other) questions

  1a. Put this file content in tensorflow/compiler/xla/XLATesting.cc

  b. Add the following to tensorflow/compiler/xla/BUILD (after filegroups)
c_binary(
    name = "XLATesting",
    srcs = ["XLATesting.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/stream_executor:stream_executor_impl",
# NOTE 0: The two below lines seems to work as well as doing NOTE 1 (but we cannot be certain when doing future stuff)
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:cpu_transfer_manager",
# NOTE 1:
# Entries below these lines added to make xla::Compiler::GetForPlatform, and xla::ClientLibrary::GetOrCreateLocalClient
# work. Otherwise their return status is NOT_FOUND for those calls.
# Observe, no code changes are required, just linking these in make these call go to OK instead of NOT_FOUND so
# probably a dependency on a global variable/constructor that registers these.
# Only //tensorflow/compiler/jit:xla_cpu_jit is specified as needed but it needs to subsequent ones to link correctly
# (i.e. no undefined symbol).
# Unfortunate side-effect is that below addition increases link time from 53s to 2572s. 
# Maybe it's possible to figure out how Compiler::GetForPlatform and ClientLibrary::GetOrCreateLocalClient depend
# on //tensorflow/compiler/jit:xla_cpu_jit to shorten this down.
#        "//tensorflow/compiler/jit:xla_cpu_jit",
#        "//tensorflow/core/common_runtime:core_cpu_impl",
#        "//tensorflow/core/common_runtime/gpu:gpu_runtime_impl",
#        "//tensorflow/cc/saved_model:bundle_v2",
#        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
#        "//tensorflow/core/profiler/internal/cpu:annotation_stack",
    ],
    visibility = ["//visibility:public"],
)

  2a. For the following targets (i.e. BUILD files under tensorflow source tree):
       //tensorflow/core/profiler/internal/cpu:traceme_recorder
       //tensorflow/core/profiler/utils:time_utils
       //tensorflow/cc/saved_model:metrics  # Ripple from "//tensorflow/compiler/jit:xla_cpu_jit"
       //tensorflow/core/profiler/internal/cpu:annotation_stack
     Remove the if_static block, and instead force the source files in the block to be included in deps section.
  b. For //tensorflow/core/profiler/internal/cpu:annotation_stack, change visibility from = ["//tensorflow/core/profiler:internal"]
     to visibility = ["//visibility:public"]

  3. Clean and Build XLA example program (assuming in directory having .git):
     $ bazel clean --expunge
     $ bazel build --copt="-fno-inline-functions" //tensorflow/compiler/xla:XLATesting
       (alt use --verbose_failures after 'build' to get verbose build output)
       -fno-inline-functions allow call tracing to work properly for inlined function definitions

  4. Run
     $ bazel-bin/tensorflow/compiler/xla/XLATesting
     If TensorFlow source tree had function tracing (MHTracer) enabled, you can enable/disable trace output by having
     'MHTRACE_ENABLE' somewhere in $PATH environment variable (i.e as an :MHTRACE_ENABLE: element)
*/

#include <iostream>
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/stream_executor/host/host_platform.h"  // HostPlatform
#include "tensorflow/compiler/xla/client/client_library.h"  // LocalClientOptions
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"

using namespace std;
using namespace tensorflow;

// TODO:
//    - This program will successfully create a CPU Compiler, a LocalClient, and initialize an XlaBuilder
//      Next steps is to use xlaBuilder to create, compiler, and execute a program.
//      For example, perform: 1+1 and print the result from a cout

int main(int argc, char* argv[]) {	

  cout << "Hello World from XLA directory, v2" << endl;

  // May need other things before, like initialization of stream_executor?

  stream_executor::host::HostPlatform platform; 
  StatusOr<xla::Compiler*> sorCompiler = xla::Compiler::GetForPlatform(&platform);   //Compiler::GetForPlatform [268 @ ./tensorflow/compiler/xla/service/compiler.cc]
  cout << "Compiler: " << sorCompiler.status().ToString() << endl;

  xla::LocalClientOptions localClientOptions(  // LocalClientOptions::LocalClientOptions [205 @ ./tensorflow/compiler/xla/client/client_library.cc]
    &platform,     // Or use default: se::Platform* platform = nullptr,
    1,             // int number_of_replicas = 1,
    -1,            // int intra_op_parallelism_threads = -1,
    absl::nullopt  //const absl::optional<std::set<int>>& allowed_devices = absl::nullopt
  );

  // nullptr for default platform according to doc
  localClientOptions = localClientOptions.set_platform(&platform);  // LocalClientOptions::set_platform [210 @ ./tensorflow/compiler/xla/client/client_library.cc]

  // Sets the thread pool size for parallel execution of an individual operator
  // Not sure what TF sets but lets use default -1
  localClientOptions = localClientOptions.set_intra_op_parallelism_threads(-1);  // LocalClientOptions::set_intra_op_parallelism_threads [240 @ ./tensorflow/compiler/xla/client/client_library.cc]

  // Set of device IDs for which the stream executor will be created, for the given platform.
  auto allowedDevices = absl::optional<std::set<int>>{};
  localClientOptions = localClientOptions.set_allowed_devices(allowedDevices);  // LocalClientOptions::set_allowed_devices [256 @ ./tensorflow/compiler/xla/client/client_library.cc]

  // Singleton constructor-or-accessor -- returns a client for the application to issue XLA commands on.
  StatusOr<xla::LocalClient*> sorLocalClient = xla::ClientLibrary::GetOrCreateLocalClient(localClientOptions);
  cout << "LocalClient: " << sorLocalClient.status().ToString() << endl;
  assert(sorLocalClient.status().ok());

  // A convenient interface for building up computations.
  xla::XlaBuilder xlaBuilder{"__inference_return1_5<LB>_XlaMustCompile=true,config_proto=3175580994766145631,executor_type=11160318154034397263<RB>"};  // XlaBuilder::XlaBuilder [computation_name: "__inference_return1_5<LB>_XlaMustCompile=true,config_proto=3175580994766145631,executor_type=11160318154034397263<RB>"] [ 505 @ ./tensorflow/compiler/xla/client/xla_builder.cc]
  Status s = xlaBuilder.GetCurrentStatus();
  cout << "XlaBuilder: " << xlaBuilder.GetCurrentStatus().ToString() << endl;
  assert(xlaBuilder.GetCurrentStatus().ok());


  // ???Are we now in a good state to build our operations and compile them???
  // ???How do we move forward from here??
  // ???If we'd like to generate a program adding 2 scalars, e.g: 1+1???

  cout << "Goodbye World from XLA directory" << endl;
  return 0;

/*
// NoOp
SetOpMetadata [1] [] [] [ 379 @ ./tensorflow/compiler/xla/client/xla_builder.h]
frontend_attributes [1] [] [] [ 444 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedFrontendAttributesAssignment [1] [] [] [ 1831 @ ./tensorflow/compiler/xla/client/xla_builder.h]
sharding [1] [] [] [ 465 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedShardingAssignment [1] [] [] [ 1792 @ ./tensorflow/compiler/xla/client/xla_builder.h]
ClearOpMetadata [1] [] [] [ 405 @ ./tensorflow/compiler/xla/client/xla_builder.h]

// Const
SetOpMetadata [1] [] [] [ 379 @ ./tensorflow/compiler/xla/client/xla_builder.h]
frontend_attributes [1] [] [] [ 444 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedFrontendAttributesAssignment [1] [] [] [ 1831 @ ./tensorflow/compiler/xla/client/xla_builder.h]
sharding [1] [] [] [ 465 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedShardingAssignment [1] [] [] [ 1792 @ ./tensorflow/compiler/xla/client/xla_builder.h]
ClearOpMetadata [1] [] [] [ 405 @ ./tensorflow/compiler/xla/client/xla_builder.h]
DfsHloVisitorBase [1] [] [] [ 235 @ ./tensorflow/compiler/xla/service/dfs_hlo_visitor.h]
DfsHloVisitorWithDefaultBase [1] [] [] [ 216 @ ./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h]
HloEvaluator::HloEvaluator [1] [] [] [ 924 @ ./tensorflow/compiler/xla/service/hlo_evaluator.cc]
ValueInference [1] [] [] [ 258 @ ./tensorflow/compiler/xla/client/value_inference.h]
...
XlaExpression::Constant [1] [] [] [ 206 @ ./tensorflow/compiler/tf2xla/xla_expression.cc]
   XlaOp [1] [] [] [ 247 @ ./tensorflow/compiler/xla/client/xla_builder.h]
...
Tensor::Tensor [1] [] [] [ 1202 @ ./tensorflow/core/framework/tensor.cc]
   Allocate [1] [] [] [ 207 @ ./tensorflow/core/framework/typed_allocator.h]
   AllocateRaw [1] [] [] [ 307 @ ./tensorflow/core/framework/allocator.h]
      AllocateRaw [1] [] [] [ 222 @ ./tensorflow/compiler/tf2xla/xla_compilation_device.cc]
        AlignedMalloc [1] [] [] [ 466 @ ./tensorflow/core/platform/default/port.cc]
           XlaOp [1] [] [] [ 247 @ ./tensorflow/compiler/xla/client/xla_builder.h]
...
ClearOpMetadata [1] [] [] [ 405 @ ./tensorflow/compiler/xla/client/xla_builder.h]

// _RetVal
SetOpMetadata [1] [] [] [ 379 @ ./tensorflow/compiler/xla/client/xla_builder.h]
frontend_attributes [1] [] [] [ 444 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedFrontendAttributesAssignment [1] [] [] [ 1831 @ ./tensorflow/compiler/xla/client/xla_builder.h]
sharding [1] [] [] [ 465 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedShardingAssignment [1] [] [] [ 1792 @ ./tensorflow/compiler/xla/client/xla_builder.h]
ClearOpMetadata [1] [] [] [ 405 @ ./tensorflow/compiler/xla/client/xla_builder.h]
...
DfsHloVisitorBase [1] [] [] [ 235 @ ./tensorflow/compiler/xla/service/dfs_hlo_visitor.h]
DfsHloVisitorWithDefaultBase [1] [] [] [ 216 @ ./tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h]
HloEvaluator::HloEvaluator [1] [] [] [ 924 @ ./tensorflow/compiler/xla/service/hlo_evaluator.cc]
...
ValueInference [1] [] [] [ 258 @ ./tensorflow/compiler/xla/client/value_inference.h]
...
XlaContext::SetRetval [1] [] [] [ 253 @ ./tensorflow/compiler/tf2xla/xla_context.cc]
   XlaOp [1] [] [] [ 247 @ ./tensorflow/compiler/xla/client/xla_builder.h]
   shape [1] [] [] [ 538 @ ./tensorflow/core/framework/tensor.h]
   TensorShapeRep::TensorShapeRep [1] [] [] [ 977 @ ./tensorflow/core/framework/tensor_shape.h]
   Tensor::Tensor [1] [] [] [ 1356 @ ./tensorflow/core/framework/tensor.h]

// NoOp
SetOpMetadata [1] [] [] [ 379 @ ./tensorflow/compiler/xla/client/xla_builder.h]
def [1] [] [] [ 335 @ ./tensorflow/core/framework/op_kernel.h]
frontend_attributes [1] [] [] [ 444 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedFrontendAttributesAssignment [1] [] [] [ 1831 @ ./tensorflow/compiler/xla/client/xla_builder.h]
sharding [1] [] [] [ 465 @ ./tensorflow/compiler/xla/client/xla_builder.h]
XlaScopedShardingAssignment [1] [] [] [ 1792 @ ./tensorflow/compiler/xla/client/xla_builder.h]
ClearOpMetadata [1] [] [] [ 405 @ ./tensorflow/compiler/xla/client/xla_builder.h]
...
XlaComputation [1] [] [] [ 200 @ ./tensorflow/compiler/xla/client/xla_computation.h]
*/

/*
NOTES

//---
Platform [abstract] (tf/stream_executor/platform.h)
   void*        Id();  // Uniquely identifies this platform
   std::string  Name();
   port::Status Initialize(const std::map<std::string, std::string>& platform_options);
   bool         Initialized() const;
   port::StatusOr<StreamExecutor*> GetExecutor(const StreamExecutorConfig& config);
   void RegisterTraceListener(std::unique_ptr<TraceListener> listener);

//---
HostPlatform : Platform (tf/stream_executor/host/host_platform.h)
   Has always Name: "Host", and a unique id

//---
Compiler [abstract] (tf/compiler/xla/service/compiler.h)
   class CompileOptions {
      se::DeviceMemoryAllocator* device_allocator;
      tf::thread::ThreadPool*    thread_pool;
   };

   Platform::Id PlatformId();

  // Runs Hlo passes to optimize the given Hlo module, returns the optimized module.
   StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module,
      se::StreamExecutor* executor,
      const CompileOptions& options);

   // Performs scheduling and buffer assignment and returns the buffer assignments.
   // This base class returns error: Unimplemented("This compiler does not support this method");
   StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(const HloModule* module);

   StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      se::StreamExecutor* executor,
      const CompileOptions& options);

   // Compiles a set of HLO modules that can run in parallel, potentially communicating data
   // between the modules, and returns a corresponding sequence of executable objects.
   StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options);

  // A CompilerFactory is a function object: std::unique_ptr<Compiler> foo()
  static std::map<se::Platform::Id, CompilerFactory> compiler_factories;
  static std::map<se::Platform::Id, std::unique_ptr<Compiler>> platform_compilers;
  static ... RegisterCompilerFactory(...);    // Map platformId -> CompilerFactory
  static ... GetPlatformCompilerFactories();  // Map platformId -> CompilerFactory
  static ... GetPlatformCompilers();          // Map platformId -> Compiler*

  // Find a compiler either caches in PlatformCompilers, or use factory to create one and cache it
  // If none is cached, and there is no factory for it, "try adding tensorflow/compiler/jit:xla_cpu_jit as deps"
  static StatusOr<Compiler*> Compiler::GetForPlatform(se:Platform*)

//---
LLVMCompiler : Compiler (tf/compiler/xla/service/llvm_compiler.cc)

  // A callback of this type can be run before and/or after IR-level optimization.
  // E.g. to dump out the generated IR to disk or gather some statistics.
  using ModuleHook = std::function<void(const llvm::Module&)>;  // "llvm/IR/Module.h"
  void SetPreOptimizationHook(ModuleHook hook);
  void SetPostOptimizationHook(ModuleHook hook);

  // I can't see much LLVM specfic in this implementation
  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      const CompileOptions& options);

//---
CpuCompiler
   Lots of interesting stuff here involving LLVM

TODO:

//---
TODO:

#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"



//---
InitModule [1] [] [] [ 1822 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
   Compiler::RegisterCompilerFactory [1] [] [] [ 256 @ ./tensorflow/compiler/xla/service/compiler.cc]
      Compiler::GetPlatformCompilerFactories [1] [] [] [ 236 @ ./tensorflow/compiler/xla/service/compiler.cc]


*/


}
