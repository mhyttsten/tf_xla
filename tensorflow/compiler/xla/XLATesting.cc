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

Traditional TF build
   bazel build --verbose_failures --copt="-fno-inline-functions" //tensorflow/tools/pip_package:build_pip_package
   rm -rf /tmp/tensorflow_pkg
   ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
   python3.9 -m pip install --force-reinstall /tmp/tensorflow_pkg/*.whl   
*/

#include <iostream>
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"  // StreamExecutorMemoryAllocator
#include "tensorflow/stream_executor/host/host_platform.h"  // HostPlatform
#include "tensorflow/stream_executor/host/host_gpu_executor.h"  // HostExecutor
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/xla/client/client_library.h"  // LocalClientOptions
#include "tensorflow/compiler/xla/client/local_client.h"  
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/compiler/xla/array2d.h"

void printShape(xla::Shape myshape);

using namespace std;
using namespace tensorflow;

int main(int argc, char* argv[]) {	
  cout << "Hello World from XLA directory" << endl;

  se::host::HostPlatform platform; 
  StatusOr<xla::Compiler*> sorCompiler = xla::Compiler::GetForPlatform(&platform);   
  if (!sorCompiler.ok()) {
    cout << "Error, could not get compiler: " << sorCompiler.status() << endl;
    return -1;
  }
  cout << "Compiler: " << sorCompiler.status() << endl;
  xla::Compiler* compiler = sorCompiler.ValueOrDie();

  xla::LocalClientOptions localClientOptions(
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
  if (!sorLocalClient.ok()) {
    cout << "Could not create LocalClient: " <<  sorLocalClient.status() << endl;
    return -1;
  }
  cout << "LocalClient: " << sorLocalClient.status() << endl;
  xla::LocalClient* localClient = sorLocalClient.ValueOrDie();

  // A convenient interface for building up computations.
  xla::XlaBuilder xlaBuilder{"__inference_return1_5<LB>_XlaMustCompile=true,config_proto=3175580994766145631,executor_type=11160318154034397263<RB>"};
  Status sXLABuilder = xlaBuilder.GetCurrentStatus();
  if (!sXLABuilder.ok()) {
    cout << "Could not create XlaBuilder: " << sXLABuilder << endl;
    return -1;
  }
  cout << "XlaBuilder: " << sXLABuilder << endl;

  // Based on CreateConstantFromScalar (/tf/compiler/xla/tests/client_library_test_base.h)
  xla::Literal r1Literal = xla::LiteralUtil::CreateR0<int>(1);
  xla::LiteralSlice r1LiteralSlice = xla::LiteralSlice(r1Literal);
  xla::XlaOp r1 = ConstantLiteral(&xlaBuilder, r1LiteralSlice);
  
  xla::Literal r2Literal = xla::LiteralUtil::CreateR0<int>(2);
  xla::LiteralSlice r2LiteralSlice = xla::LiteralSlice(r2Literal);
  xla::XlaOp r2 = ConstantLiteral(&xlaBuilder, r2LiteralSlice);

  xla::Literal r3Literal = xla::LiteralUtil::CreateR2FromArray2D((const xla::Array2D<int>){{1,2,3},{4,5,6}});
  xla::LiteralSlice r3LiteralSlice = xla::LiteralSlice(r3Literal);
  xla::XlaOp r3 = ConstantLiteral(&xlaBuilder, r3LiteralSlice);

  xla::XlaComputation addComputation = xla::CreateScalarAddComputation(
    xla::S32,  // Primitive type,
    &xlaBuilder);

  // xla::XlaOp addOp = xla::Reduce(
  //   &xlaBuilder,
  //   (absl::Span<const xla::XlaOp>){r1, r2},  // operands
  //   (absl::Span<const xla::XlaOp>){r1, r2},  // init_values
  //   addComputation,                          // computation
  //   (absl::Span<const int64_t>){}           // dimensions_to_reduce
  // );

  xla::XlaOp addOp = xla::Reduce(
    &xlaBuilder,
    (absl::Span<const xla::XlaOp>){r3},  // operands
    (absl::Span<const xla::XlaOp>){r2},  // init_values
    addComputation,                          // computation
    (absl::Span<const int64_t>){1}           // dimensions_to_reduce
  );

  {
    cout << endl << "*******" << endl << "ProgramShape" << endl;
  StatusOr<xla::ProgramShape> sorProgramShape = xlaBuilder.GetProgramShape();
  xla::ProgramShape& programShape = sorProgramShape.ValueOrDie();
  cout << "ProgramShape: " << programShape.ToString() << endl;
  xla::ProgramShapeProto programShapeProto = programShape.ToProto();
  string programShapeProtoStr;
  // https://pages.cs.wisc.edu/~starr/bots/Undermind-src/html/classgoogle_1_1protobuf_1_1io_1_1ZeroCopyOutputStream.html
  google::protobuf::io::StringOutputStream programShapeProtoOS(&programShapeProtoStr);
  bool success = google::protobuf::TextFormat::Print(programShapeProto, &programShapeProtoOS);
  cout << "ProgramShapeProto3: " << success << ", str: " << programShapeProtoStr << endl;
  cout << endl;
}
  
  // Build the computation
  StatusOr<xla::XlaComputation> sorBuild = xlaBuilder.Build(
    addOp,   // XlaOp root
    false);  // bool remove_dynamic_dimensions = false);
  if (!sorBuild.ok()) {
    cout << "XlaBuilder.Build failed: " << sorBuild.status() << endl;\   
    return -1;
  }
  cout << "XlaBuilder::Build: " << sorBuild.status() << endl;
  xla::XlaComputation& xlaComputation = sorBuild.ValueOrDie();


  // After XlaBuilder::Build it is illegal to get program shape

  // Set the executable build options
  xla::ExecutableBuildOptions executableBuildOptions;
  int deviceOrdinal = localClient->default_device_ordinal(); 
  executableBuildOptions.set_device_ordinal(deviceOrdinal);
  xla::Shape resultShape = xla::ShapeUtil::MakeShape(xla::S32, {2});
  // cout << "Result shape: " << resultShape.ToString() << endl;
  // printShape(resultShape);
  executableBuildOptions.set_result_layout(resultShape);
  // exeutableBuildOptions.set_device_allocator();
  // executableBuildOptions.mutable_debug_options()->set_xla_detailed_logging_and_dumping(options.detailed_logging);

  // Compile local executables
  StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>> sorLocalExecutables =
    localClient->Compile(
      xlaComputation,
      {},
      executableBuildOptions);
  if (!sorLocalExecutables.ok()) {
    cout << "Compiling local executables failed: " << sorLocalExecutables.status() << endl;
    return -1;

  }
  cout << "Local executables: " << sorLocalExecutables.status() << endl;
 
  std::vector<std::unique_ptr<xla::LocalExecutable>>& localExecutables = sorLocalExecutables.ValueOrDie();
  cout << "Number of executables: " << localExecutables.size() << endl;
  std::unique_ptr<xla::LocalExecutable>& localExecutable = localExecutables[0];

  // StreamExecutor manages a single device, in terms of executing work 
  //   class StreamExecutor (tf/stream_executor/stream_executor_pimpl.h)
  //     port::Status Init(DeviceOptions device_options);

  // Base class
  //     DeviceMemoryAllocator [Abstract] (tf/se/device_memory_allocator.h)
  //
  // Implementing classes
  //
  //   1a. StreamExecutorMemoryAllocator : DeviceMemoryAllocator (tf/se/device_memory_allocator.h)
  //      Default memory allocator for a platform which use StreamExecutor::Allocate/Deallocate
  //      StreamExecutorMemoryAllocator(StreamExecutor*)
  //
  //    b. StreamExecutor (tf/se/stream_executor_pimpl.h)
  //      Manages execution of work on a single device, in terms of executing work (kernel launches and memory management 
  //      Takes a StreamExecutorInterface at construction as implementation (e.g. if it's a CUDA or OpenCL executor)
  //      - StreamExecutor(const Platform*, std::unique_ptr<internal::StreamExecutorInterface>, int device_ordinal);
  //      OBS these methods will proxy to implementation class
  //      - port::Status Init()  // Uses DeviceOptions::Default()
  //      - port::Status Init(DeviceOptions device_options)
  //
  //    c. StreamExecutorInterface [abstract] (tf/se/stream_executor_internal.h)
  //      Interface for the different StreamExecutor platforms (i.e. CUDA, OpenCL).
  //
  //      1. XlaInterpreterExecutor (tf/compiler/xla/service/interpreter/executor.h)
  //         A CPU-only implementation of the StreamExecutor interface.
  //         Used for testing and to examine the performance of host-based StreamExecutor code.
  //         - XlaInterpreterExecutor(const PluginConfig &);
  //         - port::Status Init(int device_ordinal, DeviceOptions);
  // 
  //      2. GpuExecutor(const PluginConfig&) (tf/se/gpu/gpu_executor.h)
  //         The CUDA implementation of the StreamExecutorInterface functionality.
  //         StreamExecutor basically correspond to the CUDA streams programming model provided by the
  //         libcuda.so driver APIs, so we don't have to do much more than wrap the calls to the libraries appropriately.
  //         - GpuExecutor(const PluginConfig&)
  //         - port::Status Init(int device_ordinal, DeviceOptions);
  //
  //      3. HostExecutor (tf/se/host/host_gpu_executor.h)
  //         A CPU-only implementation of StreamExecutor, that does no communication or interaction with a device.  
  //         Used for testing and to examine the performance of host-based StreamExecutor code.
  //         - HostExecutor (const PluginConfig&);
  //         - port::Status Init(int device_ordinal, DeviceOptions device_options) override;
  //
  //   2. Adapter class that wraps a Tensorflow allocator. 
  //     TfAllocatorAdapter(tf::Allocator *wrapped, Stream *stream) (tf/se/tf_allocator_adapter.h)

  se::PluginConfig pluginConfig;  // Use the defaults
  std::unique_ptr<se::internal::StreamExecutorInterface> streamExecutorImpl(new se::host::HostExecutor(pluginConfig));
  se::StreamExecutor streamExecutor(&platform, std::move(streamExecutorImpl), deviceOrdinal);
  se::StreamExecutorMemoryAllocator memoryAllocator(&streamExecutor);

  xla::ExecutableRunOptions executable_run_options;
  executable_run_options
    .set_allocator(&memoryAllocator)  // Argument is (se::DeviceMemoryAllocator*)
    .set_rng_seed(42);  // Hardcoding since sample, extensive algorithm here: /tf/compiler/xla/executable_run_options.h
  //  executable_run_options.set_run_id();
  //  executable_run_options.set_stream();
  //  executable_run_options.set_intra_op_thread_pool();

  // Run the compiled computation with the given arguments and options and return the result.
  // StatusOr<ScopedShapedBuffer> Run(const absl::Span<const ShapedBuffer* const> arguments, ExecutableRunOptions run_options);
  // StatusOr<ExecutionOutput>    Run(std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options);
  StatusOr<xla::ScopedShapedBuffer> sorScopedShapedBuffer = localExecutable->Run(
      (absl::Span<const xla::ShapedBuffer* const>){},  // const absl::Span<const ShapedBuffer* const> arguments,
      executable_run_options);  // xla::ExecutableRunOptions run_options);
  if (!sorScopedShapedBuffer.ok()) {
    cout << "Run failed: " << sorScopedShapedBuffer.status() << endl;
    return -1;
  }
  cout << "Run successful: " << sorScopedShapedBuffer.status() << endl;
  xla::ScopedShapedBuffer& scopedShapedBuffer = sorScopedShapedBuffer.ValueOrDie();
  cout << "ShapedBuffer.ToString: " << scopedShapedBuffer.ToString() << endl;
  cout << "Streaming ShapedBuffer to cout" << endl;
  cout << scopedShapedBuffer << endl;

  se::DeviceMemoryBase rootBuffer = scopedShapedBuffer.root_buffer();
  se::DeviceMemory<int> deviceMemory(rootBuffer);

  cout << "DeviceMemory" << endl
    << "  ElementCount: " << deviceMemory.ElementCount() << endl
    << "  IsScalar: " << deviceMemory.IsScalar() << endl;

  int* r = (int*)rootBuffer.opaque();
  cout << "Now de-referencing result pointer" << endl;
  cout << "Result:, r0: " << r[0] << ", r1: " << r[1] << endl;


  // ShapedBuffer (tf/compiler/xla/service/shaped_buffer.h)
  //    const Shape& on_host_shape()
  //    const Shape& on_device_shape()
  //    const se::DeviceMemoryBase& root_buffer()
  //    const se::DeviceMemoryBase& buffer(const ShapeIndex& index)  // Buffer at ShapeUtil::GetSubshape index 
  //    const ShapeTree<se::DeviceMemoryBase>& buffers()  // Returns ShapeTree containing all the device addresses in ShapedBuffer
  //    
  // ScopedShapedBuffer : ShapedBuffer (tf/compiler/xla/service/shaped_buffer.h)
  //    se::DeviceMemoryAllocator* memory_allocator()



/*
XlaComputationLaunchContext::PopulateInputs [1] [] [] [ 452 @ ./tensorflow/compiler/jit/xla_launch_util.cc]
num_elements [1] [] [] [ 687 @ ./tensorflow/compiler/xla/array.h]
Array [1] [] [] [ 268 @ ./tensorflow/compiler/xla/array.h]
Array2D [1] [] [] [ 209 @ ./tensorflow/compiler/xla/array2d.h]

DeviceAssignment [1] [] [] [ 210 @ ./tensorflow/compiler/xla/service/computation_placer.h]
RunId::RunId [1] [] [] [ 191 @ ./tensorflow/compiler/xla/executable_run_options.cc]
RunId [1] [] [] [ 224 @ ./tensorflow/compiler/xla/executable_run_options.h]
ExecutableRunOptions::set_run_id [1] [] [] [ 355 @ ./tensorflow/compiler/xla/executable_run_options.cc]
ExecutableRunOptions::set_stream [1] [] [] [ 245 @ ./tensorflow/compiler/xla/executable_run_options.cc]
ExecutableRunOptions::set_allocator [1] [] [] [ 228 @ ./tensorflow/compiler/xla/executable_run_options.cc]
ExecutableRunOptions::set_intra_op_thread_pool [1] [] [] [ 277 @ ./tensorflow/compiler/xla/executable_run_options.cc]
GetXLARandomSeed [1] [] [] [ 951 @ ./tensorflow/compiler/tf2xla/tf2xla_util.cc]
ExecutableRunOptions::set_rng_seed [1] [] [] [ 342 @ ./tensorflow/compiler/xla/executable_run_options.cc]
LocalExecutable::Run [1] [] [] [ 371 @ ./tensorflow/compiler/xla/client/local_client.cc]

ConsumeResult [1] [] [] [ 436 @ ./tensorflow/compiler/xla/service/executable.h]
XlaComputationLaunchContext::PopulateOutputs [1] [] [] [ 671 @ ./tensorflow/compiler/jit/xla_launch_util.cc]
*/



/*
XlaComputation::GetProgramShape [1] [] [] [ 195 @ ./tensorflow/compiler/xla/client/xla_computation.cc]
result [1] [] [] [ 643 @ ./tensorflow/compiler/xla/shape.h]

XlaCompilationCache::BuildExecutable [1] [] [] [ 477 @ ./tensorflow/compiler/jit/xla_compilation_cache.cc]
   LocalClient::default_device_ordinal [1] [] [] [ 558 @ ./tensorflow/compiler/xla/client/local_client.cc]
ExecutableBuildOptions::set_device_ordinal [1] [] [] [ 211 @ ./tensorflow/compiler/xla/client/executable_build_options.cc]
ExecutableBuildOptions::set_result_layout [1] [] [] [ 236 @ ./tensorflow/compiler/xla/client/executable_build_options.cc]
ExecutableBuildOptions::set_device_allocator [1] [] [] [ 195 @ ./tensorflow/compiler/xla/client/executable_build_options.cc]
set_alias_passthrough_params [1] [] [] [ 316 @ ./tensorflow/compiler/xla/client/executable_build_options.h]
ExecutableBuildOptions::mutable_debug_options [1] [] [] [ 225 @ ./tensorflow/compiler/xla/client/executable_build_options.cc]
LocalClient::Compile [1] [] [] [ 610 @ ./tensorflow/compiler/xla/client/local_client.cc]

LocalClient::default_device_ordinal [1] [] [] [ 558 @ ./tensorflow/compiler/xla/client/local_client.cc]
is_on_xla_device [1] [] [] [ 250 @ ./tensorflow/compiler/jit/xla_platform_info.h]
executable [1] [] [] [ 247 @ ./tensorflow/compiler/xla/client/local_client.h]
module [1] [] [] [ 621 @ ./tensorflow/compiler/xla/service/executable.h]
input_output_alias_config [1] [] [] [ 542 @ ./tensorflow/compiler/xla/service/hlo_module.h]
*/

  cout << "Goodbye World from XLA directory" << endl;
  return 0;
}

//------------------------
void printShape(xla::Shape shape) {

  std::cout << "*** XLATesting.printShape" << std::endl;

  std::stringstream proto_sstr;
  xla::ShapeProto shape_proto = shape.ToProto();
  shape_proto.SerializeToOstream(&proto_sstr);
  std::string proto_str = proto_sstr.str();
  // std::stringstream sstr(std::string(stringArr,19));

  cout << "  ToString: [" << shape.ToString(true) << "]" << std::endl
    << "  Proto: [" << proto_str << "]" << std::endl;
  std::cout << "  element_type: "<< shape.element_type() << " (S32 is: " << xla::S32 << ")" << std::endl;
  std::cout << "  rank: " << shape.rank() << std::endl;
  std::cout << "  is_static: " << shape.is_static() << std::endl;
  std::cout << "  is_dynamic: " << shape.is_dynamic() << std::endl;
  std::cout << "  has_layout: " << shape.has_layout() << std::endl;
  std::cout << "  IsInteger: " << shape.IsInteger() << std::endl;
  std::cout << "  IsArray: " << shape.IsArray() << std::endl;
  std::cout << "  IsTuple: " << shape.IsTuple() << std::endl;
  std::cout << "  IsToken: " << shape.IsToken() << std::endl;
  std::cout << "  IsOpaque: " << shape.IsOpaque() << std::endl;
  std::cout << "  dimensions_size: " << shape.dimensions_size() << std::endl;
  for (int i=0; i < shape.dimensions_size(); i++) {
    std::cout << "    " << i << ", dimension: " << shape.dimensions(i)
      << ", is_dynamic: " << shape.is_dynamic_dimension(i) << std::endl;
  }
  std::cout << "  tuple_shapes_size: " << shape.tuple_shapes_size() << std::endl;
  for (int i=0; i < shape.tuple_shapes_size(); i++) {
    std::cout << "    " << i << ", tuple_shapes.ToString: " << shape.tuple_shapes(i).ToString() << std::endl;
  }
}

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

LocalClient : Client (tf/compiler/xla/client/local_client.h)
LocalExecutable (tf/compiler/xla/client/local_client.h)
ExecutableBuildOptions (tf/compiler/xla/client/executable_build_options.h)

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
