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

/*
Next:
   - Output files
      Check output files
      Can we disable compiler passes, e.g. constant folding? How are passes decided?
      What part of constant folding performs our change, print it
   - Look at a pass, and how to write your own pass and add it?
   - How to take HLO proto and run our program?
   - Creatig your own backend
   - Additional flags throughout the entire thing
   - GPUs and other devices and copying memory etc
   - SPMD and other distribution mechanisms

Notes
  - HLO output
    export XLA_FLAGS=--xla_dump_to=./output  # Before running will create output with debug files
    Use $ ls -lrt  # Displays file in cretion order (oldest first)
       ($ ls --full-time  # May on Unix system give time resolution more granular than seconds)
    File creation order (from function call WriteStringToFile): 
       1. "/tmp/xla_output/module_0000.UniqueNameHere.14.before_optimizations.txt"
       2. "/tmp/xla_output/module_0000.UniqueNameHere.14.cpu_after_optimizations.txt"
       3. "/tmp/xla_output/module_0000.UniqueNameHere.14.cpu_after_optimizations-buffer-assignment.txt"
       3. "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-no-opt.ll"
       4. "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-no-opt-noconst.ll"
       5. "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-with-opt.ll"
       7. "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-with-opt-noconst.ll"
       8. "/tmp/xla_output/module_0000.UniqueNameHere.14.o"
*/

#include <iostream>
#include "tensorflow/compiler/xla/client/executable_build_options.h"  // ExecutableBuildOptions
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
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"  // Defines PrimitiveType

using namespace std;
using namespace tensorflow;

Status PrintShape(const xla::Shape& shape);
Status PrintProgramShape(const xla::ProgramShape& program_shape);
Status CreateAndRunProgram(const string&  title, int test_case);
Status RunProgram(
  vector<xla::ExecutionInput> execution_inputs,
  const xla::ExecutableRunOptions& executable_run_options, 
  unique_ptr<xla::LocalExecutable>& local_executable);


//------------------------
int main(int argc, char* argv[]) {
  cout << "Hello World from XLA directory" << endl;

  Status r = Status::OK();
  r = CreateAndRunProgram("Executing Sequence", 0);
  if (!r.ok()) {
    cout << "Error when executing: " << r << endl;
  } else {
     cout << "Execution finished successfully" << endl;
  }

  cout << "Goodbye World from XLA directory" << endl;
  return 0;
}

//------------------------
Status CreateAndRunProgram(const string& title, int test_case) {
  cout << "***************************************************************************************" << endl;
  cout << title << ": " << test_case << endl;

  se::host::HostPlatform platform; 
  StatusOr<xla::Compiler*> sor_compiler = xla::Compiler::GetForPlatform(&platform);   
  if (!sor_compiler.ok()) {
    cout << "Error, could not get compiler: " << sor_compiler.status() << endl;
    return sor_compiler.status();
  }
  cout << "Compiler successfully retrieved for platform" << endl;
  xla::Compiler* compiler = sor_compiler.ValueOrDie();

  xla::LocalClientOptions local_client_options(
    &platform,     // Or use default: se::Platform* platform = nullptr,
    1,             // int number_of_replicas = 1,
    -1,            // int intra_op_parallelism_threads = -1,
    absl::nullopt  //const absl::optional<std::set<int>>& allowed_devices = absl::nullopt
  );

  // nullptr for default platform according to doc
  local_client_options
    .set_platform(&platform)
    // Sets the thread pool size for parallel execution of an individual operator, default is -1?
    .set_intra_op_parallelism_threads(-1)
  ;

  // Set of device IDs for which the stream executor will be created, for the given platform.
  local_client_options
    .set_allowed_devices(/*absl::optional<std::set<int>>*/{})
  ;

  // Singleton constructor-or-accessor -- returns a client for the application to issue XLA commands on.
  StatusOr<xla::LocalClient*> sor_local_client = xla::ClientLibrary::GetOrCreateLocalClient(local_client_options);
  if (!sor_local_client.ok()) {
    cout << "Could not create LocalClient: " <<  sor_local_client.status() << endl;
    return sor_local_client.status();
  }
  cout << "LocalClient created successfully" << endl;
  xla::LocalClient* local_client = sor_local_client.ValueOrDie();

  // A convenient interface for building up computations.
  // xla::XlaBuilder xlaBuilder{"__inference_return1_5<LB>_XlaMustCompile=true,config_proto=3175580994766145631,executor_type=11160318154034397263<RB>"};
  xla::XlaBuilder xla_builder{"UniqueNameHere"};
  Status s_xla_builder = xla_builder.GetCurrentStatus();
  if (!s_xla_builder.ok()) {
    cout << "Could not create XlaBuilder: " << s_xla_builder << endl;
    return s_xla_builder;
  }
  cout << "XlaBuilder created successfully" << endl;

  cout << "Now creating operations" << endl;
  xla::XlaComputation computation_add = xla::CreateScalarAddComputation(
    /*primitive_type=*/xla::S32,  // Signed int 32 bytes (int32_t)
    &xla_builder);

  // Constant scalar with value 1
  xla::Literal r1_literal = xla::LiteralUtil::CreateR0<int>(1);
  xla::LiteralSlice r1_literal_slice = xla::LiteralSlice(r1_literal);
  xla::XlaOp operand_constant_1 = ConstantLiteral(&xla_builder, r1_literal_slice);  

  // Constant 2D array
  xla::Literal r2_literal = xla::LiteralUtil::CreateR2FromArray2D((const xla::Array2D<int>){{1,2,3},{4,5,6}});
  xla::LiteralSlice r2_literal_slice = xla::LiteralSlice(r2_literal);
  xla::XlaOp operand_constant_array2d = ConstantLiteral(&xla_builder, r2_literal_slice);

  // Reduce operation of constant 2D array, adding constant scalar (1) to each dimension reduced
  xla::XlaOp operation_reduce = xla::Reduce(
    &xla_builder,
    /*operands=*/{operand_constant_array2d},
    /*init_values=*/{operand_constant_1},
    /*computation=*/computation_add,
    /*dimensions_to_reduce=*/{0}  // Results. 1 (row-based): [7, 16]. 0 (col-based): [6, 8, 10]
  );

  // Parameter of shape S32[3], using f/compiler/xla/shape_utils.h
  xla::Shape operand_param_array1d_shape = xla::ShapeUtil::MakeShape(xla::S32, {3});
  xla::XlaOp operand_param_array1d = Parameter(
    &xla_builder,
    /*int64_t parameter_number=*/0,
    /*const shape& shape=*/operand_param_array1d_shape,
    /*const std::string& name=*/"array_to_reduce");

  // Map operation
  xla::XlaOp operation_map = xla::Map(
     &xla_builder,
     {operation_reduce, operand_param_array1d},
     computation_add,
     {0}
  );

  // Print the program shape
  StatusOr<xla::ProgramShape> sor_program_shape = xla_builder.GetProgramShape();
  if (!sor_program_shape.ok()) {
    cout << "Could not retrieve program shape" << endl;
    return sor_program_shape.status();
  }
  Status program_shape_status = PrintProgramShape(sor_program_shape.ValueOrDie());
  xla::Shape program_result_layout = xla_builder.GetProgramShape().ValueOrDie().result();
  
  // Build the computation
  StatusOr<xla::XlaComputation> sor_build = xla_builder.Build();
  if (!sor_build.ok()) {
    cout << "Error during XlaBuilder::Build: " << sor_build.status() << endl;
    return sor_build.status();
  }
  cout << "XlaBuilder::Build: succesfull" << endl;
  xla::XlaComputation& xla_computation = sor_build.ValueOrDie();
  // PrintProgramShape, xla_builder.GetProgramShape() is illegal after XlaBuilder::Build 

  // Set the executable build options
  xla::ExecutableBuildOptions executable_build_options;  // tf/compiler/xla/client/executable_build_options.h
  int device_ordinal = local_client->default_device_ordinal(); 
  executable_build_options.set_device_ordinal(device_ordinal);
  executable_build_options.set_result_layout(program_result_layout);
  // exeutableBuildOptions.set_device_allocator();
  // executableBuildOptions.mutable_debug_options()->set_xla_detailed_logging_and_dumping(options.detailed_logging);

  // Compile local executable
  // Here is where majority of things happen
  // *** Todo: Where is CPU involved in all of this
 
  // LocalClient::Compile
  //    LocalService::CompileExecutables
  //       Service::BuildExecutable
  //          HloModule::CreateFromProto
  //          HloVerifier::Run
  //          DumpHloModuleIfEnabled name: "before_optimizations" 941 @ ./tensorflow/compiler/xla/service/dump.cc
  //          compiler 260 @ ./tensorflow/compiler/xla/service/backend.h
  //          // CpuCompiler::RunHloPasses runs majority ~100k lines of trace code
  //          CpuCompiler::RunHloPasses 1042 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc
  //             config [1] [] [] [ 450 @ ./tensorflow/compiler/xla/service/hlo_module.h]
  //             CompilerTargetOptions [1] [] [] [ 914 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
  //             config [1] [] [] [ 450 @ ./tensorflow/compiler/xla/service/hlo_module.h]
  //             CodeGenOptLevel [1] [] [] [ 924 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
  //                debug_options [1] [] [] [ 427 @ ./tensorflow/compiler/xla/service/hlo_module_config.h]
  //             SimpleOrcJIT::InferTargetMachineForJIT [1] [] [] [ 247 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]
  //             // Next line trace ~100k of code
  //             CpuCompiler::RunHloPasses [1] [] [] [ 885 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
  //                HloModule::ToProto [1] [] [] [ 492 @ ./tensorflow/compiler/xla/service/hlo_module.cc]  // 
  //                LLVMTargetMachineFeatures [1] [] [] [ 243 @ ./tensorflow/compiler/xla/service/cpu/target_machine_features.h]
  //                // Next line trace ~100k of code
  //                CpuCompiler::RunHloPassesThroughLayoutAssn [1] [] [] [ 631 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]




  StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>> sor_local_executables =
    local_client->Compile(
      xla_computation,
      /*const absl::Span<const Shape* const> argument_layouts=*/{&operand_param_array1d_shape},
      executable_build_options);
  if (!sor_local_executables.ok()) {
    cout << "Error LocalClient::Compiler failed: " << sor_local_executables.status() << endl;
    return sor_local_executables.status();
  }
  cout << "LocalClient::Compile successful" << endl;
  std::vector<std::unique_ptr<xla::LocalExecutable>>& local_executables = sor_local_executables.ValueOrDie();
  cout << "Number of executables: " << local_executables.size() << endl;
  if (local_executables.size() != 1) {
    cout << "Error, was expecting one local executables" << endl;
    return Status(tensorflow::error::NOT_FOUND, ">1 executables was unexpected");  // tf/core/protobuf/error_codes.pb.h
  }
  std::unique_ptr<xla::LocalExecutable>& local_executable = local_executables[0];
  cout << "Retrieved a single target executable, as expected" << endl;

  // Create the stream executor
  se::PluginConfig plugin_config;  // Use the defaults
  std::unique_ptr<se::internal::StreamExecutorInterface> stream_executor_impl(new se::host::HostExecutor(plugin_config));
  se::StreamExecutor stream_executor(&platform, std::move(stream_executor_impl), device_ordinal);
  se::StreamExecutorMemoryAllocator memory_allocator(&stream_executor);

  // Create run options
  xla::ExecutableRunOptions executable_run_options;
  executable_run_options
    .set_allocator(&memory_allocator)  // Argument is (se::DeviceMemoryAllocator*)
    .set_rng_seed(42);  // Hardcoding since sample, extensive algorithm here: /tf/compiler/xla/executable_run_options.h
  //  executable_run_options.set_run_id();  // what is this
  //  executable_run_options.set_stream();
  //  executable_run_options.set_intra_op_thread_pool();

  // Run our program repeatedly, changing the input parameter across runs
//  for (int i=0; i < 10000; i++) {
    int i=10;
    cout << "--------------" << endl << "Execution Run: " << i << endl;
    xla::ExecutionInput execution_input(operand_param_array1d_shape);
    int argument_1[] = {5+i, 6+i, 7+i};
    se::DeviceMemoryBase device_memory = se::DeviceMemory<int*>::MakeFromByteSize(argument_1, sizeof(argument_1));
    execution_input.SetUnownedBuffer(
      xla::ShapeIndex{},
      xla::MaybeOwningDeviceMemory(device_memory));
    vector<xla::ExecutionInput> execution_inputs;
    execution_inputs.push_back(std::move(execution_input)); 
    Status s_run_program = RunProgram(std::move(execution_inputs), executable_run_options, local_executable);
    if (!s_run_program.ok()) {
      cout << "Error on run: " << i << ", message: " << s_run_program << endl;
      return s_run_program;
    }
  // }

  return Status::OK();
}

//------------------------
Status RunProgram(
  vector<xla::ExecutionInput> execution_inputs,
  const xla::ExecutableRunOptions& executable_run_options, 
  std::unique_ptr<xla::LocalExecutable>& local_executable) {

  cout << "Next statement will Run the LocalExecutable" << endl;

  StatusOr<xla::ExecutionOutput> sor_execution_output = local_executable->Run(
    std::move(execution_inputs),
    executable_run_options);
  if (!sor_execution_output.ok()) {
    cout << "LocalExecutable::Run failed: " << sor_execution_output.status() << endl;
    return sor_execution_output.status();
  }
  cout << "Run was successful: " << sor_execution_output.status() << endl;
  xla::ExecutionOutput& execution_output = sor_execution_output.ValueOrDie();
  const xla::ScopedShapedBuffer& scoped_shaped_buffer = execution_output.Result();
  cout << "Resulting ShapedBuffer.ToString: " << scoped_shaped_buffer.ToString();

  se::DeviceMemoryBase root_buffer = scoped_shaped_buffer.root_buffer();
  se::DeviceMemory<int> device_memory(root_buffer);
  cout << "DeviceMemory, ElementCount: " << device_memory.ElementCount()
    << ", IsScalar: " << device_memory.IsScalar() << endl;

  cout << "Result: ";
  int* r = static_cast<int*>(root_buffer.opaque());
  for (int i=0; i < device_memory.ElementCount(); i++) {
    cout << *(r+i);
    if (i+1 < device_memory.ElementCount()) {
      cout << ", ";
    }
  }
  cout << endl;

  return Status::OK();
}

//------------------------
Status PrintProgramShape(const xla::ProgramShape& program_shape) {
  cout << "ProgramShape.ToString: " << program_shape.ToString() << endl;
  xla::ProgramShapeProto program_shape_proto = program_shape.ToProto();
  string program_shape_proto_str;
  // https://pages.cs.wisc.edu/~starr/bots/Undermind-src/html/classgoogle_1_1protobuf_1_1io_1_1ZeroCopyOutputStream.html
  google::protobuf::io::StringOutputStream program_shape_proto_ostr(&program_shape_proto_str);
  bool success = google::protobuf::TextFormat::Print(program_shape_proto, &program_shape_proto_ostr);
  if (success) {
    cout << "ProgramShapeProto, success: " << success << ", String: [" << program_shape_proto_str << "]" << endl;
  } else {
    cout << "Error: Could not successfully Print ProgramShapeProto" << endl;
  }
  return Status::OK();
}

//------------------------
Status PrintShape(const xla::Shape& shape) {

 // tf/compiler/xla/shape_utils.h is great

/*
  // Creates an opaque shape. These are generally used for threading a context
  // into a custom operation.
  static Shape MakeOpaqueShape();

  // Creates a token shape. Values of this shape are used for ordering
  // side-effecting operations.
  static Shape MakeTokenShape();
*/

  cout << "*** XLATesting.printShape" << endl;

  stringstream proto_sstr;
  xla::ShapeProto shape_proto = shape.ToProto();
  shape_proto.SerializeToOstream(&proto_sstr);
  string proto_str = proto_sstr.str();
  // std::stringstream sstr(std::string(stringArr,19));

  cout << "  ToString: [" << shape.ToString(true) << "]" << endl
    << "  Proto: [" << proto_str << "]" << endl;
  cout << "  element_type: "<< shape.element_type() << " (S32 is: " << xla::S32 << ")" << endl;
  cout << "  rank: " << shape.rank() << endl;
  cout << "  is_static: " << shape.is_static() << endl;
  cout << "  is_dynamic: " << shape.is_dynamic() << endl;
  cout << "  has_layout: " << shape.has_layout() << endl;
  cout << "  IsInteger: " << shape.IsInteger() << endl;
  cout << "  IsArray: " << shape.IsArray() << endl;
  cout << "  IsTuple: " << shape.IsTuple() << endl;
  cout << "  IsToken: " << shape.IsToken() << endl;
  cout << "  IsOpaque: " << shape.IsOpaque() << endl;
  cout << "  dimensions_size: " << shape.dimensions_size() << endl;
  for (int i=0; i < shape.dimensions_size(); i++) {
    cout << "    " << i << ", dimension: " << shape.dimensions(i)
      << ", is_dynamic: " << shape.is_dynamic_dimension(i) << endl;
  }
  cout << "  tuple_shapes_size: " << shape.tuple_shapes_size() << endl;
  for (int i=0; i < shape.tuple_shapes_size(); i++) {
    cout << "    " << i << ", tuple_shapes.ToString: " << shape.tuple_shapes(i).ToString() << endl;
  }


}

/*
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

  // StreamExecutor manages a single device, in terms of executing work 
  //   class StreamExecutor (tf/stream_executor/stream_executor_pimpl.h)
  //     port::Status Init(DeviceOptions device_options);
  //
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

//-------------






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
