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
class MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc() {
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

#include "tensorflow/compiler/aot/compile.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "llvm-c/Target.h"
#include "llvm/Support/ManagedStatic.h"
#include "tensorflow/compiler/aot/codegen.h"
#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/aot/quantize.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tfcompile {

static llvm::ManagedStatic<QuantizeXlaFn> quantize_xla;

bool RegisterQuantizeFn(const QuantizeXlaFn& fn) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/aot/compile.cc", "RegisterQuantizeFn");

  if (*quantize_xla) return false;
  *quantize_xla = fn;
  return true;
}

namespace {

// Compiles the XLA computation into executable code.
Status CompileXla(xla::CompileOnlyClient* client,
                  const xla::XlaComputation& computation,
                  const xla::cpu::CpuAotCompilationOptions& aot_opts,
                  CompileResult* compile_result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/aot/compile.cc", "CompileXla");

  // Retrieves arg and result layouts from the computation.
  // TODO(toddw): Should we let the user choose the major/minor ordering?
  xla::StatusOr<std::unique_ptr<xla::ProgramShape>> pshape_or =
      client->GetComputationShape(computation);
  if (!pshape_or.ok()) {
    return errors::Unknown("Couldn't get XLA program shape: ",
                           pshape_or.status().error_message());
  }
  compile_result->program_shape = pshape_or.ValueOrDie()->ToProto();
  xla::ProgramShapeProto* pshape = &compile_result->program_shape;

  // AotXlaComputationInstance::argument_layouts is a vector of Shape
  // pointers. Accumulate the Shape objects themselves in a separate vector
  // while building the vector of pointers.
  std::vector<const xla::Shape*> arg_layout_ptrs(pshape->parameters_size());
  std::vector<xla::Shape> arg_layouts(pshape->parameters_size());
  for (int i = 0; i < pshape->parameters_size(); ++i) {
    arg_layouts[i] = xla::Shape(*pshape->mutable_parameters(i));
    arg_layout_ptrs[i] = &arg_layouts[i];
  }
  xla::CompileOnlyClient::AotXlaComputationInstance instance;
  instance.computation = &computation;
  instance.argument_layouts = std::move(arg_layout_ptrs);
  xla::Shape result_shape(pshape->result());
  instance.result_layout = &result_shape;
  xla::StatusOr<std::vector<std::unique_ptr<xla::AotCompilationResult>>>
      aot_or = client->CompileAheadOfTime({instance}, aot_opts);
  if (!aot_or.ok()) {
    return errors::Unknown("XLA compilation failed: ",
                           aot_or.status().error_message());
  }
  compile_result->aot =
      xla::unique_ptr_static_cast<xla::cpu::CpuAotCompilationResult>(
          std::move(aot_or.ValueOrDie().back()));
  compile_result->entry_point = aot_opts.entry_point_name();
  compile_result->pointer_size =
      xla::CompileOnlyClient::PointerSizeForTriple(aot_opts.triple());
  return Status::OK();
}

}  // namespace

Status CompileGraph(GraphDef graph_def, const tf2xla::Config& config,
                    const MainFlags& flags, CompileResult* compile_result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_2(mht_2_v, 283, "", "./tensorflow/compiler/aot/compile.cc", "CompileGraph");

  // Converts the graph into an XLA computation, and compiles the
  // computation.
  // TODO(toddw): Should we let the user pick the XLA cpu vs. gpu client?
  se::Platform* cpu_platform =
      se::MultiPlatformManager::PlatformWithName("Host").ValueOrDie();
  xla::CompileOnlyClient* client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform)
          .ValueOrDie();
  xla::XlaComputation computation;

  bool use_mlir_hlo_lowering = false;
  bool use_mlir_bridge = false;
  if (!flags.mlir_components.empty() && flags.mlir_components != "None") {
    for (auto component : absl::StrSplit(flags.mlir_components, ',')) {
      if (component == "Bridge") {
        use_mlir_bridge = true;
      } else if (component == "HloLowering") {
        use_mlir_hlo_lowering = true;
      } else {
        return errors::Unknown("Unknown mlir_component ", component);
      }
    }
  }
  if (use_mlir_bridge) {
    TF_RETURN_IF_ERROR(ConvertGraphDefToXlaViaMlir(
        graph_def, config, &computation, flags.debug_info,
        flags.debug_info_path_begin_marker));
  } else {
    TF_RETURN_IF_ERROR(ConvertGraphDefToXla(std::move(graph_def), config,
                                            client, &computation));
  }

  if (flags.experimental_quantize && *quantize_xla) {
    TF_RETURN_IF_ERROR((*quantize_xla)(config, &computation));
  }

  if (!flags.out_session_module.empty()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloSnapshot> module,
                        computation.Snapshot());
    // Serialize the HloSnapshot deterministically so that all the outputs of a
    // tf_library genrule are deterministic.
    const size_t size = module->ByteSizeLong();
    auto serialized = absl::make_unique<char[]>(size);
    TF_RET_CHECK(
        SerializeToBufferDeterministic(*module, serialized.get(), size));
    TF_RETURN_IF_ERROR(
        WriteStringToFile(Env::Default(), flags.out_session_module,
                          absl::string_view(serialized.get(), size)));
  }
  xla::cpu::CpuAotCompilationOptions aot_opts(
      flags.target_triple, flags.target_cpu, flags.target_features,
      flags.entry_point,
      xla::cpu::CpuAotCompilationOptions::RelocationModel::BigPic);
  aot_opts.set_use_mlir_hlo_lowering(use_mlir_hlo_lowering);

  return CompileXla(client, computation, aot_opts, compile_result);
}

static Status ReadProtoFile(const string& fname, protobuf::Message* proto) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_3(mht_3_v, 346, "", "./tensorflow/compiler/aot/compile.cc", "ReadProtoFile");

  if (absl::EndsWith(fname, ".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

static absl::once_flag targets_init;

static void InitializeTargets() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_4(mht_4_v, 359, "", "./tensorflow/compiler/aot/compile.cc", "InitializeTargets");

  // Initialize all LLVM targets so we can cross compile.
#if TF_LLVM_AARCH64_AVAILABLE
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
#endif
#if TF_LLVM_S390X_AVAILABLE
  LLVMInitializeSystemZTarget();
  LLVMInitializeSystemZTargetInfo();
  LLVMInitializeSystemZTargetMC();
  LLVMInitializeSystemZAsmPrinter();
#endif
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmPrinter();
  LLVMInitializePowerPCTarget();
  LLVMInitializePowerPCTargetInfo();
  LLVMInitializePowerPCTargetMC();
  LLVMInitializePowerPCAsmPrinter();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
}

// Replaces {{tag.type tag.name}} in the error message with tag_name.
// TODO(bixia): We currently only handlge tag.type == "node".
//
// In the error message, a graph node is represented as {{tag.type, tag.name}},
// to allow a Python debugger to insert source information about the graph node.
// For example, a Python add expression may be represented as
// {{node, x_y_sum}} = Add(x, y) in the error message. See routine interpolate
// in tensorflow/python/framework/error_interpolation.py for more detail.
static std::string InterpolateErrorMessage(std::string message) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_5(mht_5_v, 399, "", "./tensorflow/compiler/aot/compile.cc", "InterpolateErrorMessage");

  // See _NAME_REGEX in tensorflow/python/framework/error_interpolation.py
  // Change "prefix {{node tag.name}} suffix" to "prefix tag.name suffix".
  static LazyRE2 pattern{"(.*){{node (.*)}}(.*)"};
  RE2::GlobalReplace(&message, *pattern, "\\1\\2\\3");

  return message;
}

Status Main(const MainFlags& flags) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSaotPScompileDTcc mht_6(mht_6_v, 411, "", "./tensorflow/compiler/aot/compile.cc", "Main");

  absl::call_once(targets_init, &InitializeTargets);

  // Process config.
  tf2xla::Config config;
  if (flags.config.empty()) {
    return errors::InvalidArgument("Must specify --config");
  }
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.config, &config));
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  if (flags.dump_fetch_nodes) {
    std::set<string> nodes;
    for (const tf2xla::Fetch& fetch : config.fetch()) {
      nodes.insert(fetch.id().node_name());
    }
    std::cout << absl::StrJoin(nodes, ",");
    return Status::OK();
  }

  // Read and initialize the graph.
  if (flags.graph.empty()) {
    return errors::InvalidArgument("Must specify --graph");
  }
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.graph, &graph_def));
  CompileResult compile_result;

  Status status =
      CompileGraph(std::move(graph_def), config, flags, &compile_result);
  if (!status.ok()) {
    return errors::CreateWithUpdatedMessage(
        status, InterpolateErrorMessage(status.error_message()));
  }

  // Write output files.
  Env* env = Env::Default();
  const std::vector<char>& obj = compile_result.aot->object_file_data();
  TF_RETURN_IF_ERROR(
      WriteStringToFile(env, flags.out_function_object,
                        absl::string_view(obj.data(), obj.size())));
  CodegenOpts codegen_opts;
  codegen_opts.gen_name_to_index = flags.gen_name_to_index;
  codegen_opts.gen_program_shape = flags.gen_program_shape;
  codegen_opts.target_triple = flags.target_triple;
  if (flags.cpp_class.empty()) {
    return errors::InvalidArgument("Must specify --cpp_class");
  }
  codegen_opts.gen_hlo_profile_printer_data =
      xla::GetDebugOptionsFromFlags().xla_hlo_profile();
  TF_RETURN_IF_ERROR(ParseCppClass(flags.cpp_class, &codegen_opts.class_name,
                                   &codegen_opts.namespaces));

  MetadataResult metadata_result;
  TF_RETURN_IF_ERROR(
      GenerateMetadata(codegen_opts, compile_result, &metadata_result));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_metadata_object,
                                       metadata_result.object_file_data));
  string header;
  TF_RETURN_IF_ERROR(GenerateHeader(codegen_opts, config, compile_result,
                                    metadata_result, &header));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_header, header));
  return Status::OK();
}

}  // namespace tfcompile
}  // namespace tensorflow
