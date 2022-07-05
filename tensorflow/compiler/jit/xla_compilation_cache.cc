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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc() {
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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <variant>

#include "tensorflow/compiler/mlir/mlir_bridge_rollout_policy.h"
#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/debug_event.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/dump_graph.h"

#include<iostream>
#include <sstream>
#include <strstream>

namespace tensorflow {
namespace {

using TensorTypeAndShape = XlaCompilationCache::Signature::TensorTypeAndShape;

constexpr char kXlaSerializedCacheKeySeparator[] = "__";

// Functor that converts a Signature's arg to a human readable string.
struct SignatureHumanStringAppender {
  explicit SignatureHumanStringAppender(string* dest) : dest(dest) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_0(mht_0_v, 253, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "SignatureHumanStringAppender");
}
  string* dest;
  void operator()(const Tensor& arg) {
    absl::StrAppend(dest, "; ", arg.DebugString());
  }
  void operator()(const TensorTypeAndShape& arg) {
    absl::StrAppend(dest, ",", DataTypeString(arg.first));
    absl::StrAppend(dest, " [", absl::StrJoin(arg.second, ","), "]");
  }
};

// Functor that compares the arg values of two different signatures. Returns
// true when the args are not equal.
struct SignatureNotEqual {
  bool operator()(const Tensor& arg, const Tensor& other) {
    return arg.dtype() != other.dtype() || arg.shape() != other.shape() ||
           arg.tensor_data() != other.tensor_data();
  }
  bool operator()(const TensorTypeAndShape& arg,
                  const TensorTypeAndShape& other) {
    return arg.first != other.first || arg.second != other.second;
  }
  bool operator()(const Tensor& arg, const TensorTypeAndShape& other) {
    return true;
  }
  bool operator()(const TensorTypeAndShape& arg, const Tensor& other) {
    return true;
  }
};

// Functor that incrementally computes a Signature's hash given its current hash
// and one of its args.
struct SignatureHashCombiner {
  explicit SignatureHashCombiner(const uint64 h) : h(h) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_1(mht_1_v, 289, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "SignatureHashCombiner");
}
  uint64 h;
  uint64 operator()(const Tensor& arg) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.dtype())));
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
    for (int dim = 0; dim < arg.dims(); ++dim) {
      h = Hash64Combine(h, std::hash<int>()(arg.dim_size(dim)));
    }
    return h;
  }
  uint64 operator()(const TensorTypeAndShape& arg) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.size()));
    for (int dim : arg.second) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
    return h;
  }
};

std::string XlaSerializedCacheKeyToString(const XlaSerializedCacheKey& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_2(mht_2_v, 313, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaSerializedCacheKeyToString");

  return absl::StrCat(
      key.prefix(), key.prefix().empty() ? "" : kXlaSerializedCacheKeySeparator,
      key.signature_fingerprint(), kXlaSerializedCacheKeySeparator,
      key.cluster_fingerprint(), kXlaSerializedCacheKeySeparator,
      key.device_type());
}

}  // namespace

constexpr int64_t XlaCompilationCache::kDefaultCompilationThreshold;
constexpr int64_t
    XlaCompilationCache::AsyncCompilationState::kNumCompilerThreads;
constexpr int64_t
    XlaCompilationCache::AsyncCompilationState::kMaxNumOngoingCompilations;

XlaCompilationCache::XlaCompilationCache(Config config,
                                         xla::LocalClient* client,
                                         DeviceType device_type)
    : client_(client),
      device_type_(std::move(device_type)),
      disable_strict_signature_checks_(config.disable_strict_signature_checks),
      persistance_prefix_(config.persistance_prefix),
      persistent_cache_directory_(config.persistent_cache_directory) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_3(mht_3_v, 339, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::XlaCompilationCache");
}

XlaCompilationCache::~XlaCompilationCache() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_4(mht_4_v, 344, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::~XlaCompilationCache");

  // Ensure any use of our programs have completed by waiting for all stream
  // executors to complete.
  for (auto* executor : client_->backend().stream_executors()) {
    bool ok = executor->SynchronizeAllActivity();
    if (!ok) {
      LOG(ERROR) << "Error synchronizing activity while waiting for all "
                    "programs to complete";
    }
  }
  // Wait for all outstanding compilations to finish.
  // Resetting the pointer explicitly in the top level destructor.
  // Without this, the pointer would be reset when the AsyncCompilationState
  // is destructed, which is dependent on the order of the members in the
  // XlaCompilationCache class, which is error prone if the order changes.
  async_compilation_state_.compiler_threads.reset();
  // TODO(b/110813685): Think about the program ownership model. Programs are
  // currently owned by the compilation cache which means we must wait for
  // program completion in the destructor. There are multiple compilation caches
  // around, which complicates things a little. Perhaps having programs be
  // shared_ptrs (an invasive change) would make the model easier to reason
  // about?
}

string XlaCompilationCache::DebugString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_5(mht_5_v, 371, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::DebugString");

  return "XLA JIT compilation cache";
}

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
string XlaCompilationCache::Signature::HumanString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_6(mht_6_v, 380, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::Signature::HumanString");

  string result = name;
  for (const auto& a : args) {
    absl::visit(SignatureHumanStringAppender(&result), a);
  }
  return result;
}

bool XlaCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (args.size() != other.args.size()) return false;
  for (int i = 0, end = args.size(); i < end; ++i) {
    if (absl::visit(SignatureNotEqual(), args[i], other.args[i])) {
      return false;
    }
  }
  return true;
}

uint64 XlaCompilationCache::Signature::Hash::operator()(
    const XlaCompilationCache::Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.args) {
    h = absl::visit(SignatureHashCombiner(h), arg);
  }
  return h;
}

StatusOr<XlaCompilationCache::Signature> XlaCompilationCache::BuildSignature(
    const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_7(mht_7_v, 413, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::BuildSignature");

  Signature signature;
  signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));

  for (const XlaCompiler::Argument& arg : args) {
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstant:
      case XlaCompiler::Argument::kConstantResource:
        signature.args.push_back(arg.constant_value);
        break;
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kResource:
        signature.args.push_back(
            TensorTypeAndShape(arg.type, arg.DimensionSizesAsInlinedVector()));
        break;
      default:
        return errors::InvalidArgument(
            "Unhandled argument kind in XlaCompilationCache: ",
            arg.HumanString());
    }
  }
  return std::move(signature);
}

static std::vector<const xla::Shape*> GetShapePointers(
    absl::Span<const xla::Shape> shapes) {
  std::vector<const xla::Shape*> shape_ptrs;
  shape_ptrs.reserve(shapes.size());
  for (const auto& shape : shapes) {
    shape_ptrs.push_back(&shape);
  }
  return shape_ptrs;
}

static xla::ExecutableBuildOptions GetBuildOptions(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result, int default_device_ordinal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_8(mht_8_v, 452, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "GetBuildOptions");

  xla::ExecutableBuildOptions build_options;
  if (result.collective_info) {
    build_options.set_num_replicas(result.collective_info->group_size);
  }
  build_options.set_device_ordinal(options.device_ordinal != -1
                                       ? options.device_ordinal
                                       : default_device_ordinal);
  build_options.set_result_layout(result.xla_output_shape);

  xla::Shape myshape = result.xla_output_shape;
  std::cout << "*** ExecutableBuildOptions GetBuildOptions" << std::endl;
  std::cout << "  Input size: " << result.xla_input_shapes.size() << std::endl;
  for (auto e: result.xla_input_shapes) {
    std::cout << "    " << e.ToString(true) << std::endl;
  }
  xla::ShapeProto myshapep = myshape.ToProto();
  std::stringstream protosstr;
  myshapep.SerializeToOstream(&protosstr);
  std::string protostr = protosstr.str();
  // std::stringstream sstr(std::string(stringArr,19));
  std::cout << "  Output_shape: " << std::endl
    << "    IsArray: " << myshape.IsArray() << std::endl
    << "    ToString: [" << myshape.ToString(true) << "]" << std::endl
    << "    Proto: [" << protostr << "]" << std::endl;
  std::cout << "    Element type: "<< myshape.element_type() << ", S32: " << xla::S32 << std::endl;
  std::cout << "    Dimensions, size: " << myshape.dimensions_size() << std::endl;
  for (int i=0; i < myshape.dimensions_size(); i++) {
    std::cout << "      " << i << ", Dimension: " << myshape.dimensions(i)
      << ", is_dynamic: " << myshape.is_dynamic_dimension(i) << std::endl;
  }
  std::cout << "Tuples" << myshape.tuple_shapes_size() << std::endl;
  for (int i=0; i < myshape.tuple_shapes_size(); i++) {
    std::cout << "      " << i << ", Tuple: " << myshape.tuple_shapes(i) << std::endl;
  }
  
  build_options.set_device_allocator(options.device_allocator.get());
  build_options.set_alias_passthrough_params(options.alias_passthrough_params);
  build_options.mutable_debug_options()->set_xla_detailed_logging_and_dumping(
      options.detailed_logging);
  if (tensorflow::OpDeterminismRequired()) {
    build_options.mutable_debug_options()->set_xla_gpu_deterministic_ops(true);
  }
  return build_options;
}

Status XlaCompilationCache::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    std::unique_ptr<xla::LocalExecutable>* executable) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_9(mht_9_v, 477, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::BuildExecutable");

  VLOG(2) << "Compiling to local executable";

  std::vector<const xla::Shape*> argument_layouts =
      GetShapePointers(result.xla_input_shapes);
  xla::ExecutableBuildOptions build_options =
      GetBuildOptions(options, result, client_->default_device_ordinal());
  TF_ASSIGN_OR_RETURN(
      auto executables,
      client_->Compile(*result.computation, argument_layouts, build_options));
  TF_RET_CHECK(executables.size() == 1);
  *executable = std::move(executables[0]);
  return Status::OK();
}

StatusOr<std::unique_ptr<xla::AotCompilationResult>>
XlaCompilationCache::BuildSerializedExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_10(mht_10_v, 498, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::BuildSerializedExecutable");

  VLOG(2) << "Compiling to local executable";

  std::vector<const xla::Shape*> argument_layouts =
      GetShapePointers(result.xla_input_shapes);
  xla::ExecutableBuildOptions build_options =
      GetBuildOptions(options, result, client_->default_device_ordinal());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::AotCompilationResult>> aot_results,
      client_->CompileAheadOfTime(*result.computation, argument_layouts,
                                  build_options));
  TF_RET_CHECK(aot_results.size() == 1);
  return std::move(aot_results[0]);
}

StatusOr<std::unique_ptr<xla::LocalExecutable>>
XlaCompilationCache::LoadExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    const std::string& serialized_aot_result) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("serialized_aot_result: \"" + serialized_aot_result + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_11(mht_11_v, 521, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::LoadExecutable");

  VLOG(2) << "Loading local executable using BEF.";

  xla::ExecutableBuildOptions build_options =
      GetBuildOptions(options, result, client_->default_device_ordinal());
  return client_->Load(serialized_aot_result, build_options);
}

Status XlaCompilationCache::Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    const XlaCompiler::CompileOptions& compile_options,
    CompileMode compile_mode,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_12(mht_12_v, 538, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::Compile");

  return CompileImpl(compile_options, options, function, args,
                     /*ctx=*/nullptr, CompileScope::kFunction, compile_mode,
                     out_compilation_result, out_executable);
}

static bool ShouldBeMegamorphic(int64_t compile_count,
                                int64_t execution_count) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_13(mht_13_v, 548, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "ShouldBeMegamorphic");

  const int64_t kCompileThreshold = 10;
  const int64_t kMinExecutionsPerCompile = 50;

  // This heuristic is trying to capture the following property: have we sunk a
  // certain minimum amount of compile time into the cluster that didn't quite
  // "pay off"?
  return compile_count > kCompileThreshold &&
         execution_count < kMinExecutionsPerCompile * compile_count;
}

StatusOr<std::unique_ptr<Graph>> CreateGraph(
    const NodeDef& node_def, absl::Span<const XlaCompiler::Argument> args,
    absl::Span<const DataType> result_types) {
  // TODO(b/74182462): We implement this by creating a new dummy Graph including
  // _Arg nodes, and let CompileGraph walk it. This could be optimized.
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // First create the actual node we care about computing.
  TF_ASSIGN_OR_RETURN(Node * main_node, graph->AddNode(node_def));

  // Create dummy _Arg nodes. Link these to `node` and also via a control
  // dependency edge to the _SOURCE node.
  for (int64_t i = 0, end = args.size(); i < end; ++i) {
    Node* node;
    string arg_name = absl::StrCat("_arg", i);
    Status status =
        NodeBuilder(arg_name, FunctionLibraryDefinition::kArgOp)
            .ControlInput(graph->source_node())
            .Attr("T", args[i].kind == XlaCompiler::Argument::kResource
                           ? DT_RESOURCE
                           : args[i].type)
            .Attr("index", i)
            .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
    graph->AddEdge(node, 0, main_node, i);
  }

  // Similarly with return values, create dummy _Retval nodes fed by `node`.
  for (int64_t i = 0, end = result_types.size(); i < end; ++i) {
    Node* node;
    string retval_name = absl::StrCat("_retval", i);
    Status status = NodeBuilder(retval_name, FunctionLibraryDefinition::kRetOp)
                        .Input(main_node, i)
                        .Attr("T", result_types[i])
                        .Attr("index", i)
                        .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
  }
  FixupSourceAndSinkEdges(graph.get());
  return graph;
}

Status XlaSingleOpToHlo(XlaCompiler* compiler,
                        const XlaCompiler::Options& options,
                        const std::vector<XlaCompiler::Argument>& args,
                        OpKernelContext* ctx,
                        const XlaCompiler::CompileOptions& compile_options,
                        XlaCompiler::CompilationResult* compilation_result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_14(mht_14_v, 609, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaSingleOpToHlo");

  std::vector<DataType> result_dtypes(ctx->num_outputs());
  for (int i = 0, end = result_dtypes.size(); i < end; ++i) {
    result_dtypes[i] = ctx->expected_output_dtype(i);
  }

  const NodeDef& node_def = ctx->op_kernel().def();
  TF_ASSIGN_OR_RETURN(auto graph, CreateGraph(node_def, args, result_dtypes));

  auto compile_with_old_bridge = [&]() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_15(mht_15_v, 621, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "lambda");

    *compilation_result = {};
    return compiler->CompileGraph(compile_options, node_def.name(),
                                  std::move(graph), args, compilation_result);
  };

  const ConfigProto* config = ctx->function_library()->config_proto();
  auto bridge_rollout = GetMlirBridgeRolloutState(
      config ? absl::optional<ConfigProto>(*config) : absl::nullopt);
  if (bridge_rollout ==
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED ||
      node_def.op() == "VarIsInitializedOp" ||
      (bridge_rollout !=
           ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED &&
       options.device_type.type_string() != DEVICE_TPU_XLA_JIT)) {
    return compile_with_old_bridge();
  }

  GraphDebugInfo debug_info;
  std::vector<std::string> control_rets;
  if (result_dtypes.empty()) {
    control_rets.push_back(node_def.name());
  }

  bool mlir_enabled = (bridge_rollout ==
                       ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED);
  VLOG(1) << "Attempting MLIR bridge."
          << (mlir_enabled ? " MLIR is explicitly enabled." : "");
  auto mlir_result = CompileGraphToXlaHlo(
      *graph, mlir::SpanToArrayRef<XlaCompiler::Argument>(args), control_rets,
      options.device_type.type_string(), compile_options.use_tuple_arg,
      /*analyse_graph=*/!mlir_enabled, *options.flib_def, debug_info,
      options.shape_determination_fns.shape_representation_fn,
      compilation_result);

  if (mlir_result.ok() || mlir_enabled) {
    return mlir_result;
  }

  VLOG(2) << "Failed second phase of the MLIR bridge. Will "
             "retry with the old bridge. MLIR bridge compilation status: "
          << mlir_result;
  return compile_with_old_bridge();
}

Status XlaCompilationCache::CompileSingleOp(
    const XlaCompiler::Options& options,
    const std::vector<XlaCompiler::Argument>& args, OpKernelContext* ctx,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_16(mht_16_v, 674, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::CompileSingleOp");

  const NodeDef& def = ctx->op_kernel().def();
  NameAttrList name;
  name.set_name(def.op());
  *name.mutable_attr() = def.attr();
  // Remove the "_class" attribute from the attribute set used to create the
  // compilation cache key. This attribute is information for the colocator
  // and causes false uniqueness between nodes.
  name.mutable_attr()->erase("_class");
  return CompileImpl(compile_options, options, name, args, ctx,
                     CompileScope::kOp, CompileMode::kStrict,
                     out_compilation_result, out_executable);
}

namespace {
// Print something that users can search for to definitively ascertain that XLA
// was used for their TF model.
//
// Prints only once to avoid spamming LOG(INFO).
void LogOnceXlaCompiledFirstCluster() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_17(mht_17_v, 696, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "LogOnceXlaCompiledFirstCluster");

  static absl::once_flag log_once;
  absl::call_once(log_once, [] {
    LOG(INFO) << "Compiled cluster using XLA!  This line is logged at most "
                 "once for the lifetime of the process.";
  });
}
}  // namespace

Status XlaCompilationCache::CompileStrict(
    const Signature& sig, Entry* entry,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::Options& options,
    const std::vector<XlaCompiler::Argument>& args,
    const NameAttrList& function, OpKernelContext* ctx, CompileScope scope) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_18(mht_18_v, 713, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::CompileStrict");

  tensorflow::Env* env = tensorflow::Env::Default();
  const uint64 compile_start_us = env->NowMicros();

  XlaCompiler compiler(options);
  entry->compile_state = CompileState::kCompiled;
  entry->compilation_status = [&] {
    if (scope == CompileScope::kOp) {
      return XlaSingleOpToHlo(&compiler, options, args, ctx, compile_options,
                              &entry->compilation_result);

    } else {
      CHECK(scope == CompileScope::kFunction);  // Crash OK
      return compiler.CompileFunction(compile_options, function, args,
                                      &entry->compilation_result);
    }
  }();
  TF_RETURN_IF_ERROR(entry->compilation_status);
  TF_RET_CHECK(entry->executable.get() == nullptr);
  TF_RET_CHECK(entry->compilation_result.computation != nullptr);

  absl::optional<XlaSerializedCacheEntry> serialized_entry;
  if (!persistent_cache_directory_.empty()) {
    const xla::HloModuleProto& hlo_module =
        entry->compilation_result.computation->proto();

    XlaSerializedCacheKey cache_key = BuildSerializedCacheKey(sig, hlo_module);

    {
      XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
          "Try loading serialized cache entry:", sig.HumanString()));
      TF_ASSIGN_OR_RETURN(serialized_entry, TryLoadSerializedEntry(cache_key));
    }

    if (serialized_entry.has_value()) {
      TF_RETURN_IF_ERROR(
          VerifyLoadedCacheEntry(cache_key, hlo_module, *serialized_entry));
    }
  }

  if (serialized_entry.has_value()) {
    VLOG(1) << "Loading cached entry for: " << sig.HumanString();
    StatusOr<std::unique_ptr<xla::LocalExecutable>> executable = LoadExecutable(
        options, entry->compilation_result, serialized_entry->executable());
    entry->compilation_status = executable.status();
    if (executable.ok()) {
      entry->executable = *std::move(executable);
    }
  } else {
    entry->compilation_status =
        BuildExecutable(options, entry->compilation_result, &entry->executable);

    // Caching is done regardless of the entry->compilation_status. To take
    // advantage of newer compilation code, a cache flush is required.
    if (!persistent_cache_directory_.empty()) {
      XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
          "Serializing and saving cache entry: ", sig.HumanString()));
      TF_ASSIGN_OR_RETURN(XlaSerializedCacheEntry serialized_entry,
                          SerializeEntry(options, sig, *entry));
      TF_RETURN_IF_ERROR(SaveSerializedEntry(std::move(serialized_entry)));
    }
  }

  const uint64 compile_end_us = env->NowMicros();
  const uint64 compile_time_us = compile_end_us - compile_start_us;
  metrics::UpdateXlaCompilationTime(compile_time_us);

  mutex_lock lock(cluster_compile_stats_mu_);
  const std::string& function_name = function.name();
  auto it = cluster_compile_stats_.find(function_name);
  const uint64 compile_time_s = compile_time_us / 1.0e6;
  it->second.compile_count++;
  it->second.cumulative_compile_time_us += compile_time_us;
  LogOnceXlaCompiledFirstCluster();
  VLOG(1) << "compiled " << function_name << " " << it->second.compile_count
          << " times, compile time: " << compile_time_us
          << " us, cumulative: " << it->second.cumulative_compile_time_us
          << " us ("
          << tensorflow::strings::HumanReadableElapsedTime(compile_time_s)
          << " / "
          << tensorflow::strings::HumanReadableElapsedTime(
                 it->second.cumulative_compile_time_us / 1.0e6)
          << ")";

  XlaJitCompilationActivity jit_compilation_activity;
  jit_compilation_activity.set_cluster_name(function_name);
  jit_compilation_activity.set_compile_count(it->second.compile_count);
  jit_compilation_activity.set_compile_time_us(compile_time_us);
  jit_compilation_activity.set_cumulative_compile_time_us(
      it->second.cumulative_compile_time_us);
  jit_compilation_activity.set_used_persistent_cache(
      serialized_entry.has_value());
  TF_RETURN_IF_ERROR(BroadcastXlaActivity(std::move(jit_compilation_activity)));

  return Status::OK();
}

Status XlaCompilationCache::CompileAsynchronous(
    const Signature& signature, Entry* entry,
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::Options& options,
    const std::vector<XlaCompiler::Argument>& args,
    const NameAttrList& function, OpKernelContext* ctx, CompileScope scope) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_19(mht_19_v, 818, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::CompileAsynchronous");

  // Explicitly capture all required data by value for async compilation.
  entry->compile_state = CompileState::kCompiling;
  {
    mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
    async_compilation_state_.num_ongoing_compilations++;
  }
  // Don't move the above code into the thread function as it synchronously
  // updates the async compilation state!

  // When the ThreadPool for the compilation cache is destroyed, it waits for
  // compilations to have finished. This means that both 'entry' and 'this' will
  // be alive for the duration of the compilation.
  // !!Pay attention when additional variables must be captured by this lambda!!
  // All values are captured by value. Make sure that all pointer values (like
  // entry) do not get freed until the lambda has finished,\.
  const std::string& function_name = function.name();
  async_compilation_state_.compiler_threads->Schedule([=] {
    Entry local_entry;
    VLOG(2) << "Starting asynchronous compilation of cluster " << function_name
            << '.';
    // We don't need to lock local_entry.mu, but do it anyway to satisfy
    // thread safety analysis.
    mutex_lock entry_lock(local_entry.mu);
    Status s = CompileStrict(signature, &local_entry, compile_options, options,
                             args, function, ctx, scope);
    VLOG(2) << "Finished asynchronous compililation of cluster "
            << function_name << '.';
    {
      mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
      async_compilation_state_.num_ongoing_compilations--;
    }
    {  // Populate original entry with compilation result.
      mutex_lock entry_lock(entry->mu);
      if (!s.ok()) {
        entry->compilation_status = s;
      } else {
        entry->compilation_status = local_entry.compilation_status;
      }
      entry->compilation_result = local_entry.compilation_result;
      entry->compile_state = local_entry.compile_state;
      entry->executable = std::move(local_entry.executable);
    }
  });
  return Status::OK();
}

bool XlaCompilationCache::ShouldCompileCluster(CompileMode compile_mode,
                                               bool is_megamorphic,
                                               bool is_first_execution,
                                               int64_t current_request_count,
                                               const NameAttrList& function) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_20(mht_20_v, 872, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::ShouldCompileCluster");

  absl::optional<int64_t> compile_threshold;
  if (compile_mode == CompileMode::kLazy) {
    compile_threshold = kDefaultCompilationThreshold;
  } else if (compile_mode == CompileMode::kAsync) {
    compile_threshold = 0;  // for now, always compile right away.
  }

  if (compile_mode == CompileMode::kStrict) {
    // Lazy compilation is disabled.
    return true;
  }

  if (is_megamorphic) {
    BroadcastOptimizationRemark(XlaOptimizationRemark::MEGAMORPHIC_FUNCTION,
                                function.name())
        .IgnoreError();
    VLOG(2) << "Not compiling cluster " << function.name()
            << " because it is megamorphic.";
    return false;
  }

  if (is_first_execution) {
    return true;
  }

  if (compile_mode == CompileMode::kAsync) {
    // Asynchronous compilation is enabled.
    mutex_lock lock(async_compilation_state_.async_compilation_state_mu);
    if (async_compilation_state_.num_ongoing_compilations >=
        async_compilation_state_.kMaxNumOngoingCompilations) {
      VLOG(2) << "Not asynchronously compiling cluster " << function.name()
              << " because of too many ongoing compilations.";
      return false;
    }
  }

  bool reached_compile_threshold = current_request_count >= *compile_threshold;
  if (!reached_compile_threshold) {
    VLOG(2) << "Not compiling cluster " << function.name()
            << " because it has not reached compile threshold; threshold is "
            << *compile_threshold << " execution count "
            << current_request_count << ".";
  }
  return reached_compile_threshold;
}

Status XlaCompilationCache::CompileImpl(
    const XlaCompiler::CompileOptions& compile_options,
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args, OpKernelContext* ctx,
    CompileScope scope, CompileMode compile_mode,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_21(mht_21_v, 928, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::CompileImpl");

  DCHECK_NE(out_executable, nullptr);
  VLOG(2) << "XlaCompilationCache::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "num_inputs=" << args.size();
    for (int i = 0, end = args.size(); i < end; i++) {
      VLOG(3) << i << ": " << args[i].HumanString();
    }
  }
  TF_ASSIGN_OR_RETURN(Signature signature, BuildSignature(function, args));

  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Find or create a cache entry.
    std::unique_ptr<Entry>& e = cache_[signature];
    if (!e) {
      e.reset(new Entry);
    }
    entry = e.get();
  }

  // We always compile a cluster the very first time it is executed.  This is an
  // optimistic guess that pays off for statically shaped TensorFlow graphs
  // (since they get the benefit of XLA right away without waiting for warmup)
  // and doesn't hurt much for dynamically shaped TensorFlow graphs (we "pay" at
  // most one cluster-compilation's worth of compile time).
  bool is_first_execution;

  // We avoid compiling clusters that have "gone megamorphic" i.e. have an
  // excessive amount of shape dynamism.
  bool is_megamorphic;

  {
    mutex_lock lock(cluster_compile_stats_mu_);
    auto it =
        cluster_compile_stats_.emplace(function.name(), ClusterCompileStats{})
            .first;
    is_first_execution = it->second.execution_count++ == 0;

    // The is_megamorphic bit is "sticky".  We assume clusters that have been
    // observed to be megamorphic once stay megamorphic forever.
    if (!it->second.is_megamorphic &&
        ShouldBeMegamorphic(/*compile_count=*/it->second.compile_count,
                            /*execution_count=*/it->second.execution_count)) {
      VLOG(1) << "Marking " << function.name()
              << " as megamorphic, compile_count=" << it->second.compile_count
              << " execution_count=" << it->second.execution_count;
      it->second.is_megamorphic = true;
    }

    is_megamorphic = it->second.is_megamorphic;
  }

  string human_signature;
  if (VLOG_IS_ON(2)) {
    human_signature = VLOG_IS_ON(3) ? signature.HumanString() : function.name();
    VLOG(2) << "Signature: " << human_signature;
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock entry_lock(entry->mu);
  int64_t current_request_count = ++entry->request_count;
  VLOG(2) << "Compilation cache entry hit: "
          << static_cast<int>(entry->compile_state)
          << " signature: " << human_signature << " with request count "
          << current_request_count;

  CompileState state = entry->compile_state;
  *out_compilation_result = nullptr;
  *out_executable = nullptr;

  // Check if the requested entry is uncompiled and return an error if
  // compilation is disabled. This will raise an error for kLazy even if we have
  // not yet hit the compilation threshold and no compilation happens this
  // round. This is to avoid non-determanism of when compilation is disallowed,
  // for example by changing the threshold.
  if (state == CompileState::kUncompiled && FailOnXlaCompilation()) {
    VLOG(1) << "XLA compilation disabled: " << function.name() << "\n"
            << absl::StrJoin(
                   args, "\n",
                   [](std::string* out, const XlaCompiler::Argument& arg) {
                     absl::StrAppend(out, " arg: ", arg.HumanString());
                   });

    return errors::Internal("XLA compilation disabled");
  }

  if (state == CompileState::kUncompiled) {
    XLA_SCOPED_LOGGING_TIMER("Compilation of XLA executable");
    if (!ShouldCompileCluster(compile_mode, is_megamorphic, is_first_execution,
                              current_request_count, function)) {
      VLOG(2) << "Not compiling for signature: " << human_signature;
      return Status::OK();
    } else if (compile_mode == CompileMode::kAsync) {
      VLOG(2) << "Queueing asynchronous compilation for signature: "
              << human_signature;
      TF_RETURN_IF_ERROR(CompileAsynchronous(signature, entry, compile_options,
                                             options, args, function, ctx,
                                             scope));
      return Status::OK();
    } else {
      VLOG(2) << "Instantly compiling for signature: " << human_signature;
      TF_RETURN_IF_ERROR(CompileStrict(signature, entry, compile_options,
                                       options, args, function, ctx, scope));
    }
  } else if (state == CompileState::kCompiling) {
    VLOG(2) << "Ongoing asynchronous compilation for signature: "
            << human_signature;
    return Status::OK();
  } else if (state == CompileState::kCompiled) {
    VLOG(2) << "Already Compiled for signature: " << human_signature;
  }

  TF_RETURN_IF_ERROR(entry->compilation_status);
  *out_compilation_result = &entry->compilation_result;
  *out_executable = entry->executable.get();
  return Status::OK();
}

XlaSerializedCacheKey XlaCompilationCache::BuildSerializedCacheKey(
    const Signature& sig, const xla::HloModuleProto& hlo_module) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_22(mht_22_v, 1057, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::BuildSerializedCacheKey");

  XlaSerializedCacheKey serialized_cache_key;
  serialized_cache_key.set_signature_fingerprint(Signature::Hash()(sig));
  serialized_cache_key.set_cluster_fingerprint(
      DeterministicProtoHash64(hlo_module));
  serialized_cache_key.set_device_type(device_type_.type_string());
  serialized_cache_key.set_prefix(persistance_prefix_);
  return serialized_cache_key;
}

Status XlaCompilationCache::VerifyLoadedCacheEntry(
    const XlaSerializedCacheKey& key, const xla::HloModuleProto& hlo_module,
    const XlaSerializedCacheEntry& entry) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_23(mht_23_v, 1072, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::VerifyLoadedCacheEntry");

  XLA_SCOPED_LOGGING_TIMER(absl::StrCat("Verifying loaded cache entry: ",
                                        hlo_module.entry_computation_name()));

  if (!AreSerializedProtosEqual(key, entry.key())) {
    VLOG(2) << "Serialized cache key does not match:\n"
            << "got:\n"
            << entry.key().DebugString() << "\nexpected:\n"
            << key.DebugString() << "\n";
    return errors::InvalidArgument("Serialized cache key does not match.");
  }

  // Perform a stricter (slower) check of the snapshot to verify that they
  // match exactly.
  if (!disable_strict_signature_checks_) {
    if (!AreSerializedProtosEqual(hlo_module, entry.hlo_module())) {
      VLOG(2) << "HLOs do not match:\n"
              << "got:\n"
              << hlo_module.DebugString() << "\nexpected:\n"
              << entry.hlo_module().DebugString() << "\n";
      return errors::InvalidArgument("Serialized HLO does not match.");
    }
  }

  if (entry.executable().empty()) {
    return errors::InvalidArgument("No binary found in serialized entry.");
  }
  return Status::OK();
}

StatusOr<XlaSerializedCacheEntry> XlaCompilationCache::SerializeEntry(
    const XlaCompiler::Options& options, const Signature& sig,
    const Entry& entry) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_24(mht_24_v, 1107, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::SerializeEntry");

  if (entry.compile_state != CompileState::kCompiled) {
    return errors::FailedPrecondition(
        "Cache entry to serialize is not compiled.");
  }
  if (entry.executable == nullptr) {
    return errors::FailedPrecondition(
        "LocalExecutable not found for cache entry to serialize.");
  }
  if (entry.executable->executable() == nullptr) {
    return errors::FailedPrecondition(
        "Executable not found for cache entry to serialize.");
  }

  XlaSerializedCacheEntry serialized_entry;
  const xla::HloModuleProto& hlo_module =
      entry.compilation_result.computation->proto();
  *serialized_entry.mutable_key() = BuildSerializedCacheKey(sig, hlo_module);
  *serialized_entry.mutable_hlo_module() = hlo_module;

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::AotCompilationResult> aot_result,
      BuildSerializedExecutable(options, entry.compilation_result));
  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  serialized_entry.set_executable(std::move(serialized));
  return serialized_entry;
}

namespace {

std::string GetFilePath(const XlaSerializedCacheKey& key,
                        absl::string_view persistent_cache_directory) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("persistent_cache_directory: \"" + std::string(persistent_cache_directory.data(), persistent_cache_directory.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_25(mht_25_v, 1142, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "GetFilePath");

  const std::string file_name =
      absl::StrCat(XlaSerializedCacheKeyToString(key), ".pb");
  return io::JoinPath(persistent_cache_directory, file_name);
}

}  // namespace

Status XlaCompilationCache::SaveSerializedEntry(
    const XlaSerializedCacheEntry& entry) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_26(mht_26_v, 1154, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::SaveSerializedEntry");

  Env* env = Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(persistent_cache_directory_));
  const std::string file_path =
      GetFilePath(entry.key(), persistent_cache_directory_);
  return WriteBinaryProto(env, file_path, entry);
}

StatusOr<absl::optional<XlaSerializedCacheEntry>>
XlaCompilationCache::TryLoadSerializedEntry(const XlaSerializedCacheKey& key) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_compilation_cacheDTcc mht_27(mht_27_v, 1166, "", "./tensorflow/compiler/jit/xla_compilation_cache.cc", "XlaCompilationCache::TryLoadSerializedEntry");

  Env* env = Env::Default();
  const std::string file_path = GetFilePath(key, persistent_cache_directory_);
  if (!env->FileExists(file_path).ok()) {
    return StatusOr<absl::optional<XlaSerializedCacheEntry>>(absl::nullopt);
  }

  XlaSerializedCacheEntry entry;
  TF_RETURN_IF_ERROR(ReadTextOrBinaryProto(env, file_path, &entry));
  return StatusOr<absl::optional<XlaSerializedCacheEntry>>(entry);
}

}  // namespace tensorflow
