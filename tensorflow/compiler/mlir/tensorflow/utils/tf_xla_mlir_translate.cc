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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <type_traits>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace {

// NOLINTNEXTLINE
llvm::cl::opt<std::string> input_types(
    "tf-xla-input-types",
    llvm::cl::desc("XLA input argument types (kinds), separated by ','. "
                   "Supported types include ['parameter', 'resource']. If "
                   "empty, all arguments are assumed to be parameters."),
    llvm::cl::init(""));
// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_use_tuple_arg(
    "tf-xla-emit-use-tuple-args",
    llvm::cl::desc(
        "Emit HLO modules using tuples as args for the entry computation"),
    llvm::cl::init(false));
// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_return_tuple(
    "tf-xla-emit-return-tuple",
    llvm::cl::desc("Emit HLO modules with entry computations returning tuple"),
    llvm::cl::init(false));
}  // namespace

namespace tensorflow {

namespace {

mlir::LogicalResult PrintHloModuleText(
    const XlaCompilationResult& compilation_result, llvm::raw_ostream& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_0(mht_0_v, 252, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "PrintHloModuleText");

  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  if (!status_or_hlo_module.ok()) {
    LOG(ERROR) << "Conversion to HLO module failed: "
               << status_or_hlo_module.status().ToString();
    return mlir::failure();
  }

  xla::HloModule* hlo_module = status_or_hlo_module.ValueOrDie().get();

  output << hlo_module->ToString();

  if (!compilation_result.input_mapping.empty())
    output << "// InputMapping {"
           << absl::StrJoin(compilation_result.input_mapping, ", ") << "}\n";

  for (const auto& xla_input_shape : compilation_result.xla_input_shapes)
    output << "// XlaInputShape " << xla_input_shape.ToString() << '\n';

  output << "// XlaOutputShape "
         << compilation_result.xla_output_shape.ToString() << '\n';

  for (const auto& xla_output_description : compilation_result.outputs) {
    output << "// XlaOutputDescription type="
           << DataTypeString(xla_output_description.type) << " shape=("
           << absl::StrJoin(xla_output_description.shape.dim_sizes(), ", ")
           << ')';
    if (xla_output_description.input_index >= 0)
      output << " input_index=" << xla_output_description.input_index;
    if (xla_output_description.is_constant) output << " constant";
    if (xla_output_description.is_tensor_list) output << " tensor_list";
    output << '\n';
  }

  for (const auto& resource_update : compilation_result.resource_updates) {
    output << "// ResourceUpdate input_index=" << resource_update.input_index
           << " type=" << DataTypeString(resource_update.type) << " shape=("
           << absl::StrJoin(resource_update.shape.dim_sizes(), " ") << ')';
    if (resource_update.modified) output << " modified";
    output << '\n';
  }

  return mlir::success();
}

Status ParseArgumentShapes(
    absl::string_view input_shapes_str,
    llvm::SmallVectorImpl<TensorOrResourceShape>& arg_shapes) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input_shapes_str: \"" + std::string(input_shapes_str.data(), input_shapes_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_1(mht_1_v, 306, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "ParseArgumentShapes");

  arg_shapes.clear();
  std::vector<llvm::Optional<std::vector<int>>> input_shapes_vector;
  TF_RETURN_IF_ERROR(ParseNodeShapes(input_shapes_str, input_shapes_vector));
  arg_shapes.resize(input_shapes_vector.size());
  for (const auto& shape : llvm::enumerate(input_shapes_vector)) {
    if (!shape.value().hasValue()) {
      TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
          static_cast<int*>(nullptr), 0, &arg_shapes[shape.index()].shape));
      continue;
    }
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
        shape.value().getValue(), &arg_shapes[shape.index()].shape));
  }

  return Status::OK();
}

Status ParseDataTypes(absl::string_view data_types_str,
                      llvm::SmallVectorImpl<DataType>& data_types) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("data_types_str: \"" + std::string(data_types_str.data(), data_types_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_2(mht_2_v, 329, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "ParseDataTypes");

  data_types.clear();
  std::vector<std::string> input_dtypes_vector;
  TF_RETURN_IF_ERROR(ParseNodeDataTypes(data_types_str, input_dtypes_vector));
  data_types.resize(input_dtypes_vector.size(), DT_INVALID);
  for (auto data_type : llvm::enumerate(input_dtypes_vector)) {
    if (!DataType_Parse(data_type.value(), &data_types[data_type.index()]))
      return errors::InvalidArgument("Invalid dtype at index ",
                                     data_type.index(), ": ",
                                     data_type.value());
    const auto& resolved_dtype = data_types[data_type.index()];
    if (resolved_dtype == DT_INVALID || resolved_dtype == DT_STRING ||
        resolved_dtype == DT_RESOURCE || resolved_dtype == DT_VARIANT ||
        IsRefType(resolved_dtype))
      return errors::InvalidArgument("Unsupported dtype at index ",
                                     data_type.index(), ": ",
                                     data_type.value());
  }

  return Status::OK();
}

Status ParseArgumentKinds(
    absl::string_view input_types_str,
    llvm::SmallVectorImpl<XlaArgument::Kind>& argument_kinds) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("input_types_str: \"" + std::string(input_types_str.data(), input_types_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_3(mht_3_v, 357, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "ParseArgumentKinds");

  argument_kinds.clear();
  if (input_types_str.empty()) return Status::OK();

  std::vector<absl::string_view> argument_kind_strs =
      absl::StrSplit(input_types_str, ',');
  argument_kinds.reserve(argument_kind_strs.size());
  for (const auto& argument_kind_str : llvm::enumerate(argument_kind_strs)) {
    const auto& value = argument_kind_str.value();
    if (value == "parameter") {
      argument_kinds.push_back(XlaArgument::Kind::kParameter);
    } else if (value == "resource") {
      argument_kinds.push_back(XlaArgument::Kind::kResource);
    } else {
      return errors::InvalidArgument(
          "Unsupported TF/XLA argument kind at index ",
          argument_kind_str.index(), ": ", value);
    }
  }

  return Status::OK();
}

Status ParseXlaArguments(absl::string_view input_shapes_str,
                         absl::string_view input_dtypes_str,
                         absl::string_view arg_kinds_str,
                         llvm::SmallVectorImpl<XlaArgument>& xla_arguments) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("input_shapes_str: \"" + std::string(input_shapes_str.data(), input_shapes_str.size()) + "\"");
   mht_4_v.push_back("input_dtypes_str: \"" + std::string(input_dtypes_str.data(), input_dtypes_str.size()) + "\"");
   mht_4_v.push_back("arg_kinds_str: \"" + std::string(arg_kinds_str.data(), arg_kinds_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_4(mht_4_v, 389, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "ParseXlaArguments");

  xla_arguments.clear();
  std::vector<llvm::Optional<std::vector<int>>> input_shapes_vector;
  TF_RETURN_IF_ERROR(
      tensorflow::ParseNodeShapes(input_shapes_str, input_shapes_vector));
  llvm::SmallVector<DataType, 4> dtypes_vector;
  TF_RETURN_IF_ERROR(ParseDataTypes(input_dtypes_str, dtypes_vector));
  llvm::SmallVector<XlaArgument::Kind, 4> arg_kinds_vector;
  TF_RETURN_IF_ERROR(ParseArgumentKinds(arg_kinds_str, arg_kinds_vector));

  if (input_shapes_vector.empty())
    input_shapes_vector.resize(dtypes_vector.size());

  if (arg_kinds_vector.empty())
    arg_kinds_vector.resize(input_shapes_vector.size(),
                            XlaArgument::Kind::kParameter);

  if (input_shapes_vector.size() != dtypes_vector.size() ||
      input_shapes_vector.size() != arg_kinds_vector.size())
    return errors::InvalidArgument(
        "Input shapes, dtypes, and types/kinds must be of the same "
        "length, but got ",
        input_shapes_vector.size(), ", ", dtypes_vector.size(), ", and ",
        arg_kinds_vector.size(), " respectively");

  xla_arguments.resize(input_shapes_vector.size());
  for (const auto& arg_components :
       llvm::zip(xla_arguments, input_shapes_vector, dtypes_vector,
                 arg_kinds_vector)) {
    XlaArgument& arg = std::get<0>(arg_components);
    TensorShape shape;
    auto input_shapes = std::get<1>(arg_components);
    if (input_shapes.hasValue()) {
      TF_RETURN_IF_ERROR(
          TensorShapeUtils::MakeShape(input_shapes.getValue(), &shape));
    } else {
      TF_RETURN_IF_ERROR(
          TensorShapeUtils::MakeShape(static_cast<int*>(nullptr), 0, &shape));
    }
    arg.shape = std::move(shape);
    arg.type = std::get<2>(arg_components);
    arg.kind = std::get<3>(arg_components);
  }

  return Status::OK();
}

}  // anonymous namespace

// Test BuildHloFromTf. BuildHloFromTf only performs part of the conversion, so
// to make this test comparable to other compile tests, the test implements
// the remaining parts of the conversion.
Status CompileMlirToXlaHloViaBuilder(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    llvm::StringRef device_type, XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_5(mht_5_v, 448, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "CompileMlirToXlaHloViaBuilder");

  // This call to RefineShapes is redundant with the call in BuildHloFromTf.
  // It's here so xla::Parameters that are created form block.getArguments will
  // have the proper shapes.
  TF_RETURN_IF_ERROR(RefineShapes(arg_shapes, module_op));

  mlir::func::FuncOp main = module_op.lookupSymbol<mlir::func::FuncOp>("main");
  mlir::Block& block = main.getRegion().front();
  xla::XlaBuilder builder("main");

  // Create xla_params.
  std::vector<xla::XlaOp> xla_params;
  for (mlir::BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    xla::Shape shape = xla::TypeToShape(arg.getType());
    xla::XlaOp argop =
        xla::Parameter(&builder, num, shape, absl::StrCat("Arg_", num));
    xla_params.push_back(argop);
  }

  std::vector<xla::XlaOp> returns(1);
  TF_RETURN_IF_ERROR(BuildHloFromTf(module_op, builder, xla_params, returns,
                                    arg_shapes, device_type,
                                    custom_legalization_passes));

  xla::XlaOp return_value;
  if (returns.size() == 1)
    return_value = returns[0];
  else
    return_value = xla::Tuple(&builder, returns);

  TF_ASSIGN_OR_RETURN(
      xla::XlaComputation computation,
      return_value.valid() ? builder.Build(return_value) : builder.Build());
  auto hlo_module = computation.proto();
  xla::HloProto hlo_proto;
  hlo_proto.mutable_hlo_module()->Swap(&hlo_module);

  compilation_result->computation = std::make_shared<xla::XlaComputation>();
  xla::XlaComputation* xla_computation = compilation_result->computation.get();
  *xla_computation = xla::XlaComputation(hlo_proto.hlo_module());

  XlaHelpers::ShapeRepresentationFn shape_representation_fn =
      IdentityShapeRepresentationFn();
  return PopulateResultIOInfo(module_op, arg_shapes, /*use_tuple_args=*/false,
                              /*use_resource_updates_for_aliases=*/false,
                              shape_representation_fn, compilation_result);
}

static mlir::LogicalResult MlirTfToHloTextTranslateFunctionImpl(
    mlir::ModuleOp module_op, llvm::raw_ostream& output, bool via_builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_6(mht_6_v, 501, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "MlirTfToHloTextTranslateFunctionImpl");

  if (!module_op) return mlir::failure();

  llvm::SmallVector<TensorOrResourceShape, 4> arg_shapes;
  auto args_status =
      ParseArgumentShapes(mlir::StringRefToView(input_shapes), arg_shapes);
  if (!args_status.ok()) {
    LOG(ERROR) << args_status.ToString();
    return mlir::failure();
  }

  auto device_type = "XLA_CPU_JIT";
  llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
      custom_legalization_passes{};
  XlaCompilationResult compilation_result;
  auto compilation_status =
      via_builder ? CompileMlirToXlaHloViaBuilder(
                        module_op, arg_shapes, device_type, &compilation_result,
                        custom_legalization_passes)
                  : CompileMlirToXlaHlo(
                        module_op, arg_shapes, device_type, emit_use_tuple_arg,
                        /*analyse_graph=*/false, emit_return_tuple,
                        /*use_resource_updates_for_aliases=*/true,
                        IdentityShapeRepresentationFn(), &compilation_result,
                        custom_legalization_passes);
  if (!compilation_status.ok()) {
    LOG(ERROR) << "TF/XLA compilation failed: "
               << compilation_status.ToString();
    return mlir::failure();
  }

  return PrintHloModuleText(compilation_result, output);
}

static mlir::LogicalResult MlirTfGraphToHloTextTranslateFunction(
    mlir::ModuleOp module_op, llvm::raw_ostream& output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_7(mht_7_v, 539, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "MlirTfGraphToHloTextTranslateFunction");

  if (!module_op) return mlir::failure();

  llvm::SmallVector<XlaArgument, 4> xla_arguments;
  auto args_status = ParseXlaArguments(
      mlir::StringRefToView(input_shapes), mlir::StringRefToView(input_dtypes),
      mlir::StringRefToView(input_types), xla_arguments);
  if (!args_status.ok()) {
    LOG(ERROR) << args_status.ToString();
    return mlir::failure();
  }

  XlaCompilationResult compilation_result;
  auto compilation_status =
      CompileGraphToXlaHlo(module_op, xla_arguments,
                           /*device_type=*/"XLA_CPU_JIT", emit_use_tuple_arg,
                           /*analyse_graph=*/false, emit_return_tuple,
                           IdentityShapeRepresentationFn(), &compilation_result,
                           /*custom_legalization_passes=*/{});
  if (!compilation_status.ok()) {
    LOG(ERROR) << "TF/XLA compilation failed: "
               << compilation_status.ToString();
    return mlir::failure();
  }

  return PrintHloModuleText(compilation_result, output);
}

static void RegisterMlirInputDialects(mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_8(mht_8_v, 570, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "RegisterMlirInputDialects");

  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::TF::TensorFlowDialect>();
}

static void RegisterGraphInputDialects(mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_9(mht_9_v, 578, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "RegisterGraphInputDialects");

  RegisterMlirInputDialects(registry);
  registry.insert<mlir::tf_executor::TensorFlowExecutorDialect>();
}

static mlir::OwningOpRef<mlir::ModuleOp>
SerializedMlirStringAttrToMlirModuleTranslate(llvm::StringRef input,
                                              mlir::MLIRContext* context) {
  mlir::Attribute attr = mlir::parseAttribute(input, context);
  if (!attr || !attr.isa<mlir::StringAttr>()) {
    LOG(ERROR) << "Input is not parsable as a MLIR StringAttr.";
    return nullptr;
  }
  auto str_attr = attr.cast<mlir::StringAttr>();

  mlir::DialectRegistry registry;
  RegisterMlirInputDialects(registry);
  context->appendDialectRegistry(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module_ref;
  auto status =
      DeserializeMlirModule(str_attr.getValue().str(), context, &module_ref);
  if (!status.ok()) {
    LOG(ERROR) << status.ToString();
    return nullptr;
  }

  return module_ref;
}

static mlir::LogicalResult MlirModuleToSerializedMlirStringAttrTranslate(
    mlir::ModuleOp module_op, llvm::raw_ostream& output) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_10(mht_10_v, 611, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "MlirModuleToSerializedMlirStringAttrTranslate");

  output << "\"";
  std::string serialized_module = SerializeMlirModule(module_op);
  llvm::printEscapedString(serialized_module, output);
  output << "\"";
  return mlir::success();
}

static mlir::LogicalResult MlirTfToHloTextTranslateFunction(
    mlir::ModuleOp module_op, llvm::raw_ostream& output) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_11(mht_11_v, 623, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "MlirTfToHloTextTranslateFunction");

  return MlirTfToHloTextTranslateFunctionImpl(module_op, output, false);
}

static mlir::LogicalResult MlirTfToHloTextViaBuilderTranslateFunction(
    mlir::ModuleOp module_op, llvm::raw_ostream& output) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStf_xla_mlir_translateDTcc mht_12(mht_12_v, 631, "", "./tensorflow/compiler/mlir/tensorflow/utils/tf_xla_mlir_translate.cc", "MlirTfToHloTextViaBuilderTranslateFunction");

  return MlirTfToHloTextTranslateFunctionImpl(module_op, output, true);
}

}  // namespace tensorflow

static mlir::TranslateFromMLIRRegistration MlirTfToHloTextTranslate(
    "mlir-tf-to-hlo-text", tensorflow::MlirTfToHloTextTranslateFunction,
    tensorflow::RegisterMlirInputDialects);

static mlir::TranslateFromMLIRRegistration MlirTfToHloTextViaBuilderTranslate(
    "mlir-tf-to-hlo-text-via-builder",
    tensorflow::MlirTfToHloTextViaBuilderTranslateFunction,
    tensorflow::RegisterMlirInputDialects);

static mlir::TranslateFromMLIRRegistration MlirTfGraphToHloTextTranslate(
    "mlir-tf-graph-to-hlo-text",
    tensorflow::MlirTfGraphToHloTextTranslateFunction,
    tensorflow::RegisterGraphInputDialects);

static mlir::TranslateToMLIRRegistration SerializedMlirStringAttrToMlirModule(
    "mlir-tf-str-attr-to-mlir",
    tensorflow::SerializedMlirStringAttrToMlirModuleTranslate);

static mlir::TranslateFromMLIRRegistration MlirModuleToSerializedMlirStringAttr(
    "mlir-tf-mlir-to-str-attr",
    tensorflow::MlirModuleToSerializedMlirStringAttrTranslate,
    tensorflow::RegisterMlirInputDialects);
