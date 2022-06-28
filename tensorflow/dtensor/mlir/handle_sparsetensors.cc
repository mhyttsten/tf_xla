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
class MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kEntryFuncAttr[] = "tf.entry_function";
constexpr char kSparseIndicesStr[] = "op_input_sparse_indices";
constexpr char kSparseDenseShapesStr[] = "op_input_sparse_dense_shapes";
constexpr char kSparseValuesStr[] = "op_input_sparse_values";

typedef struct SparseTensorToComponentInfo {
  mlir::RankedTensorType indices;
  mlir::RankedTensorType values;
  mlir::RankedTensorType dense_shapes;
  unsigned int func_op_arg_index;
} SparseTensorToComponentInfo;

void UpdateFunctionSignature(mlir::func::FuncOp function,
                             mlir::OpBuilder& builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc mht_0(mht_0_v, 234, "", "./tensorflow/dtensor/mlir/handle_sparsetensors.cc", "UpdateFunctionSignature");

  function.setType(mlir::FunctionType::get(
      builder.getContext(),
      llvm::to_vector<4>(function.front().getArgumentTypes()),
      function.getFunctionType().getResults()));
}

// Add input attributes for new sparsetensor components and remove the
// old sparsetensor value input attributes.
//
// TF has a list of comma separated input names within `kEntryFuncAttr`
// attribute, under 'inputs'. Update this comma separated list of input names
// by correctly deleting the sparse tensor input name and replacing it with
// three new sparse component input names.
//
// Without this update, MLIR conversion to GraphDef will fail since
// the number of input names will not match with the FuncOp num arguments.
//
// e.g. "op_input_1" should become
// "op_input_sparse_indices_0,op_input_sparse_dense_shapes_0,
// "op_input_sparse_values_0"
mlir::LogicalResult UpdateFunctionInputAttributes(
    mlir::MLIRContext& context, mlir::func::FuncOp main_func,
    mlir::OpBuilder& builder,
    const std::vector<SparseTensorToComponentInfo>& sparse_tensor_components) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc mht_1(mht_1_v, 261, "", "./tensorflow/dtensor/mlir/handle_sparsetensors.cc", "UpdateFunctionInputAttributes");

  llvm::SmallVector<llvm::StringRef, 2> input_names;

  auto dict_attr =
      main_func->getAttrOfType<mlir::DictionaryAttr>(kEntryFuncAttr);
  if (dict_attr) {
    if (!dict_attr.get("inputs").isa<mlir::StringAttr>())
      return main_func.emitOpError("Missing attribute inputs in main FuncOp.");

    dict_attr.get("inputs").cast<mlir::StringAttr>().getValue().split(
        input_names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

    llvm::SmallVector<std::string, 2> new_input_names;

    absl::flat_hash_set<int> skip_indices;
    for (const auto component : sparse_tensor_components) {
      skip_indices.insert(component.func_op_arg_index);
    }

    for (auto i = 0; i < input_names.size(); ++i) {
      if (skip_indices.find(i) == skip_indices.end()) {
        new_input_names.push_back(input_names[i].str());
      }
    }

    for (const auto component : sparse_tensor_components) {
      int arg_index = component.func_op_arg_index;
      new_input_names.push_back(
          absl::StrCat(kSparseIndicesStr, "_", arg_index));
      new_input_names.push_back(
          absl::StrCat(kSparseDenseShapesStr, "_", arg_index));
      new_input_names.push_back(absl::StrCat(kSparseValuesStr, "_", arg_index));
    }

    mlir::NamedAttrList attributes(dict_attr);
    attributes.set(
        "inputs",
        mlir::StringAttr::get(&context, absl::StrJoin(new_input_names, ",")));
    main_func->setAttr(kEntryFuncAttr, attributes.getDictionary(&context));
  }
  UpdateFunctionSignature(main_func, builder);
  return mlir::success();
}

// For each SparseTensor block argument of the main FuncOp, create
// three of the component tensors, `indices`, `values`, and `dense_shapes`
// and add it to `sparse_tensor_components`.
void CreateComponentTensorsFromSparseTensors(
    mlir::func::FuncOp main_func, mlir::OpBuilder& builder,
    std::vector<SparseTensorToComponentInfo>* sparse_tensor_components) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc mht_2(mht_2_v, 313, "", "./tensorflow/dtensor/mlir/handle_sparsetensors.cc", "CreateComponentTensorsFromSparseTensors");

  for (const auto block_arg : main_func.getArguments()) {
    const auto is_sparse = main_func.getArgAttrOfType<mlir::BoolAttr>(
        block_arg.getArgNumber(), kSparseValue);
    if (is_sparse) {
      sparse_tensor_components->push_back(SparseTensorToComponentInfo{
          /*indices=*/mlir::RankedTensorType::get({-1, ValueRank(block_arg)},
                                                  builder.getI64Type()),
          /*values=*/
          mlir::RankedTensorType::get({-1},
                                      block_arg.getType()
                                          .dyn_cast<mlir::RankedTensorType>()
                                          .getElementType()),
          /*dense_shapes=*/
          mlir::RankedTensorType::get({ValueRank(block_arg)},
                                      builder.getI64Type()),
          /*func_op_arg_index=*/block_arg.getArgNumber()});
    }
  }
}

// Inserts SparseTensor components `components` into `main_func` at the end
// of block arguments list.
void UpdateFunctionWithSparseTensorComponents(
    mlir::MLIRContext& context, mlir::func::FuncOp main_func,
    mlir::OpBuilder& builder, const SparseTensorToComponentInfo& component) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc mht_3(mht_3_v, 341, "", "./tensorflow/dtensor/mlir/handle_sparsetensors.cc", "UpdateFunctionWithSparseTensorComponents");

  main_func.front().addArgument(component.indices, main_func.getLoc());
  main_func.front().addArgument(component.dense_shapes, main_func.getLoc());
  main_func.front().addArgument(component.values, main_func.getLoc());
  UpdateFunctionSignature(main_func, builder);
}

struct DTensorSparseTensorToDenseTensor
    : public DTensorSparseTensorToDenseTensorBase<
          DTensorSparseTensorToDenseTensor> {
  void runOnOperation() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_sparsetensorsDTcc mht_4(mht_4_v, 354, "", "./tensorflow/dtensor/mlir/handle_sparsetensors.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    auto module = getOperation();
    mlir::OpBuilder builder(&context);

    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");

    // Save Arg Attributes for each argument for later use, this will be
    // reset and reordered after we insert sparse tensor components arguments.
    llvm::DenseMap<mlir::Value, llvm::ArrayRef<mlir::NamedAttribute>>
        arg_attribute_map;
    for (auto block_arg : main_func.getArguments()) {
      arg_attribute_map.insert(std::make_pair(
          block_arg, main_func.getArgAttrs(block_arg.getArgNumber())));
    }

    std::vector<SparseTensorToComponentInfo> sparse_tensor_components;
    CreateComponentTensorsFromSparseTensors(main_func, builder,
                                            &sparse_tensor_components);

    // Update func arguments in place by replacing SparseTensors with their
    // components and emitting a SparseToDenseOp before all ops that consume
    // a SparseTensor.
    for (const SparseTensorToComponentInfo& components :
         sparse_tensor_components) {
      // Insert SparseTensor component into the main function's block
      // arguments.
      mlir::Value sparse_tensor_value =
          main_func.getArgument(components.func_op_arg_index);

      UpdateFunctionWithSparseTensorComponents(context, main_func, builder,
                                               components);
      mlir::Operation* front_op = &main_func.front().front();
      builder.setInsertionPoint(front_op);

      // Emit a SparseToDenseOp and replace the SparseTensor with the result of
      // this new op.
      auto zero_scalar = CreateZeroScalarConst(builder, front_op->getLoc(),
                                               sparse_tensor_value.getType()
                                                   .cast<mlir::TensorType>()
                                                   .getElementType());
      if (!zero_scalar.has_value()) return signalPassFailure();
      mlir::TF::SparseToDenseOp sparse_to_dense_op =
          builder.create<mlir::TF::SparseToDenseOp>(
              front_op->getLoc(), sparse_tensor_value.getType(),
              mlir::ValueRange(
                  {main_func.getArgument(main_func.getNumArguments() - 3),
                   main_func.getArgument(main_func.getNumArguments() - 2),
                   main_func.getArgument(main_func.getNumArguments() - 1),
                   zero_scalar.value()}));

      sparse_tensor_value.replaceAllUsesWith(sparse_to_dense_op);
      if (!sparse_tensor_value.use_empty()) return signalPassFailure();
    }

    // Erase sparse tensor arguments now that we converted all of them.
    for (int i = 0; i < sparse_tensor_components.size(); ++i)
      main_func.front().eraseArgument(
          sparse_tensor_components[i].func_op_arg_index - i);

    // Reset block argument attributes since they are likely mixed up
    // due to change in ordering of arguments.
    for (auto block_arg : main_func.getArguments()) {
      if (arg_attribute_map.find(block_arg) == arg_attribute_map.end()) {
        main_func.setArgAttrs(block_arg.getArgNumber(),
                              llvm::ArrayRef<mlir::NamedAttribute>{});
      } else {
        main_func.setArgAttrs(block_arg.getArgNumber(),
                              arg_attribute_map[block_arg]);
      }
    }
    if (mlir::failed(UpdateFunctionInputAttributes(context, main_func, builder,
                                                   sparse_tensor_components)))
      return signalPassFailure();
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSparseTensorToDenseTensor() {
  return std::make_unique<DTensorSparseTensorToDenseTensor>();
}

}  // namespace dtensor
}  // namespace tensorflow
