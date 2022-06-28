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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc() {
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

#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

#include <cstdint>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {
namespace {

RankedTensorType GetRankedTensorType(mlir::Value val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_0(mht_0_v, 205, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "GetRankedTensorType");

  mlir::Type type = val.getType();
  if (auto type_with_subtype =
          mlir::getElementTypeOrSelf(val)
              .dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>()) {
    if (type_with_subtype.GetSubtypes().size() == 1) {
      type = type_with_subtype.GetSubtypes().front();
    }
  }
  return type.dyn_cast_or_null<RankedTensorType>();
}
}  // namespace

mlir::LogicalResult DTensorLayout::verify() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_1(mht_1_v, 221, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "DTensorLayout::verify");

  DTensorLayout op = *this;
  const auto& layout = op.layout();
  if (layout.IsEmpty()) return mlir::success();

  auto input_value = op.input();

  RankedTensorType type = GetRankedTensorType(input_value);

  if (!type) return mlir::success();

  const auto& num_shards = layout.num_shards();
  if (num_shards.size() != type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "requires matching rank for layout and input, but got {0} as suggested "
        "rank from layout but {1} from shape.",
        num_shards.size(), type.getRank()));
  }

  for (const auto& dim_and_index :
       llvm::enumerate(llvm::zip(type.getShape(), num_shards))) {
    const int dimension_index = dim_and_index.index();
    const auto& dim_and_shards = dim_and_index.value();
    const int dim = std::get<0>(dim_and_shards);
    const int num_shard_for_dim = std::get<1>(dim_and_shards);
    if (dim <= 0) continue;

    if (dim % num_shard_for_dim != 0)
      return op.emitOpError(llvm::formatv(
          "requires dimension {0} to be divisible by sharding "
          "specified in DTensorLayout, but got dimension size={1} is not "
          "divisible by number of shards in layout for this dimension={2}.",
          dimension_index, dim, num_shard_for_dim));
  }

  return mlir::success();
}

mlir::LogicalResult DTensorAllGatherOp::verify() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_2(mht_2_v, 262, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "DTensorAllGatherOp::verify");

  DTensorAllGatherOp op = *this;
  const tensorflow::dtensor::Layout input_layout = op.input_layout();
  const tensorflow::dtensor::Layout output_layout = op.output_layout();

  if (input_layout.rank() != output_layout.rank())
    return op.emitOpError()
           << "received input and output layouts of unequal ranks "
           << input_layout.rank() << " and " << output_layout.rank();

  for (int32_t i = 0; i < input_layout.rank(); ++i) {
    if (input_layout.sharding_spec(i) != output_layout.sharding_spec(i) &&
        tensorflow::dtensor::Layout::IsShardedDimension(
            output_layout.sharding_spec(i))) {
      return op.emitOpError()
             << "dimension " << i << " of output layout has sharding spec "
             << output_layout.sharding_spec(i)
             << " which is more sharded then the input layout spec "
             << input_layout.sharding_spec(i);
    }
  }

  RankedTensorType input_type =
      op.input().getType().dyn_cast<RankedTensorType>();
  if (!input_type) return mlir::success();

  if (input_type.getRank() != input_layout.rank())
    return op.emitOpError()
           << "input layout rank " << input_layout.rank()
           << " is not equal to input rank " << input_type.getRank();

  RankedTensorType output_type =
      op.output().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return mlir::success();

  if (output_type.getRank() != output_layout.rank())
    return op.emitOpError()
           << "output layout rank " << output_layout.rank()
           << " is not equal to output rank " << output_type.getRank();

  std::vector<int64_t> computed_output_shape =
      output_layout.LocalShapeFromGlobalShape(
          input_layout.GlobalShapeFromLocalShape(input_type.getShape()));

  for (int32_t i = 0; i < computed_output_shape.size(); ++i) {
    if (computed_output_shape[i] != output_type.getShape()[i]) {
      return op.emitOpError()
             << "computed output shape " << computed_output_shape[i]
             << " at dimension " << i << " is not equal to actual output shape "
             << output_type.getShape()[i];
    }
  }

  return mlir::success();
}

mlir::LogicalResult DTensorAllScatterOp::verify() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_3(mht_3_v, 321, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "DTensorAllScatterOp::verify");

  DTensorAllScatterOp op = *this;
  const tensorflow::dtensor::Layout input_layout = op.input_layout();
  const tensorflow::dtensor::Layout output_layout = op.output_layout();

  if (input_layout.rank() != output_layout.rank())
    return op.emitOpError()
           << "received input and output layouts of unequal ranks "
           << input_layout.rank() << " and " << output_layout.rank();

  for (int32_t i = 0; i < input_layout.rank(); ++i) {
    if (input_layout.sharding_spec(i) != output_layout.sharding_spec(i) &&
        tensorflow::dtensor::Layout::IsShardedDimension(
            input_layout.sharding_spec(i))) {
      return op.emitOpError()
             << "dimension " << i << " of input layout has sharding spec "
             << input_layout.sharding_spec(i)
             << " which is more sharded then the output layout spec "
             << output_layout.sharding_spec(i);
    }
  }

  RankedTensorType input_type =
      op.input().getType().dyn_cast<RankedTensorType>();
  if (!input_type) return mlir::success();

  if (input_type.getRank() != input_layout.rank())
    return op.emitOpError()
           << "input layout rank " << input_layout.rank()
           << " is not equal to input rank " << input_type.getRank();

  RankedTensorType output_type =
      op.output().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return mlir::success();

  if (output_type.getRank() != output_layout.rank())
    return op.emitOpError()
           << "output layout rank " << output_layout.rank()
           << " is not equal to output rank " << output_type.getRank();

  std::vector<int64_t> computed_output_shape =
      output_layout.LocalShapeFromGlobalShape(
          input_layout.GlobalShapeFromLocalShape(input_type.getShape()));

  for (int32_t i = 0; i < computed_output_shape.size(); ++i) {
    if (computed_output_shape[i] != output_type.getShape()[i]) {
      return op.emitOpError()
             << "computed output shape " << computed_output_shape[i]
             << " at dimension " << i << " is not equal to actual output shape "
             << output_type.getShape()[i];
    }
  }

  return mlir::success();
}

LogicalResult DTensorLayout::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_4(mht_4_v, 383, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "DTensorLayout::inferReturnTypes");

  assert(operands.size() == 1);
  inferredReturnTypes.assign({operands[0].getType()});
  return success();
}

void DTensorOpAdderHook(TensorFlowDialect& dialect) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_5(mht_5_v, 392, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "DTensorOpAdderHook");

  dialect.addOperations<
#define GET_OP_LIST
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.cc.inc"
      >();
}

int RegisterOnce() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_6(mht_6_v, 402, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "RegisterOnce");

  TF_DIALECT_REGISTER_ADDITIONAL_OPERATIONS(DTensorOpAdderHook)
  return 0;
}

int RegisterDTensorTFOps() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSirPStf_dtensorDTcc mht_7(mht_7_v, 410, "", "./tensorflow/dtensor/mlir/ir/tf_dtensor.cc", "RegisterDTensorTFOps");

  static int r = RegisterOnce();
  return r;
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.cc.inc"
