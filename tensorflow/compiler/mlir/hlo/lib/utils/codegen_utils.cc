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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScodegen_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScodegen_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScodegen_utilsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/utils/codegen_utils.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

using llvm::SmallVector;

namespace mlir {
namespace codegen_utils {

Value emitNumElementsComputation(OpBuilder& b, Location loc, Value memref) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScodegen_utilsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/mlir/hlo/lib/utils/codegen_utils.cc", "emitNumElementsComputation");

  int rank = memref.getType().cast<MemRefType>().getRank();
  Value num_elements;
  num_elements = b.create<mlir::arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  for (int r = 0; r < rank; ++r) {
    auto dim_size = b.create<memref::DimOp>(loc, memref, r);
    num_elements = b.create<arith::MulIOp>(loc, num_elements, dim_size);
  }
  return num_elements;
}

Value emitNumElementsComputation(OpBuilder& b, Location loc, Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSutilsPScodegen_utilsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/mlir/hlo/lib/utils/codegen_utils.cc", "emitNumElementsComputation");

  // only const rank is supported for now
  assert(op->getDialect()->getNamespace() == "lmhlo");
  int num_operands = op->getNumOperands();
  Value result_memref = op->getOperand(num_operands - 1);
  return emitNumElementsComputation(b, loc, result_memref);
}

SmallVector<Value> calcMultiDimIndex(OpBuilder& b, Location loc,
                                     Value linear_index,
                                     ArrayRef<Value> shape) {
  int rank = shape.size();
  SmallVector<Value> result;
  if (rank == 0) return result;
  if (rank == 1) {
    result.push_back(linear_index);
    return result;
  }

  // dim_acc_mul_vec = [d, c*d, b*c*d]
  SmallVector<Value> dim_acc_mul_vec;
  Value tmp_acc_mul = shape[rank - 1];
  dim_acc_mul_vec.emplace_back(tmp_acc_mul);
  for (int i = rank - 2; i > 0; --i) {
    tmp_acc_mul = b.create<arith::MulIOp>(loc, tmp_acc_mul, shape[i]);
    dim_acc_mul_vec.emplace_back(tmp_acc_mul);
  }
  Value block_index = linear_index;
  for (int i = 0; i < rank; ++i) {
    Value index;
    if (i == rank - 1) {
      index = block_index;
    } else {
      index =
          b.create<arith::DivUIOp>(loc, block_index, dim_acc_mul_vec.back());
      block_index =
          b.create<arith::RemUIOp>(loc, block_index, dim_acc_mul_vec.back());
      dim_acc_mul_vec.pop_back();
    }
    result.push_back(index);
  }
  return result;
}

SmallVector<Value> calcMultiDimIndex(OpBuilder& b, Location loc,
                                     Value linear_index, Value memref) {
  int rank = memref.getType().cast<MemRefType>().getRank();
  SmallVector<Value> result;
  if (rank == 0) return result;
  if (rank == 1) {
    result.push_back(linear_index);
    return result;
  }
  // shape = [a, b, c, d]
  SmallVector<Value, 4> shape_vec;
  for (int i = 0; i < rank; ++i) {
    shape_vec.push_back(b.create<memref::DimOp>(loc, memref, i));
  }

  return calcMultiDimIndex(b, loc, linear_index, shape_vec);
}

SmallVector<Value> calcMultiDimIndexForFirstOperand(OpBuilder& b, Location loc,
                                                    Value linear_index,
                                                    Operation* op) {
  assert(op->getDialect()->getNamespace() == "lmhlo");
  Value operand_memref = op->getOperand(0);
  return calcMultiDimIndex(b, loc, linear_index, operand_memref);
}

}  // namespace codegen_utils
}  // namespace mlir
