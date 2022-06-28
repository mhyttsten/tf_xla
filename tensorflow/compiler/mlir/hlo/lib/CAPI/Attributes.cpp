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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp() {
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

#include "mlir-hlo-c/Attributes.h"

#include <string>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir/CAPI/IR.h"

//
// ScatterDimensionNumbersAttr.
//

MlirAttribute mlirMhloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::mhlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::makeArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::makeArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::makeArrayRef(scatteredDimsToOperandDims,
                         nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool mlirMhloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_0(mht_0_v, 206, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAScatterDimensionNumbers");

  return unwrap(attr).isa<mlir::mhlo::ScatterDimensionNumbersAttr>();
}

intptr_t mlirMhloScatterDimensionNumbersGetUpdateWindowDimsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_1(mht_1_v, 214, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloScatterDimensionNumbersGetUpdateWindowDimsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetUpdateWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloScatterDimensionNumbersGetUpdateWindowDimsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()[pos];
}

intptr_t mlirMhloScatterDimensionNumbersGetInsertedWindowDimsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_3(mht_3_v, 235, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloScatterDimensionNumbersGetInsertedWindowDimsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetInsertedWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_4(mht_4_v, 246, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloScatterDimensionNumbersGetInsertedWindowDimsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()[pos];
}

intptr_t mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_5(mht_5_v, 256, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()
      .size();
}

int64_t mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_6(mht_6_v, 267, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()[pos];
}

int64_t mlirMhloDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_7(mht_7_v, 276, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDimensionNumbersGetIndexVectorDim");

  return unwrap(attr)
      .cast<mlir::mhlo::ScatterDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//
// GatherDimensionNumbersAttr.
//

MlirAttribute mlirMhloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_8(mht_8_v, 293, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGet");

  return wrap(mlir::mhlo::GatherDimensionNumbersAttr::get(
      unwrap(ctx), llvm::makeArrayRef(offsetDims, nOffsetDims),
      llvm::makeArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::makeArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool mlirMhloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_9(mht_9_v, 303, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAGatherDimensionNumbers");

  return unwrap(attr).isa<mlir::mhlo::GatherDimensionNumbersAttr>();
}

intptr_t mlirMhloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_10(mht_10_v, 310, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetOffsetDimsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                        intptr_t pos) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_11(mht_11_v, 321, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetOffsetDimsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()[pos];
}

intptr_t mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_12(mht_12_v, 331, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_13(mht_13_v, 342, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetCollapsedSliceDimsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()[pos];
}

intptr_t mlirMhloGatherDimensionNumbersGetStartIndexMapSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_14(mht_14_v, 352, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetStartIndexMapSize");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()
      .size();
}

int64_t mlirMhloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                           intptr_t pos) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_15(mht_15_v, 363, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetStartIndexMapElem");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()[pos];
}

int64_t mlirMhloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_16(mht_16_v, 372, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloGatherDimensionNumbersGetIndexVectorDim");

  return unwrap(attr)
      .cast<mlir::mhlo::GatherDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//
// DotDimensionNumbersAttr.
//

MlirAttribute mlirMhloDotDimensionNumbersGet(
    MlirContext ctx, intptr_t nLhsBatchingDimensions,
    const int64_t *lhsBatchingDimensions, intptr_t nRhsBatchingDimensions,
    const int64_t *rhsBatchingDimensions, intptr_t nLhsContractingDimensions,
    const int64_t *lhsContractingDimensions, intptr_t nRhsContractingDimensions,
    const int64_t *rhsContractingDimensions) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_17(mht_17_v, 390, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGet");

  return wrap(mlir::mhlo::DotDimensionNumbersAttr::get(
      unwrap(ctx),
      llvm::makeArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::makeArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::makeArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::makeArrayRef(rhsContractingDimensions, nRhsContractingDimensions)));
}

bool mlirMhloAttributeIsADotDimensionNumbers(MlirAttribute attr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_18(mht_18_v, 402, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsADotDimensionNumbers");

  return unwrap(attr).isa<mlir::mhlo::DotDimensionNumbersAttr>();
}

intptr_t mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_19(mht_19_v, 410, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_20(mht_20_v, 421, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetLhsBatchingDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_21(mht_21_v, 431, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_22(mht_22_v, 442, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetRhsBatchingDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_23(mht_23_v, 452, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetLhsContractingDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_24(mht_24_v, 463, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetLhsContractingDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()[pos];
}

intptr_t mlirMhloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_25(mht_25_v, 473, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetRhsContractingDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()
      .size();
}

int64_t mlirMhloDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_26(mht_26_v, 484, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDotDimensionNumbersGetRhsContractingDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()[pos];
}

//
// ConvDimensionNumbersAttr.
//

MlirAttribute mlirMhloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_27(mht_27_v, 503, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGet");

  return wrap(mlir::mhlo::ConvDimensionNumbersAttr::get(
      unwrap(ctx), inputBatchDimension, inputFeatureDimension,
      llvm::makeArrayRef(inputSpatialDimensions, nInputSpatialDimensions),
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      llvm::makeArrayRef(kernelSpatialDimensions, nKernelSpatialDimensions),
      outputBatchDimension, outputFeatureDimension,
      llvm::makeArrayRef(outputSpatialDimensions, nOutputSpatialDimensions)));
}

bool mlirMhloAttributeIsAConvDimensionNumbers(MlirAttribute attr) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_28(mht_28_v, 516, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAConvDimensionNumbers");

  return unwrap(attr).isa<mlir::mhlo::ConvDimensionNumbersAttr>();
}

int64_t mlirMhloConvDimensionNumbersGetInputBatchDimension(MlirAttribute attr) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_29(mht_29_v, 523, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetInputBatchDimension");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputBatchDimension();
}

int64_t mlirMhloConvDimensionNumbersGetInputFeatureDimension(
    MlirAttribute attr) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_30(mht_30_v, 533, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetInputFeatureDimension");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetInputSpatialDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_31(mht_31_v, 543, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetInputSpatialDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetInputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_32(mht_32_v, 554, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetInputSpatialDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()[pos];
}

int64_t mlirMhloConvDimensionNumbersGetKernelInputFeatureDimension(
    MlirAttribute attr) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_33(mht_33_v, 564, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetKernelInputFeatureDimension");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelInputFeatureDimension();
}

int64_t mlirMhloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_34(mht_34_v, 574, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetKernelOutputFeatureDimension");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelOutputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_35(mht_35_v, 584, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_36(mht_36_v, 595, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetKernelSpatialDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()[pos];
}

int64_t mlirMhloConvDimensionNumbersGetOutputBatchDimension(
    MlirAttribute attr) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_37(mht_37_v, 605, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetOutputBatchDimension");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputBatchDimension();
}

int64_t mlirMhloConvDimensionNumbersGetOutputFeatureDimension(
    MlirAttribute attr) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_38(mht_38_v, 615, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetOutputFeatureDimension");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputFeatureDimension();
}

intptr_t mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsSize(
    MlirAttribute attr) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_39(mht_39_v, 625, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsSize");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()
      .size();
}

int64_t mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_40(mht_40_v, 636, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloConvDimensionNumbersGetOutputSpatialDimensionsElem");

  return unwrap(attr)
      .cast<mlir::mhlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()[pos];
}

//
// ComparisonDirectionAttr.
//
MlirAttribute mlirMhloComparisonDirectionAttrGet(MlirContext ctx,
                                                 const std::string &direction) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("direction: \"" + direction + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_41(mht_41_v, 650, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloComparisonDirectionAttrGet");

  llvm::Optional<mlir::mhlo::ComparisonDirection> compare_direction =
      mlir::mhlo::symbolizeComparisonDirection(direction);
  if (!compare_direction)
    llvm_unreachable("Invalid comparison-direction specified.");
  return wrap(mlir::mhlo::ComparisonDirectionAttr::get(
      unwrap(ctx), compare_direction.getValue()));
}

bool mlirMhloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_42(mht_42_v, 662, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAComparisonDirectionAttr");

  return unwrap(attr).isa<mlir::mhlo::ComparisonDirectionAttr>();
}

std::string mlirMhloComparisonDirectionAttrGetDirection(MlirAttribute attr) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_43(mht_43_v, 669, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloComparisonDirectionAttrGetDirection");

  return mlir::mhlo::stringifyComparisonDirection(
             unwrap(attr)
                 .cast<mlir::mhlo::ComparisonDirectionAttr>()
                 .getValue())
      .str();
}

//
// ComparisonTypeAttr.
//

MlirAttribute mlirMhloComparisonTypeAttrGet(MlirContext ctx,
                                            const std::string &type) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_44(mht_44_v, 686, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloComparisonTypeAttrGet");

  llvm::Optional<mlir::mhlo::ComparisonType> compare_type =
      mlir::mhlo::symbolizeComparisonType(type);
  if (!compare_type) llvm_unreachable("Invalid comparison-type specified.");
  return wrap(mlir::mhlo::ComparisonTypeAttr::get(unwrap(ctx),
                                                  compare_type.getValue()));
}

bool mlirMhloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_45(mht_45_v, 697, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAComparisonTypeAttr");

  return unwrap(attr).isa<mlir::mhlo::ComparisonTypeAttr>();
}

std::string mlirMhloComparisonTypeAttrGetType(MlirAttribute attr) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_46(mht_46_v, 704, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloComparisonTypeAttrGetType");

  return mlir::mhlo::stringifyComparisonType(
             unwrap(attr).cast<mlir::mhlo::ComparisonTypeAttr>().getValue())
      .str();
}

//
// PrecisionAttr.
//

MlirAttribute mlirMhloPrecisionAttrGet(MlirContext ctx,
                                       const std::string &type) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_47(mht_47_v, 719, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloPrecisionAttrGet");

  llvm::Optional<mlir::mhlo::Precision> precision_type =
      mlir::mhlo::symbolizePrecision(type);
  if (!precision_type) llvm_unreachable("Invalid precision-type specified.");
  return wrap(
      mlir::mhlo::PrecisionAttr::get(unwrap(ctx), precision_type.getValue()));
}

bool mlirMhloAttributeIsAPrecisionAttr(MlirAttribute attr) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_48(mht_48_v, 730, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAPrecisionAttr");

  return unwrap(attr).isa<mlir::mhlo::PrecisionAttr>();
}

std::string mlirMhloPrecisionAttrGetPrecision(MlirAttribute attr) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_49(mht_49_v, 737, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloPrecisionAttrGetPrecision");

  return mlir::mhlo::stringifyPrecision(
             unwrap(attr).cast<mlir::mhlo::PrecisionAttr>().getValue())
      .str();
}

//
// FftTypeAttr.
//

MlirAttribute mlirMhloFftTypeAttrGet(MlirContext ctx, const std::string &type) {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_50(mht_50_v, 751, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloFftTypeAttrGet");

  llvm::Optional<mlir::mhlo::FftType> fft_type =
      mlir::mhlo::symbolizeFftType(type);
  if (!fft_type) llvm_unreachable("Invalid fft-type specified.");
  return wrap(mlir::mhlo::FftTypeAttr::get(unwrap(ctx), fft_type.getValue()));
}

bool mlirMhloAttributeIsAFftTypeAttr(MlirAttribute attr) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_51(mht_51_v, 761, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAFftTypeAttr");

  return unwrap(attr).isa<mlir::mhlo::FftTypeAttr>();
}

std::string mlirMhloFftTypeAttrGetFftType(MlirAttribute attr) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_52(mht_52_v, 768, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloFftTypeAttrGetFftType");

  return mlir::mhlo::stringifyFftType(
             unwrap(attr).cast<mlir::mhlo::FftTypeAttr>().getValue())
      .str();
}

//
// DequantizeModeAttr.
//

MlirAttribute mlirMhloDequantizeModeAttrGet(MlirContext ctx,
                                            const std::string &mode) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("mode: \"" + mode + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_53(mht_53_v, 783, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDequantizeModeAttrGet");

  llvm::Optional<mlir::mhlo::DequantizeMode> dequantize_mode =
      mlir::mhlo::symbolizeDequantizeMode(mode);
  if (!dequantize_mode) llvm_unreachable("Invalid dequantize-mode specified.");
  return wrap(mlir::mhlo::DequantizeModeAttr::get(unwrap(ctx),
                                                  dequantize_mode.getValue()));
}

bool mlirMhloAttributeIsADequantizeModeAttr(MlirAttribute attr) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_54(mht_54_v, 794, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsADequantizeModeAttr");

  return unwrap(attr).isa<mlir::mhlo::DequantizeModeAttr>();
}

std::string mlirMhloDequantizeModeAttrGetDequantizeMode(MlirAttribute attr) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_55(mht_55_v, 801, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloDequantizeModeAttrGetDequantizeMode");

  return mlir::mhlo::stringifyDequantizeMode(
             unwrap(attr).cast<mlir::mhlo::DequantizeModeAttr>().getValue())
      .str();
}

//
// TransposeAttr.
//

MlirAttribute mlirMhloTransposeAttrGet(MlirContext ctx,
                                       const std::string &type) {
   std::vector<std::string> mht_56_v;
   mht_56_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_56(mht_56_v, 816, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloTransposeAttrGet");

  llvm::Optional<mlir::mhlo::Transpose> transpose_type =
      mlir::mhlo::symbolizeTranspose(type);
  if (!transpose_type) llvm_unreachable("Invalid transpose-type specified.");
  return wrap(
      mlir::mhlo::TransposeAttr::get(unwrap(ctx), transpose_type.getValue()));
}

bool mlirMhloAttributeIsATransposeAttr(MlirAttribute attr) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_57(mht_57_v, 827, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsATransposeAttr");

  return unwrap(attr).isa<mlir::mhlo::TransposeAttr>();
}

std::string mlirMhloTransposeAttrGetTranspose(MlirAttribute attr) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_58(mht_58_v, 834, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloTransposeAttrGetTranspose");

  return mlir::mhlo::stringifyTranspose(
             unwrap(attr).cast<mlir::mhlo::TransposeAttr>().getValue())
      .str();
}

//
// FusionKindAttr.
//

MlirAttribute mlirMhloFusionKindAttrGet(MlirContext ctx,
                                        const std::string &kind) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("kind: \"" + kind + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_59(mht_59_v, 849, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloFusionKindAttrGet");

  llvm::Optional<mlir::mhlo::FusionKind> fusion_kind =
      mlir::mhlo::symbolizeFusionKind(kind);
  if (!fusion_kind) llvm_unreachable("Invalid fusion-kind specified.");
  return wrap(
      mlir::mhlo::FusionKindAttr::get(unwrap(ctx), fusion_kind.getValue()));
}

bool mlirMhloAttributeIsAFusionKindAttr(MlirAttribute attr) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_60(mht_60_v, 860, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloAttributeIsAFusionKindAttr");

  return unwrap(attr).isa<mlir::mhlo::FusionKindAttr>();
}

std::string mlirMhloFusionKindAttrGetFusionKind(MlirAttribute attr) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSCAPIPSAttributesDTcpp mht_61(mht_61_v, 867, "", "./tensorflow/compiler/mlir/hlo/lib/CAPI/Attributes.cpp", "mlirMhloFusionKindAttrGetFusionKind");

  return mlir::mhlo::stringifyFusionKind(
             unwrap(attr).cast<mlir::mhlo::FusionKindAttr>().getValue())
      .str();
}
