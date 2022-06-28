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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc() {
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

#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.h"

#include <string>
#include <typeinfo>

#include "absl/strings/string_view.h"
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"

// Generated dialect defs.
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.cc.inc"

namespace mlir {
namespace dtensor {

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void DTensorDialect::initialize() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "DTensorDialect::initialize");

  addOperations<
#define GET_OP_LIST
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc.inc"
      >();
  registerAttributes();
}

// Parses a #dtensor.mesh attribute of the following format:
//
//   #dtensor.mesh<serializedMesh>
//
// where the first element is a SymbolRefAttr and the second element is the
// location.
static MeshAttr ParseMeshAttr(MLIRContext *context, StringRef spec,
                              Location loc) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_1(mht_1_v, 224, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "ParseMeshAttr");

  // Define error function.
  auto emit_error = [&](std::string text) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_2(mht_2_v, 230, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "lambda");

    emitError(loc, "invalid TensorFlow Mesh attribute ") << text;
    return nullptr;
  };
  // Check correct format and consume prefix, otherwise throw error.
  if (!spec.consume_front("mesh<"))
    return emit_error("Unexpected start to mesh specification");

  // Consume back from ">".
  if (!spec.consume_back(">"))
    return emit_error("Unexpected closing of mesh specification");

  // Cast from StringRef to string.
  std::string mesh_str = spec.str();

  // Check if serializedMesh is correct.
  using Mesh = tensorflow::dtensor::Mesh;
  using MeshOr = tensorflow::dtensor::StatusOr<Mesh>;
  MeshOr mesh_or = Mesh::FromString(mesh_str);
  if (!mesh_or.ok()) {
    std::string status_msg = mesh_or.status().ToString();
    return emit_error("parsing serialized string. More details: " + status_msg);
  }
  return MeshAttr::get(context, mesh_or.ValueOrDie());
}

// Parses a #dtensor.layout attribute of the following format:
//
//   #dtensor.layout<serializedLayout>
static LayoutAttr ParseLayoutAttr(MLIRContext *context, StringRef spec,
                                  Location loc) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_3(mht_3_v, 263, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "ParseLayoutAttr");

  // Define error function.
  auto emit_error = [&](std::string text) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_4(mht_4_v, 269, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "lambda");

    emitError(loc, "invalid TensorFlow Mesh attribute ") << text;
    return nullptr;
  };
  // Check correct format and consume prefix, otherwise throw error.
  if (!spec.consume_front("layout<"))
    return emit_error("Unexpected start to layout specification");

  // Consume back from "\">".
  if (!spec.consume_back(">"))
    return emit_error("Unexpected closing of layout specification");

  // Cast into string
  std::string layout_str = spec.str();

  // Check if serializedMesh is correct, else error from line 37.
  using Layout = tensorflow::dtensor::Layout;
  using LayoutOr = tensorflow::dtensor::StatusOr<Layout>;
  LayoutOr layout_or = Layout::FromString(layout_str);
  if (!layout_or.ok()) {
    std::string status_msg = layout_or.status().ToString();
    return emit_error("parsing serialized string. More details: " + status_msg);
  }
  // Extract layout.
  Layout layout = layout_or.ValueOrDie();

  return LayoutAttr::get(context, layout);
}

Attribute DTensorDialect::parseAttribute(DialectAsmParser &parser,
                                         Type type) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_5(mht_5_v, 302, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "DTensorDialect::parseAttribute");

  StringRef spec = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  if (spec.startswith("mesh")) return ParseMeshAttr(getContext(), spec, loc);

  if (spec.startswith("layout"))
    return ParseLayoutAttr(getContext(), spec, loc);

  return (emitError(loc, "unknown DTensor attribute: " + spec), nullptr);
}

// Print a type registered to this dialect.
// Prints a #dtensor.dtensor attribute of the following format:
//
//   #dtensor.mesh<mesh>
static void printMeshAttr(MeshAttr attr, DialectAsmPrinter &os) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_6(mht_6_v, 321, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "printMeshAttr");

  os << "mesh<" << attr.getValue().ToString() << ">";
}

// Prints a #dtensor.dtensor attribute of the following format:
//
//   #dtensor.layout<layout>
static void printLayoutAttr(LayoutAttr attr, DialectAsmPrinter &os) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_7(mht_7_v, 331, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "printLayoutAttr");

  os << "layout<" << attr.getValue().ToString() << ">";
}

// Override general virtual function
void DTensorDialect::printAttribute(Attribute attr,
                                    DialectAsmPrinter &os) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSopsDTcc mht_8(mht_8_v, 340, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc", "DTensorDialect::printAttribute");

  // Cast into correct attribute and print
  if (auto mesh_attr = attr.dyn_cast<MeshAttr>()) printMeshAttr(mesh_attr, os);

  if (auto layout_attr = attr.dyn_cast<LayoutAttr>())
    printLayoutAttr(layout_attr, os);
}
}  // namespace dtensor
}  // namespace mlir

// Ops definition from ODS

#define GET_OP_CLASSES
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/ops.cc.inc"
