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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc() {
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

#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"

#include <utility>

#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"

namespace mlir {
namespace dtensor {

// Storage class for MeshAttr.
namespace detail {
struct MeshAttrStorage : public AttributeStorage {
  using Mesh = tensorflow::dtensor::Mesh;
  using KeyTy = Mesh;

  explicit MeshAttrStorage(Mesh mesh) : mesh(std::move(mesh)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_0(mht_0_v, 202, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "MeshAttrStorage");
}

  bool operator==(const KeyTy& key) const { return key == KeyTy(mesh); }

  static llvm::hash_code hashKey(const KeyTy& key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_1(mht_1_v, 209, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "hashKey");

    const Mesh& mesh = key;
    return llvm::hash_value(mesh.ToString());
  }

  static MeshAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                    const KeyTy& key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_2(mht_2_v, 218, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "construct");

    return new (allocator.allocate<MeshAttrStorage>()) MeshAttrStorage(key);
  }
  Mesh mesh;
};
}  // namespace detail

MeshAttr MeshAttr::get(MLIRContext* context, const Mesh& mesh) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_3(mht_3_v, 228, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "MeshAttr::get");

  return Base::get(context, mesh);
}

const MeshAttr::Mesh& MeshAttr::getValue() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_4(mht_4_v, 235, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "MeshAttr::getValue");
 return getImpl()->mesh; }

// The storage class for LayoutAttr.
namespace detail {
struct LayoutAttrStorage : public AttributeStorage {
  using Layout = tensorflow::dtensor::Layout;
  using KeyTy = Layout;

  explicit LayoutAttrStorage(Layout layout) : layout(std::move(layout)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_5(mht_5_v, 246, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "LayoutAttrStorage");
}

  bool operator==(const KeyTy& key) const { return key == KeyTy(layout); }

  static llvm::hash_code hashKey(const KeyTy& key) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_6(mht_6_v, 253, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "hashKey");

    const Layout& layout = key;
    return llvm::hash_value(layout.ToString());
  }

  static LayoutAttrStorage* construct(
      mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_7(mht_7_v, 262, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "construct");

    const Layout& layout = key;
    return new (allocator.allocate<LayoutAttrStorage>())
        LayoutAttrStorage(layout);
  }
  Layout layout;
};
}  // namespace detail

LayoutAttr LayoutAttr::get(mlir::MLIRContext* context,
                           tensorflow::dtensor::Layout layout) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_8(mht_8_v, 275, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "LayoutAttr::get");

  return Base::get(context, std::move(layout));
}

const LayoutAttr::Layout& LayoutAttr::getValue() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_9(mht_9_v, 282, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "LayoutAttr::getValue");

  return getImpl()->layout;
}

void DTensorDialect::registerAttributes() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_dialectPSirPSdtensor_attributesDTcc mht_10(mht_10_v, 289, "", "./tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.cc", "DTensorDialect::registerAttributes");

  addAttributes<dtensor::MeshAttr, dtensor::LayoutAttr>();
}

}  // namespace dtensor
}  // namespace mlir
