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

#ifndef TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_TYPES_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh() {
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


#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
namespace TFR {

class TFRType : public Type {
 public:
  using Type::Type;

  static bool classof(Type type);
};

namespace detail {

struct TFRTypeStorage final
    : public TypeStorage,
      public llvm::TrailingObjects<TFRTypeStorage, StringAttr> {
  using KeyTy = ArrayRef<StringAttr>;

  explicit TFRTypeStorage(unsigned num_attrs) : num_attrs(num_attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "TFRTypeStorage");
}

  static TFRTypeStorage* construct(TypeStorageAllocator& allocator, KeyTy key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_1(mht_1_v, 219, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "construct");

    // Allocate a new storage instance.
    auto byteSize = TFRTypeStorage::totalSizeToAlloc<StringAttr>(key.size());
    auto rawMem = allocator.allocate(byteSize, alignof(TFRTypeStorage));
    auto result = ::new (rawMem) TFRTypeStorage(key.size());

    // Copy in the string attributes into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<StringAttr>());
    return result;
  }

  bool operator==(const KeyTy& attrs) const { return attrs == GetAttrs(); }

  KeyTy GetAttrs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_2(mht_2_v, 236, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "GetAttrs");

    return {getTrailingObjects<StringAttr>(), num_attrs};
  }

  unsigned num_attrs;
};

template <typename Derived>
class TFRTypeImpl : public Type::TypeBase<Derived, TFRType, TFRTypeStorage> {
 public:
  using Base = Type::TypeBase<Derived, TFRType, TFRTypeStorage>;
  using TFRBase = TFRTypeImpl<Derived>;
  using Base::Base;

  static Derived get(ArrayRef<StringAttr> attrs, MLIRContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_3(mht_3_v, 253, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "get");

    return Base::get(context, attrs);
  }

  static Derived getChecked(ArrayRef<StringAttr> attrs, Location loc) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_4(mht_4_v, 260, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "getChecked");

    return Base::getChecked(loc, loc.getContext(), attrs);
  }
  static Derived getChecked(function_ref<InFlightDiagnostic()> emitError,
                            MLIRContext* context, ArrayRef<StringAttr> attrs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_5(mht_5_v, 267, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "getChecked");

    return Base::getChecked(emitError, context, attrs);
  }

  static Derived get(MLIRContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_6(mht_6_v, 274, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "get");
 return get({}, context); }

  // TODO(fengliuai): fix the implementation
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<StringAttr> attrs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_7(mht_7_v, 281, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "verify");

    return success();
  }

  ArrayRef<StringAttr> getAttrKeys() { return Base::getImpl()->GetAttrs(); }
};
}  // namespace detail

class TFRTensorType : public detail::TFRTypeImpl<TFRTensorType> {
 public:
  using TFRBase::TFRBase;
  static std::string getTypeName() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_8(mht_8_v, 295, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "getTypeName");
 return "TFRTensorType"; }
};

class TFRTensorListType : public detail::TFRTypeImpl<TFRTensorListType> {
 public:
  using TFRBase::TFRBase;
  static std::string getTypeName() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_9(mht_9_v, 304, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "getTypeName");
 return "TFRTensorListType"; }
};

class TFRAttrType : public Type::TypeBase<TFRAttrType, TFRType, TypeStorage> {
 public:
  using Base::Base;
  static std::string getTypeName() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_typesDTh mht_10(mht_10_v, 313, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_types.h", "getTypeName");
 return "TFRAttrType"; }
};

}  // namespace TFR
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_TYPES_H_
