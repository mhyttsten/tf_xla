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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_VECTOR_SUPPORT_LIBRARY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_VECTOR_SUPPORT_LIBRARY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh() {
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


#include <string>

#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace cpu {

// Simple wrappers around llvm::APFloat::APFloat to make the calling code more
// obvious.

inline llvm::APFloat GetIeeeF32(float f) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "GetIeeeF32");
 return llvm::APFloat(f); }
inline llvm::APFloat GetIeeeF32FromBitwiseRep(int32_t bitwise_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_1(mht_1_v, 207, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "GetIeeeF32FromBitwiseRep");

  return llvm::APFloat(llvm::APFloat::IEEEsingle(),
                       llvm::APInt(/*numBits=*/32, /*val=*/bitwise_value));
}

// A thin wrapper around llvm_util.h to make code generating vector math flow
// more readable.
class VectorSupportLibrary {
 public:
  // This VectorSupportLibrary instance remembers `primitive_type` and
  // `vector_size`, and these are implicitly used by the methods on this
  // instance (i.e. LoadVector will load a vector of type <`vector_size` x
  // `primitive_type`>).
  VectorSupportLibrary(PrimitiveType primitive_type, int64_t vector_size,
                       llvm::IRBuilder<>* b, std::string name);

  llvm::Value* Mul(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Mul(int64_t lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "Mul");

    return Mul(b()->getInt64(lhs), rhs);
  }
  llvm::Value* Mul(const llvm::APFloat& lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_3(mht_3_v, 233, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "Mul");

    return Mul(GetConstantFloat(rhs->getType(), lhs), rhs);
  }

  // If your call resolved to these then you probably wanted the versions taking
  // APFloat.
  llvm::Value* Mul(double lhs, llvm::Value* rhs) = delete;
  llvm::Value* Mul(float lhs, llvm::Value* rhs) = delete;

  llvm::Value* Add(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Add(int64_t lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_4(mht_4_v, 246, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "Add");

    return Add(b()->getInt64(lhs), rhs);
  }
  llvm::Value* Add(const llvm::APFloat& lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_5(mht_5_v, 252, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "Add");

    return Add(GetConstantFloat(rhs->getType(), lhs), rhs);
  }

  // If your call resolved to these then you probably wanted the versions taking
  // APFloat.
  llvm::Value* Add(double lhs, llvm::Value* rhs) = delete;
  llvm::Value* Add(float lhs, llvm::Value* rhs) = delete;

  llvm::Value* Sub(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* Sub(llvm::Value* lhs, const llvm::APFloat& rhs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_6(mht_6_v, 265, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "Sub");

    return Sub(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  llvm::Value* Max(llvm::Value* lhs, llvm::Value* rhs,
                   bool enable_fast_min_max);
  llvm::Value* Max(const llvm::APFloat& lhs, llvm::Value* rhs,
                   bool enable_fast_min_max) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_7(mht_7_v, 274, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "Max");

    return Max(GetConstantFloat(rhs->getType(), lhs), rhs, enable_fast_min_max);
  }
  llvm::Value* Div(llvm::Value* lhs, llvm::Value* rhs);

  llvm::Value* MulAdd(llvm::Value* a, llvm::Value* b, llvm::Value* c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_8(mht_8_v, 282, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "MulAdd");

    return Add(c, Mul(a, b));
  }

  llvm::Value* MulAdd(llvm::Value* a, llvm::Value* b, const llvm::APFloat& c) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_9(mht_9_v, 289, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "MulAdd");

    return Add(GetConstantFloat(vector_type(), c), Mul(a, b));
  }

  llvm::Value* MulAdd(llvm::Value* a, const llvm::APFloat& b,
                      const llvm::APFloat& c) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_10(mht_10_v, 297, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "MulAdd");

    return Add(GetConstantFloat(a->getType(), c),
               Mul(a, GetConstantFloat(a->getType(), b)));
  }

  llvm::Value* Floor(llvm::Value* a);

  // Precondition: Neither `low` nor `high` is nan.
  llvm::Value* Clamp(llvm::Value* a, const llvm::APFloat& low,
                     const llvm::APFloat& high);

  llvm::Value* SplatFloat(const llvm::APFloat& d) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_11(mht_11_v, 311, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "SplatFloat");

    return GetConstantFloat(vector_type(), d);
  }

  // These compare instructions return a floating point typed mask instead of an
  // i1.  For instance, on a vector typed input, lanes where the predicate is
  // true get a float with all ones and other lanes get a float with all zeros.
  // This is slightly odd from the perspective of LLVM's type system, but it
  // makes kernel IR generation code written using VectorSupportLibrary (its
  // raison d'etre) less cluttered.

  llvm::Value* FCmpEQMask(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* FCmpEQMask(llvm::Value* lhs, const llvm::APFloat& rhs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_12(mht_12_v, 326, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "FCmpEQMask");

    return FCmpEQMask(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  llvm::Value* FCmpULEMask(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* FCmpOLTMask(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* FCmpOLTMask(llvm::Value* lhs, const llvm::APFloat& rhs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_13(mht_13_v, 334, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "FCmpOLTMask");

    return FCmpOLTMask(lhs, GetConstantFloat(lhs->getType(), rhs));
  }

  // These boolean operations operate on the bitwise values of the floating
  // point inputs.  They return a (vector of) float(s) but like in the mask
  // generating predicates above this type system oddity makes the kernel IR
  // generation code less cluttered.
  llvm::Value* FloatAnd(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* FloatAnd(llvm::Value* lhs, const llvm::APFloat& rhs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_14(mht_14_v, 346, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "FloatAnd");

    return FloatAnd(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  llvm::Value* FloatOr(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* FloatOr(llvm::Value* lhs, const llvm::APFloat& rhs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_15(mht_15_v, 353, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "FloatOr");

    return FloatOr(lhs, GetConstantFloat(lhs->getType(), rhs));
  }
  llvm::Value* FloatNot(llvm::Value* lhs);
  llvm::Value* FloatAndNot(llvm::Value* lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_16(mht_16_v, 360, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "FloatAndNot");

    return FloatAnd(FloatNot(lhs), rhs);
  }

  llvm::Value* BroadcastScalar(llvm::Value* x);
  llvm::Value* BroadcastScalar(const llvm::APFloat& d) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_17(mht_17_v, 368, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "BroadcastScalar");

    return BroadcastScalar(GetConstantFloat(scalar_type(), d));
  }

  llvm::Value* ComputeOffsetPointer(llvm::Value* base_pointer,
                                    llvm::Value* offset_elements);
  llvm::Value* ComputeOffsetPointer(llvm::Value* base_pointer,
                                    llvm::Value* offset_elements,
                                    int64_t scale) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_18(mht_18_v, 379, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "ComputeOffsetPointer");

    return ComputeOffsetPointer(
        base_pointer, b_->CreateMul(b_->getInt64(scale), offset_elements));
  }
  llvm::Value* ComputeOffsetPointer(llvm::Value* base_pointer,
                                    int64_t offset_elements) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_19(mht_19_v, 387, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "ComputeOffsetPointer");

    return ComputeOffsetPointer(base_pointer, b()->getInt64(offset_elements));
  }

  llvm::Value* LoadVector(llvm::Value* pointer);

  llvm::Value* LoadVector(llvm::Value* base_pointer,
                          llvm::Value* offset_elements) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_20(mht_20_v, 397, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "LoadVector");

    return LoadVector(ComputeOffsetPointer(base_pointer, offset_elements));
  }

  llvm::Value* LoadVector(llvm::Value* base_pointer, int64_t offset_elements) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_21(mht_21_v, 404, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "LoadVector");

    return LoadVector(base_pointer, b()->getInt64(offset_elements));
  }

  llvm::Value* LoadScalar(llvm::Value* pointer);

  llvm::Value* LoadScalar(llvm::Value* base_pointer,
                          llvm::Value* offset_elements) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_22(mht_22_v, 414, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "LoadScalar");

    return LoadScalar(ComputeOffsetPointer(base_pointer, offset_elements));
  }

  llvm::Value* LoadScalar(llvm::Value* base_pointer, int64_t offset_elements) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_23(mht_23_v, 421, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "LoadScalar");

    return LoadScalar(base_pointer, b()->getInt64(offset_elements));
  }

  void StoreVector(llvm::Value* value, llvm::Value* pointer);

  void StoreVector(llvm::Value* value, llvm::Value* base_pointer,
                   llvm::Value* offset_elements) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_24(mht_24_v, 431, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "StoreVector");

    StoreVector(value, ComputeOffsetPointer(base_pointer, offset_elements));
  }

  void StoreVector(llvm::Value* value, llvm::Value* base_pointer,
                   int64_t offset_elements) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_25(mht_25_v, 439, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "StoreVector");

    StoreVector(value, base_pointer, b()->getInt64(offset_elements));
  }

  void StoreScalar(llvm::Value* value, llvm::Value* pointer);
  void StoreScalar(llvm::Value* value, llvm::Value* base_pointer,
                   llvm::Value* offset_elements) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_26(mht_26_v, 448, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "StoreScalar");

    StoreScalar(value, ComputeOffsetPointer(base_pointer, offset_elements));
  }

  void StoreScalar(llvm::Value* value, llvm::Value* base_pointer,
                   int64_t offset_elements) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_27(mht_27_v, 456, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "StoreScalar");

    StoreScalar(base_pointer, b()->getInt64(offset_elements));
  }

  llvm::Value* LoadBroadcast(llvm::Value* pointer);
  llvm::Value* LoadBroadcast(llvm::Value* base_pointer,
                             llvm::Value* offset_elements) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_28(mht_28_v, 465, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "LoadBroadcast");

    return LoadBroadcast(ComputeOffsetPointer(base_pointer, offset_elements));
  }
  llvm::Value* LoadBroadcast(llvm::Value* base_pointer,
                             int64_t offset_elements) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_29(mht_29_v, 472, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "LoadBroadcast");

    return LoadBroadcast(base_pointer, b()->getInt64(offset_elements));
  }

  // Compute the horizontal sum of each vector in `vectors`.  The i'th element
  // in the result vector is the (scalar) horizontal sum of the i'th vector in
  // `vectors`.  If `init_values` is not nullptr then the value in the i'th lane
  // in `init_values` is added to the i'th horizontal sum.
  std::vector<llvm::Value*> ComputeHorizontalSums(
      std::vector<llvm::Value*> vectors, llvm::Value* init_values = nullptr);

  llvm::Value* GetZeroVector();
  llvm::Value* GetZeroScalar();

  llvm::IRBuilder<>* b() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_30(mht_30_v, 489, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "b");
 return b_; }
  int64_t vector_size() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_31(mht_31_v, 493, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "vector_size");
 return vector_size_; }
  llvm::Type* vector_type() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_32(mht_32_v, 497, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "vector_type");
 return vector_type_; }
  llvm::Type* vector_pointer_type() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_33(mht_33_v, 501, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "vector_pointer_type");
 return vector_pointer_type_; }
  llvm::Type* scalar_type() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_34(mht_34_v, 505, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "scalar_type");
 return scalar_type_; }
  llvm::Type* scalar_pointer_type() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_35(mht_35_v, 509, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "scalar_pointer_type");
 return scalar_pointer_type_; }
  int64_t scalar_byte_size() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_36(mht_36_v, 513, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "scalar_byte_size");

    return primitive_util::BitWidth(primitive_type_) / 8;
  }

  const std::string& name() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_37(mht_37_v, 520, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "name");
 return name_; }

 private:
  llvm::Value* ExtractLowHalf(llvm::Value*);
  llvm::Value* ExtractHighHalf(llvm::Value*);

  llvm::Value* MulInternal(llvm::Value* lhs, llvm::Value* rhs);
  llvm::Value* AddInternal(llvm::Value* lhs, llvm::Value* rhs);

  llvm::Value* AddReduce(llvm::Value* vector);

  // Checks that each value in `values` is either of type scalar_type() or
  // vector_type().  This LOG(FATAL)'s so it should only be called in cases
  // where a mismatching type is a programmer bug.
  void AssertCorrectTypes(std::initializer_list<llvm::Value*> values);

  // Perform an X86 AVX style horizontal add between `lhs` and `rhs`.  The
  // resulting IR for an 8-float wide vector is expected to lower to a single
  // vhaddps instruction on a CPU that supports vhaddps, and not be too bad in
  // other cases.
  //
  // For a vector width of 8, the result vector is computed as:
  //   Result[0] = Lhs[0] + Lhs[1]
  //   Result[1] = Lhs[2] + Lhs[3]
  //   Result[2] = Rhs[0] + Rhs[1]
  //   Result[3] = Rhs[2] + Rhs[3]
  //   Result[4] = Lhs[4] + Lhs[5]
  //   Result[5] = Lhs[6] + Lhs[7]
  //   Result[6] = Rhs[4] + Rhs[5]
  //   Result[7] = Rhs[6] + Rhs[7]
  llvm::Value* AvxStyleHorizontalAdd(llvm::Value* lhs, llvm::Value* rhs);

  std::vector<llvm::Value*> ComputeAvxOptimizedHorizontalSums(
      std::vector<llvm::Value*> vectors, llvm::Value* init_values);

  llvm::Type* IntegerTypeForFloatSize(bool vector);
  llvm::Value* I1ToFloat(llvm::Value* i1);
  llvm::Value* GetConstantFloat(llvm::Type* type, const llvm::APFloat& f) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_38(mht_38_v, 560, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "GetConstantFloat");

    llvm::Constant* scalar_value = llvm::ConstantFP::get(type->getContext(), f);
    if (llvm::isa<llvm::VectorType>(type)) {
      return llvm::ConstantVector::getSplat(
          llvm::ElementCount::getFixed(vector_size()), scalar_value);
    }
    return scalar_value;
  }

  int64_t vector_size_;
  PrimitiveType primitive_type_;
  llvm::IRBuilder<>* b_;
  llvm::Type* vector_type_;
  llvm::Type* vector_pointer_type_;
  llvm::Type* scalar_type_;
  llvm::Type* scalar_pointer_type_;
  std::string name_;
};

// This wraps an alloca-backed stack variable which LLVM's SSA construction pass
// can later convert to a SSA value.
class LlvmVariable {
 public:
  LlvmVariable(llvm::Type*, llvm::IRBuilder<>* b);

  llvm::Value* Get() const;
  void Set(llvm::Value* new_value);

 private:
  llvm::AllocaInst* alloca_;
  llvm::IRBuilder<>* b_;
};

class VectorVariable : public LlvmVariable {
 public:
  VectorVariable(VectorSupportLibrary* vector_support,
                 llvm::Value* initial_value)
      : LlvmVariable(vector_support->vector_type(), vector_support->b()) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_39(mht_39_v, 600, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "VectorVariable");

    Set(initial_value);
  }
};

class ScalarVariable : public LlvmVariable {
 public:
  ScalarVariable(VectorSupportLibrary* vector_support,
                 llvm::Value* initial_value)
      : LlvmVariable(vector_support->scalar_type(), vector_support->b()) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTh mht_40(mht_40_v, 612, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.h", "ScalarVariable");

    Set(initial_value);
  }
};

// This wraps a set of alloca-backed stack variables that can, as a whole, store
// a tile.  A "tile" is a sequence of vectors that is typically used as a 2D
// grid of scalar values (e.g. for tiled GEMMs).
class TileVariable {
 public:
  TileVariable(VectorSupportLibrary* vector_support,
               std::vector<llvm::Value*> initial_value);

  std::vector<llvm::Value*> Get() const;
  void Set(absl::Span<llvm::Value* const> value);

 private:
  std::vector<VectorVariable> storage_;
};
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_VECTOR_SUPPORT_LIBRARY_H_
