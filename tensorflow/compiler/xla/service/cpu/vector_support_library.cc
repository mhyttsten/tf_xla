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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/vector_support_library.h"

#include "absl/algorithm/container.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {
VectorSupportLibrary::VectorSupportLibrary(PrimitiveType primitive_type,
                                           int64_t vector_size,
                                           llvm::IRBuilder<>* b,
                                           std::string name)
    : vector_size_(vector_size),
      primitive_type_(primitive_type),
      b_(b),
      name_(std::move(name)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::VectorSupportLibrary");

  scalar_type_ = llvm_ir::PrimitiveTypeToIrType(
      primitive_type, b_->GetInsertBlock()->getModule());
  scalar_pointer_type_ = llvm::PointerType::getUnqual(scalar_type_);
  vector_type_ = llvm::VectorType::get(scalar_type_, vector_size, false);
  vector_pointer_type_ = llvm::PointerType::getUnqual(vector_type_);
}

static std::string TypeToString(llvm::Type* type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "TypeToString");

  std::string o;
  llvm::raw_string_ostream ostream(o);
  type->print(ostream);
  return ostream.str();
}

void VectorSupportLibrary::AssertCorrectTypes(
    std::initializer_list<llvm::Value*> values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::AssertCorrectTypes");

  for (llvm::Value* v : values) {
    llvm::Type* type = v->getType();
    if (type != scalar_type() && type != vector_type()) {
      LOG(FATAL) << "Expected either " << TypeToString(scalar_type()) << " or "
                 << TypeToString(vector_type()) << " but got "
                 << TypeToString(type);
    }
  }
}

llvm::Value* VectorSupportLibrary::Mul(llvm::Value* lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_3(mht_3_v, 238, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Mul");

  AssertCorrectTypes({lhs, rhs});
  return MulInternal(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::MulInternal(llvm::Value* lhs,
                                               llvm::Value* rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_4(mht_4_v, 247, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::MulInternal");

  if (scalar_type_->isFloatingPointTy()) {
    return b()->CreateFMul(lhs, rhs, name());
  } else {
    return b()->CreateMul(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::Add(llvm::Value* lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_5(mht_5_v, 258, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Add");

  AssertCorrectTypes({lhs, rhs});
  return AddInternal(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::Sub(llvm::Value* lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_6(mht_6_v, 266, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Sub");

  AssertCorrectTypes({lhs, rhs});
  return b()->CreateFSub(lhs, rhs);
}

llvm::Value* VectorSupportLibrary::Max(llvm::Value* lhs, llvm::Value* rhs,
                                       bool enable_fast_min_max) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_7(mht_7_v, 275, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Max");

  AssertCorrectTypes({lhs, rhs});
  if (scalar_type_->isFloatingPointTy()) {
    return llvm_ir::EmitFloatMax(lhs, rhs, b_, enable_fast_min_max);
  } else {
    LOG(FATAL) << "Max for integers is unimplemented";
  }
}

llvm::Value* VectorSupportLibrary::Floor(llvm::Value* a) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_8(mht_8_v, 287, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Floor");

  AssertCorrectTypes({a});
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::floor, {a},
                                      {a->getType()}, b());
}

llvm::Value* VectorSupportLibrary::Div(llvm::Value* lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_9(mht_9_v, 296, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Div");

  AssertCorrectTypes({lhs, rhs});
  if (scalar_type_->isFloatingPointTy()) {
    return b()->CreateFDiv(lhs, rhs, name());
  } else {
    LOG(FATAL) << "Division for integers is unimplemented";
  }
}

llvm::Value* VectorSupportLibrary::Clamp(llvm::Value* a,
                                         const llvm::APFloat& low,
                                         const llvm::APFloat& high) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_10(mht_10_v, 310, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::Clamp");

  CHECK(!low.isNaN());
  CHECK(!high.isNaN());
  CHECK(low.compare(high) == llvm::APFloat::cmpLessThan);

  AssertCorrectTypes({a});
  llvm::Type* type = a->getType();
  CHECK(scalar_type_->isFloatingPointTy());

  llvm::Value* low_value = GetConstantFloat(type, low);
  llvm::Value* high_value = GetConstantFloat(type, high);
  a = b_->CreateSelect(b_->CreateFCmpUGE(a, low_value), a, low_value);
  a = b_->CreateSelect(b_->CreateFCmpULE(a, high_value), a, high_value);
  return a;
}

llvm::Value* VectorSupportLibrary::FCmpEQMask(llvm::Value* lhs,
                                              llvm::Value* rhs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_11(mht_11_v, 330, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::FCmpEQMask");

  AssertCorrectTypes({lhs, rhs});
  return I1ToFloat(b()->CreateFCmpOEQ(lhs, rhs, name()));
}

llvm::Value* VectorSupportLibrary::FCmpOLTMask(llvm::Value* lhs,
                                               llvm::Value* rhs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_12(mht_12_v, 339, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::FCmpOLTMask");

  AssertCorrectTypes({lhs, rhs});
  return I1ToFloat(b()->CreateFCmpOLT(lhs, rhs, name()));
}

llvm::Value* VectorSupportLibrary::FCmpULEMask(llvm::Value* lhs,
                                               llvm::Value* rhs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_13(mht_13_v, 348, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::FCmpULEMask");

  AssertCorrectTypes({lhs, rhs});
  return I1ToFloat(b()->CreateFCmpULE(lhs, rhs, name()));
}

llvm::Value* VectorSupportLibrary::I1ToFloat(llvm::Value* i1) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_14(mht_14_v, 356, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::I1ToFloat");

  bool is_vector = llvm::isa<llvm::VectorType>(i1->getType());
  llvm::Type* integer_type = IntegerTypeForFloatSize(is_vector);
  return b()->CreateBitCast(b()->CreateSExt(i1, integer_type, name()),
                            is_vector ? vector_type() : scalar_type(), name());
}

llvm::Type* VectorSupportLibrary::IntegerTypeForFloatSize(bool vector) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_15(mht_15_v, 366, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::IntegerTypeForFloatSize");

  CHECK(scalar_type()->isFloatingPointTy());
  const llvm::DataLayout& data_layout =
      b()->GetInsertBlock()->getModule()->getDataLayout();
  int64_t float_size_bits = data_layout.getTypeSizeInBits(scalar_type());
  llvm::Type* scalar_int_type = b()->getIntNTy(float_size_bits);
  if (vector) {
    return llvm::VectorType::get(scalar_int_type, vector_size(), false);
  } else {
    return scalar_int_type;
  }
}

llvm::Value* VectorSupportLibrary::BroadcastScalar(llvm::Value* x) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_16(mht_16_v, 382, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::BroadcastScalar");

  CHECK_EQ(x->getType(), scalar_type());
  return b()->CreateVectorSplat(vector_size(), x, name());
}

llvm::Value* VectorSupportLibrary::FloatAnd(llvm::Value* lhs,
                                            llvm::Value* rhs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_17(mht_17_v, 391, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::FloatAnd");

  AssertCorrectTypes({lhs, rhs});
  llvm::Type* int_type =
      IntegerTypeForFloatSize(lhs->getType() == vector_type());
  return b()->CreateBitCast(
      b()->CreateAnd(b()->CreateBitCast(lhs, int_type, name()),
                     b()->CreateBitCast(rhs, int_type, name()), name()),
      vector_type());
}

llvm::Value* VectorSupportLibrary::FloatNot(llvm::Value* lhs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_18(mht_18_v, 404, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::FloatNot");

  AssertCorrectTypes({lhs});
  llvm::Type* int_type =
      IntegerTypeForFloatSize(lhs->getType() == vector_type());
  return b()->CreateBitCast(
      b()->CreateNot(b()->CreateBitCast(lhs, int_type, name()), name()),
      vector_type());
}

llvm::Value* VectorSupportLibrary::FloatOr(llvm::Value* lhs, llvm::Value* rhs) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_19(mht_19_v, 416, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::FloatOr");

  AssertCorrectTypes({lhs, rhs});
  llvm::Type* int_type =
      IntegerTypeForFloatSize(lhs->getType() == vector_type());
  return b()->CreateBitCast(
      b()->CreateOr(b()->CreateBitCast(lhs, int_type, name()),
                    b()->CreateBitCast(rhs, int_type, name()), name()),
      vector_type(), name());
}

llvm::Value* VectorSupportLibrary::AddInternal(llvm::Value* lhs,
                                               llvm::Value* rhs) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_20(mht_20_v, 430, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::AddInternal");

  if (scalar_type_->isFloatingPointTy()) {
    return b()->CreateFAdd(lhs, rhs, name());
  } else {
    return b()->CreateAdd(lhs, rhs, name());
  }
}

llvm::Value* VectorSupportLibrary::ComputeOffsetPointer(
    llvm::Value* base_pointer, llvm::Value* offset_elements) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_21(mht_21_v, 442, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::ComputeOffsetPointer");

  if (base_pointer->getType() != scalar_pointer_type()) {
    base_pointer =
        b()->CreateBitCast(base_pointer, scalar_pointer_type(), name());
  }
  return b()->CreateInBoundsGEP(scalar_type(), base_pointer, offset_elements,
                                name());
}

llvm::Value* VectorSupportLibrary::LoadVector(llvm::Value* pointer) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_22(mht_22_v, 454, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::LoadVector");

  if (pointer->getType() != vector_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, vector_pointer_type(), name());
  }
  return b()->CreateAlignedLoad(
      vector_type(), pointer,
      llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_)), name());
}

llvm::Value* VectorSupportLibrary::LoadScalar(llvm::Value* pointer) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_23(mht_23_v, 466, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::LoadScalar");

  if (pointer->getType() != scalar_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  return b()->CreateAlignedLoad(
      scalar_type(), pointer,
      llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_)), name());
}

void VectorSupportLibrary::StoreVector(llvm::Value* value,
                                       llvm::Value* pointer) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_24(mht_24_v, 479, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::StoreVector");

  AssertCorrectTypes({value});
  if (pointer->getType() != vector_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, vector_pointer_type());
  }
  b()->CreateAlignedStore(
      value, pointer,
      llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_)));
}

void VectorSupportLibrary::StoreScalar(llvm::Value* value,
                                       llvm::Value* pointer) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_25(mht_25_v, 493, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::StoreScalar");

  AssertCorrectTypes({value});
  if (pointer->getType() != scalar_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  b()->CreateAlignedStore(
      value, pointer,
      llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(primitive_type_)));
}

llvm::Value* VectorSupportLibrary::LoadBroadcast(llvm::Value* pointer) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_26(mht_26_v, 506, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::LoadBroadcast");

  if (pointer->getType() != scalar_pointer_type()) {
    pointer = b()->CreateBitCast(pointer, scalar_pointer_type(), name());
  }
  return b()->CreateVectorSplat(
      vector_size(), b()->CreateLoad(scalar_type(), pointer), name());
}

llvm::Value* VectorSupportLibrary::AddReduce(llvm::Value* vector) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_27(mht_27_v, 517, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::AddReduce");

  llvm::SmallVector<llvm::Constant*, 32> mask(vector_size(), nullptr);
  for (unsigned i = vector_size(); i != 1; i >>= 1) {
    // On every iteration, we shuffle half of the remaining lanes to the top
    // half of shuffle, and add two old and the new vector.

    for (unsigned j = 0; j < vector_size(); ++j) {
      if (j < (i / 2)) {
        mask[j] = b()->getInt32(i / 2 + j);
      } else {
        mask[j] = llvm::UndefValue::get(b()->getInt32Ty());
      }
    }

    llvm::Value* half_remaining_lanes =
        b()->CreateShuffleVector(vector, llvm::UndefValue::get(vector_type()),
                                 llvm::ConstantVector::get(mask), "");
    vector = Add(vector, half_remaining_lanes);
  }

  return b()->CreateExtractElement(vector, b()->getInt32(0), name());
}

llvm::Value* VectorSupportLibrary::AvxStyleHorizontalAdd(llvm::Value* lhs,
                                                         llvm::Value* rhs) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_28(mht_28_v, 544, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::AvxStyleHorizontalAdd");

  CHECK_EQ(lhs->getType(), vector_type());
  CHECK_EQ(rhs->getType(), vector_type());
  CHECK_EQ(vector_size() % 2, 0);

  llvm::SmallVector<llvm::Constant*, 32> mask_a, mask_b;

  // Adding the values shuffled using mask_a and mask_b gives us the
  // AVX-style horizontal add we want.  The masks work as documented
  // in https://llvm.org/docs/LangRef.html#shufflevector-instruction
  //
  // Here are the masks for vector_width() == 8:
  //
  //    index: |0 |1 |2 | 3 |4 |5 | 6 | 7
  //   --------+--+--+--+---+--+--+---+---
  //   mask_a: |0 |2 |8 |10 |4 |6 |12 |14
  //   mask_b: |1 |3 |9 |11 |5 |7 |13 |16
  //
  // So, as an example, the value at lane 3 of the result vector is
  // the result of adding lane 10 and lane 11 in the combined lhs++rhs
  // vector, which are the lanes 2 and 3 in the rhs vector.
  for (int i = 0; i < vector_size(); i += 2) {
    int increment = i < vector_size() / 2 ? 0 : (vector_size() / 2);
    mask_a.push_back(b()->getInt32(increment + i));
    mask_b.push_back(b()->getInt32(increment + i + 1));
  }
  for (int i = 0; i < vector_size(); i += 2) {
    int increment = i < vector_size() / 2 ? (vector_size() / 2) : vector_size();
    mask_a.push_back(b()->getInt32(increment + i));
    mask_b.push_back(b()->getInt32(increment + i + 1));
  }

  llvm::Value* shuffle_0 =
      b()->CreateShuffleVector(lhs, rhs, llvm::ConstantVector::get(mask_a));
  llvm::Value* shuffle_1 =
      b()->CreateShuffleVector(lhs, rhs, llvm::ConstantVector::get(mask_b));

  return Add(shuffle_0, shuffle_1);
}

llvm::Value* VectorSupportLibrary::ExtractLowHalf(llvm::Value* vector) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_29(mht_29_v, 587, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::ExtractLowHalf");

  llvm::SmallVector<llvm::Constant*, 32> mask;
  for (int i = 0; i < vector_size() / 2; i++) {
    mask.push_back(b()->getInt32(i));
  }

  return b()->CreateShuffleVector(vector, llvm::UndefValue::get(vector_type()),
                                  llvm::ConstantVector::get(mask));
}

llvm::Value* VectorSupportLibrary::ExtractHighHalf(llvm::Value* vector) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_30(mht_30_v, 600, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::ExtractHighHalf");

  llvm::SmallVector<llvm::Constant*, 32> mask;
  for (int i = 0; i < vector_size() / 2; i++) {
    mask.push_back(b()->getInt32(i + vector_size() / 2));
  }

  return b()->CreateShuffleVector(vector, llvm::UndefValue::get(vector_type()),
                                  llvm::ConstantVector::get(mask));
}

std::vector<llvm::Value*> VectorSupportLibrary::ComputeHorizontalSums(
    std::vector<llvm::Value*> vectors, llvm::Value* init_values) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_31(mht_31_v, 614, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::ComputeHorizontalSums");

  const int x86_avx_vector_elements =
      TargetMachineFeatures::kX86AvxVectorByteSize / scalar_byte_size();
  if (vector_size() == x86_avx_vector_elements &&
      vectors.size() == x86_avx_vector_elements) {
    return ComputeAvxOptimizedHorizontalSums(std::move(vectors), init_values);
  }

  std::vector<llvm::Value*> result;
  std::transform(vectors.begin(), vectors.end(), std::back_inserter(result),
                 [this](llvm::Value* vector) { return AddReduce(vector); });
  if (init_values) {
    for (int64_t i = 0, e = result.size(); i < e; i++) {
      result[i] = Add(result[i],
                      b()->CreateExtractElement(init_values, b()->getInt32(i)));
    }
  }
  return result;
}

std::vector<llvm::Value*>
VectorSupportLibrary::ComputeAvxOptimizedHorizontalSums(
    std::vector<llvm::Value*> vectors, llvm::Value* init_values) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_32(mht_32_v, 639, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::ComputeAvxOptimizedHorizontalSums");

  // vectors are N llvm vector values, each with N elements.
  int64_t lane_width = vectors.size();

  while (vectors.size() != 2) {
    std::vector<llvm::Value*> new_vectors;
    new_vectors.reserve(vectors.size() / 2);
    for (int i = 0; i < vectors.size(); i += 2) {
      new_vectors.push_back(AvxStyleHorizontalAdd(vectors[i], vectors[i + 1]));
    }

    vectors = std::move(new_vectors);
  }

  llvm::Value* low =
      AddInternal(ExtractLowHalf(vectors[0]), ExtractHighHalf(vectors[0]));
  if (init_values) {
    low = AddInternal(ExtractLowHalf(init_values), low);
  }
  llvm::Value* high =
      AddInternal(ExtractLowHalf(vectors[1]), ExtractHighHalf(vectors[1]));
  if (init_values) {
    high = AddInternal(ExtractHighHalf(init_values), high);
  }

  // `low` has the first `lane_width / 2` horizontal reductions, and `high` has
  // the next `lane_width / 2` horizontal reductions.

  std::vector<llvm::Value*> results;
  for (int i = 0; i < lane_width; i++) {
    llvm::Value* scalar_result =
        b()->CreateExtractElement(i < (lane_width / 2) ? low : high,
                                  b()->getInt32(i % (lane_width / 2)), name());
    results.push_back(scalar_result);
  }

  return results;
}

llvm::Value* VectorSupportLibrary::GetZeroVector() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_33(mht_33_v, 681, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::GetZeroVector");

  return llvm::Constant::getNullValue(vector_type());
}

llvm::Value* VectorSupportLibrary::GetZeroScalar() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_34(mht_34_v, 688, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "VectorSupportLibrary::GetZeroScalar");

  return llvm::Constant::getNullValue(scalar_type());
}

LlvmVariable::LlvmVariable(llvm::Type* type, llvm::IRBuilder<>* b) : b_(b) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_35(mht_35_v, 695, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "LlvmVariable::LlvmVariable");

  alloca_ = llvm_ir::EmitAllocaAtFunctionEntry(type, "", b_);
}

llvm::Value* LlvmVariable::Get() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_36(mht_36_v, 702, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "LlvmVariable::Get");

  return b_->CreateLoad(alloca_->getType()->getPointerElementType(), alloca_);
}

void LlvmVariable::Set(llvm::Value* new_value) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_37(mht_37_v, 709, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "LlvmVariable::Set");

  b_->CreateStore(new_value, alloca_);
}

TileVariable::TileVariable(VectorSupportLibrary* vector_support,
                           std::vector<llvm::Value*> initial_value) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_38(mht_38_v, 717, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "TileVariable::TileVariable");

  for (llvm::Value* initial_vector_value : initial_value) {
    storage_.emplace_back(vector_support, initial_vector_value);
  }
}

std::vector<llvm::Value*> TileVariable::Get() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_39(mht_39_v, 726, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "TileVariable::Get");

  std::vector<llvm::Value*> result;
  absl::c_transform(storage_, std::back_inserter(result),
                    [&](VectorVariable vect_var) { return vect_var.Get(); });
  return result;
}

void TileVariable::Set(absl::Span<llvm::Value* const> value) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSvector_support_libraryDTcc mht_40(mht_40_v, 736, "", "./tensorflow/compiler/xla/service/cpu/vector_support_library.cc", "TileVariable::Set");

  CHECK_EQ(value.size(), storage_.size());
  for (int64_t i = 0, e = value.size(); i < e; i++) {
    storage_[i].Set(value[i]);
  }
}

}  // namespace cpu
}  // namespace xla
