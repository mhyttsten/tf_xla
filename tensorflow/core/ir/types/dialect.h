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

#ifndef TENSORFLOW_CORE_IR_TYPE_DIALECT_H_
#define TENSORFLOW_CORE_IR_TYPE_DIALECT_H_
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
class MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh {
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
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh() {
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

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project

// Include the dialect class generated from dialect.td.
// The constructor and the printing/parsing of dialect types are manually
// implemented (see ops.cpp).
#include "tensorflow/core/ir/types/dialect.h.inc"

// Include the Type classes declaration generated from types.td
#define GET_TYPEDEF_CLASSES
#include "tensorflow/core/ir/types/types.h.inc"

namespace mlir {
namespace tf_type {

//===----------------------------------------------------------------------===//
// TensorFlow types
//===----------------------------------------------------------------------===//

// The base class in the TensorFlow type hierarchy.
class TensorFlowType : public Type {
 public:
  using Type::Type;

  // Support method to enable LLVM-style type casting.
  static bool classof(Type type);
};

// Returns true if the specified type is a valid TensorFlow element type.
inline bool IsValidTFElementType(Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/ir/types/dialect.h", "IsValidTFElementType");

  return type.isa<ComplexType, FloatType, IntegerType, TensorFlowType>();
}

// Returns true if this is a valid TensorFlow tensor type.
inline bool IsValidTFTensorType(Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_1(mht_1_v, 229, "", "./tensorflow/core/ir/types/dialect.h", "IsValidTFTensorType");

  // TensorFlow types should be tensors of one of the valid TensorFlow element
  // types.
  if (auto tensor_ty = type.dyn_cast<TensorType>())
    return IsValidTFElementType(tensor_ty.getElementType());
  return false;
}

namespace detail {
// Common implementation of TensorFlow types. The template argument indicates
// the concrete derived class per CRTP.
template <typename Derived>
class TensorFlowTypeImpl
    : public Type::TypeBase<Derived, TensorFlowType, TypeStorage> {
 public:
  using Base = typename Type::TypeBase<Derived, TensorFlowType, TypeStorage>;
  using TFBase = TensorFlowTypeImpl<Derived>;
  using Base::Base;
};
}  // namespace detail

// TensorFlowRefType class supports all the ref types in TensorFlow dialect.
class TensorFlowRefType : public TensorFlowType {
 public:
  using TensorFlowType::TensorFlowType;

  // Checks if a type is TensorFlow Ref type.
  static bool classof(Type type);

  // Converts a type to the corresponding TensorFlowRef type.
  static TensorFlowType get(Type type);
  static TensorFlowType getChecked(Type type, MLIRContext* context,
                                   Location loc) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_2(mht_2_v, 264, "", "./tensorflow/core/ir/types/dialect.h", "getChecked");

    if (failed(verify(loc, type))) {
      return TensorFlowRefType();
    }
    return get(type);
  }

  static LogicalResult verify(Location loc, Type type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_3(mht_3_v, 274, "", "./tensorflow/core/ir/types/dialect.h", "verify");

    // type should be a valid TensorFlow type.
    if (!IsValidTFTensorType(type)) {
      return emitError(loc) << "invalid TensorFlow type: " << type;
    }
    return success();
  }

  // Converts a TensorFlowRef type to the corresponding TensorFlow or standard
  // type.
  Type RemoveRef();
};

// Define a class for each individual TensorFlow type (dtype), see types.def
// for the list.
#define HANDLE_TF_TYPE(tftype, enumerant, name)                          \
  class tftype##Type : public detail::TensorFlowTypeImpl<tftype##Type> { \
   public:                                                               \
    using TFBase::TFBase;                                                \
  };
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
#include "tensorflow/core/ir/types/types.def"

namespace detail {
// Storage type contains inferred subtypes for TypeWithSubtype.
class TypeWithSubtypeStorage : public TypeStorage {
 public:
  using KeyTy = ArrayRef<TensorType>;

  // NOLINTNEXTLINE
  static TypeWithSubtypeStorage* construct(TypeStorageAllocator& allocator,
                                           const KeyTy& key) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_4(mht_4_v, 308, "", "./tensorflow/core/ir/types/dialect.h", "construct");

    ArrayRef<TensorType> subtypes = allocator.copyInto(key);
    return new (allocator.allocate<TypeWithSubtypeStorage>())
        TypeWithSubtypeStorage(subtypes);
  }

  explicit TypeWithSubtypeStorage(const KeyTy& key) : subtypes_(key) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_5(mht_5_v, 317, "", "./tensorflow/core/ir/types/dialect.h", "TypeWithSubtypeStorage");
}

  bool operator==(const KeyTy& key) const { return key == subtypes_; }

  static llvm::hash_code hashKey(const KeyTy& key) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_6(mht_6_v, 324, "", "./tensorflow/core/ir/types/dialect.h", "hashKey");

    return llvm::hash_combine_range(key.begin(), key.end());
  }

  KeyTy subtypes_;
};

// Common implementation of TensorFlow types with subtypes. These subtypes are
// opaque and their interpretation depends on the actual underlying type.
// The template argument indicates the concrete derived class per CRTP. Concrete
// classes must implement the following:
//   - `static std::string getTypeName()` that returns the name of the type for
//     verification logging.
template <typename Derived>
class TypeWithSubtypeImpl
    : public Type::TypeBase<Derived, TensorFlowType, TypeWithSubtypeStorage> {
 public:
  using Base = Type::TypeBase<Derived, TensorFlowType, TypeWithSubtypeStorage>;
  using TFBase = TypeWithSubtypeImpl<Derived>;
  using Base::Base;

  static Derived get(ArrayRef<TensorType> subtypes, MLIRContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_7(mht_7_v, 348, "", "./tensorflow/core/ir/types/dialect.h", "get");

    return Base::get(context, subtypes);
  }

  static Derived getChecked(ArrayRef<TensorType> subtypes, MLIRContext* context,
                            Location loc) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_8(mht_8_v, 356, "", "./tensorflow/core/ir/types/dialect.h", "getChecked");

    return Base::getChecked(loc, subtypes);
  }
  static Derived getChecked(function_ref<InFlightDiagnostic()> emitError,
                            MLIRContext* context,
                            ArrayRef<TensorType> subtypes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_9(mht_9_v, 364, "", "./tensorflow/core/ir/types/dialect.h", "getChecked");

    return Base::getChecked(emitError, context, subtypes);
  }

  static Derived get(MLIRContext* context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_10(mht_10_v, 371, "", "./tensorflow/core/ir/types/dialect.h", "get");
 return get({}, context); }

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<TensorType> subtypes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_11(mht_11_v, 377, "", "./tensorflow/core/ir/types/dialect.h", "verify");

    // Each of the subtypes should be a valid TensorFlow type.
    for (TensorType subtype : subtypes) {
      if (!IsValidTFTensorType(subtype)) {
        return emitError() << "invalid " << Derived::getTypeName()
                           << " subtype: " << subtype;
      }
    }
    return success();
  }

  ArrayRef<TensorType> getSubtypes() { return Base::getImpl()->subtypes_; }
};
}  // namespace detail

// TensorFlowTypeWithSubtype class supports all the types with subtypes in
// TensorFlow dialect.
class TensorFlowTypeWithSubtype : public TensorFlowType {
 public:
  using TensorFlowType::TensorFlowType;

  // Checks if a type is TensorFlow type with subtypes.
  static bool classof(Type type);

  // Converts a TypeWithSubtype type to the same type but without its subtypes.
  Type RemoveSubtypes();

  // Clone the current Type with new subtypes.
  TensorFlowTypeWithSubtype clone(ArrayRef<TensorType> new_subtypes);

  // Returns the subtypes.
  ArrayRef<TensorType> GetSubtypes();
};

// Returns the corresponding TensorFlow type with subtypes but without its
// subtypes.
inline Type GetDefaultTypeOf(TensorFlowTypeWithSubtype type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_12(mht_12_v, 416, "", "./tensorflow/core/ir/types/dialect.h", "GetDefaultTypeOf");

  return type.RemoveSubtypes();
}

// TensorFlow resource type is used to support TensorFlow resource variables,
// which represent shared, persistent state manipulated by a TensorFlow program.
// ResourceType stores shape and datatype for subtypes unlike most other data
// types that don't have any associated information.
class ResourceType : public detail::TypeWithSubtypeImpl<ResourceType> {
 public:
  using TFBase::TFBase;
  static std::string getTypeName() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_13(mht_13_v, 430, "", "./tensorflow/core/ir/types/dialect.h", "getTypeName");
 return "ResourceType"; }
};

// TensorFlow variant type is used to support arbitrary custom C++ data types.
// VariantType stores inferred shape and datatype for subtypes unlike most other
// data types that don't have any associated information. For example, variants
// encoding TensorList type stores the common shape and dtype of the list
// elements as the only subtype.
class VariantType : public detail::TypeWithSubtypeImpl<VariantType> {
 public:
  using TFBase::TFBase;
  static std::string getTypeName() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_14(mht_14_v, 444, "", "./tensorflow/core/ir/types/dialect.h", "getTypeName");
 return "VariantType"; }
};

// Given two types `a` and `b`, returns a refined type which is cast compatible
// with both `a` and `b` and is equal to or more precise than both of them. It
// returns empty Type if the input types are not cast compatible.
// Provides option to ignore ref types on 'a'. This is useful for TF ops that
// might allow operands to either be same as result type or be a ref type
// corresponding to it.
Type GetCastCompatibleType(Type a, Type b, bool may_ignore_ref_type_a = false);

// Returns whether two arrays of Type are broadcast compatible.
bool BroadcastCompatible(TypeRange lhs, TypeRange rhs);

// Returns whether the two elemental types are compatible. Shapes are compatible
// if:
// - the types are statically equal
// - could be dynamically equal
//   - considering dynamic shapes equal unless contradictory info known;
//   - element types are equivalent, modulo subtypes possible be less exact
//     (e.g., a resource type without subtype is considered compatible with
//      resource type with known subtype).
// Provide option to ignore ref types on 'lhs'.
bool HasCompatibleElementTypes(Type lhs, Type rhs,
                               bool may_ignore_ref_type_lhs = false);

// Returns true if all TensorFlow types can be cast to one
// another. In other words, a single run-time value is legal for all the types.
// For example, tensor<*xf32>, tensor<?xf32> and tensor<3xf32> are cast
// compatible.
bool AreCastCompatible(TypeRange types);

// Returns true if corresponding elements of lhs and rhs AreCastCompatible and
// lhs and rhs are the same length.
bool ArraysAreCastCompatible(TypeRange lhs, TypeRange rhs);

// If `ty` is a tensor type and its element type has subtypes, then returns a
// new type of same shape but dropped subtypes for the element type.
// Otherwise, if `ty` has subtypes, then returns corresponding type with dropped
// subtypes.
// Otherwise, returns the original type `ty`.
Type DropSubTypes(Type ty);

// If `ty` is a tensor type and has elements of a ref type, then returns a new
// type of same shape but corresponding non-ref type as element type.
// Otherwise, if `ty` is a ref type, then returns corresponding non-ref type.
// Otherwise, returns the original type `ty`.
Type DropRefType(Type ty);

// Convenience call for executing both `DropRefType` and `DropSubTypes`.
Type DropRefAndSubTypes(Type ty);

//===----------------------------------------------------------------------===//
// Utility iterators
//===----------------------------------------------------------------------===//

// An iterator for the tensor shapes of an op's operands of shaped types.
// Returns llvm::None if a operand is unranked; returns ArrayRef<int64_t> as the
// shape otherwise.
class OperandShapeIterator final
    : public llvm::mapped_iterator<Operation::operand_iterator,
                                   llvm::Optional<ArrayRef<int64_t>> (*)(
                                       Value)> {
 public:
  using reference = llvm::Optional<ArrayRef<int64_t>>;

  /// Initializes the operand shape iterator to the specified operand iterator.
  explicit OperandShapeIterator(Operation::operand_iterator it);
};

using OperandShapeRange = iterator_range<OperandShapeIterator>;

// An iterator for the tensor shapes of an op's results of shaped types.
// Returns llvm::None if a result is unranked; returns ArrayRef<int64_t> as the
// shape otherwise.
class ResultShapeIterator final
    : public llvm::mapped_iterator<Operation::result_iterator,
                                   llvm::Optional<ArrayRef<int64_t>> (*)(
                                       Value)> {
 public:
  using reference = llvm::Optional<ArrayRef<int64_t>>;

  /// Initializes the result shape iterator to the specified result iterator.
  explicit ResultShapeIterator(Operation::result_iterator it);
};

using ResultShapeRange = iterator_range<ResultShapeIterator>;

// Returns a range with just resource type values from the input range
// preserved.
template <typename RangeT>
auto filter_resources(RangeT&& range) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_15(mht_15_v, 538, "", "./tensorflow/core/ir/types/dialect.h", "filter_resources");

  return llvm::make_filter_range(std::forward<RangeT>(range), [](Value val) {
    return getElementTypeOrSelf(val.getType()).isa<ResourceType>();
  });
}

// Returns the element type if `type` is a `ShapedType` and the type itself
// otherwise, converting `TensorFlowRef` type to corresponding `TensorFlow` or
// standard type if necessary.
inline Type GetElementTypeOrSelfResolveRef(Type type) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSirPStypesPSdialectDTh mht_16(mht_16_v, 550, "", "./tensorflow/core/ir/types/dialect.h", "GetElementTypeOrSelfResolveRef");

  Type element_type = getElementTypeOrSelf(type);
  if (auto ref_type = element_type.dyn_cast<TensorFlowRefType>()) {
    element_type = ref_type.RemoveRef();
  }
  return element_type;
}

}  // namespace tf_type
}  // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "tensorflow/core/ir/types/attributes.h.inc"
#include "tensorflow/core/ir/types/attributes_enum.h.inc"

#endif  // TENSORFLOW_CORE_IR_TYPE_DIALECT_H_
