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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_
#define TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh() {
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


#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

template <typename T>
std::string TypeNameVariant(const T& value);

template <typename T>
std::string DebugStringVariant(const T& value);

// Allows for specializations of Variant Decoding.  `data` may be modified in
// the process of decoding to `value`.
template <typename T>
bool DecodeVariant(VariantTensorData* data, T* value);

template <typename T>
bool DecodeVariant(std::string* buf, T* value);

template <typename T>
void EncodeVariant(const T& value, VariantTensorData* data);

template <typename T>
void EncodeVariant(const T& value, std::string* buf);

// This is an implementation of a type-erased container that can store an
// object of any type. The implementation is very similar to std::any, but has
// restrictions on the types of objects that can be stored, and eschews some of
// the fancier constructors available for std::any. An object of
// tensorflow::Variant is intended to be used as the value that will be stored
// in a tensorflow::Tensor object when its type is DT_VARIANT.
//
// tensorflow::Variant can store an object of a class that satisfies the
// following constraints:
//
// * The class is CopyConstructible.
// * The class has a default constructor.
// * It's either a protocol buffer, a tensorflow::Tensor, or defines the
// following functions:
//
//   string TypeName() const;
//   void Encode(VariantTensorData* data) const;
//   bool Decode(VariantTensorData data);
//
// Simple POD types can elide the Encode/Decode functions, they are provided by
// helper methods.
// Here are some typical usage patterns:
//
//   Variant x = 10;
//   EXPECT_EQ(*x.get<int>(), 10);
//
//   Tensor t(DT_FLOAT, TensorShape({}));
//   t.flat<float>()(0) = 42.0f;
//   Variant x = t;
//   EXPECT_EQ(x.get<Tensor>()->flat<float>()(0), 42.0f);
//
// Accessing the stored object:
//
// The get<T> function is the main mechanism to access the object
// stored in the container. It is type-safe, that is, calling
// get<T> when the stored object's type is not T, returns a
// nullptr. A raw pointer to the stored object can be obtained by calling
// get<void>().
//
// Serializing/deserializing Variant object:
//
// The Variant class delegates serializing and deserializing operations to the
// contained object. Helper functions to do these operations are provided for
// POD data types, tensorflow::Tensor, and protocol buffer objects. However,
// other classes have to provide Encode/Decode functions to handle
// serialization.
//
// Objects stored in a Variant object often contain references to other
// tensorflow::Tensors of primitive types (Eg., a list of tensorflow::Tensors).
// To efficiently support those use cases, a structure is imposed on the
// serialization format. Namely, classes should serialize their contents into a
// VariantTensorData object:
//
//   struct VariantTensorData {
//     string type_name;
//     string metadata;
//     std::vector<Tensor> tensors;
//   };
//
// Objects with references to other Tensors can simply store those tensors in
// the `tensors` field, and serialize other metadata content in to the
// `metadata` field.
//
// Serialization example:
//
//   Foo f = Foo {...};
//   Variant x = f;
//   string serialized_f;
//   x.Encode(&serialized_f);
//
//   Variant y = Foo(); // default constructed Foo.
//   y.Decode(std::move(serialized_f));
//   EXPECT_EQ(*x.get<Foo>(), *y.get<Foo>());
//
//
// A Variant storing serialized Variant data (a value of type
// VariantTensorDataProto) has different behavior from a standard Variant.
// Namely, its TypeName matches the TypeName of the original Variant;
// and its non-const get method performs lazy deserialization.
//
// Decode and copy example:
//
//   Foo f = Foo {...};
//   Variant x = f;
//
//   VariantTensorData serialized_data_f;
//   VariantTensorDataProto serialized_proto_f;
//   x.Encode(&serialized_data_f);
//   serialized_data_f.ToProto(&serialized_proto_f);
//
//   Variant y_type_unknown = serialized_proto_f;  // Store serialized Variant.
//
//   EXPECT_EQ(x.TypeName(), y_type_unknown.TypeName());  // Looks like Foo.
//   EXPECT_EQ(TypeIndex::Make<VariantTensorDataProto>(),
//             y_type_unknown.TypeId());
//
class Variant {
 public:
  // Constructs a Variant holding no value (aka `is_empty()`).
  //
  // This is done by pointing at nullptr via the heap value.
  Variant() noexcept : heap_value_(/*pointer=*/nullptr), is_inline_(false) {}

  ~Variant();

  Variant(const Variant& other);
  Variant(Variant&& other) noexcept;

  // Make sure that the type is CopyConstructible and not a
  // tensorflow::Variant object itself. We want the copy constructor to be
  // chosen for the tensorflow::Variant case.
  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_move_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant(T&& value);

  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_copy_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant(const T& value);

  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_copy_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant& operator=(const T& value);

  template <typename T, typename VT = typename std::decay<T>::type,
            typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                        std::is_move_constructible<VT>::value,
                                    void>::type* = nullptr>
  Variant& operator=(T&& value);

  Variant& operator=(const Variant& rhs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_0(mht_0_v, 358, "", "./tensorflow/core/framework/variant.h", "=");

    if (&rhs == this) return *this;
    Variant(rhs).swap(*this);
    return *this;
  }

  Variant& operator=(Variant&& rhs) noexcept {
    if (&rhs == this) return *this;
    Variant(std::move(rhs)).swap(*this);
    return *this;
  }

  // Constructs a value of type T with the given args in-place in this Variant.
  // Returns a reference to the newly constructed value.
  // The signature is based on std::variant<Types...>::emplace() in C++17.
  template <typename T, class... Args>
  T& emplace(Args&&... args) {
    ResetMemory();
    is_inline_ = CanInlineType<T>();
    if (is_inline_) {
      new (&inline_value_)
          InlineValue(InlineValue::Tag<T>{}, std::forward<Args>(args)...);
      return static_cast<Variant::Value<T>*>(inline_value_.AsValueInterface())
          ->value;
    } else {
      new (&heap_value_) HeapValue(
          absl::make_unique<Value<T>>(InPlace(), std::forward<Args>(args)...));
      return static_cast<Variant::Value<T>*>(heap_value_.get())->value;
    }
  }

  bool is_empty() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_1(mht_1_v, 392, "", "./tensorflow/core/framework/variant.h", "is_empty");
 return GetValue() == nullptr; }

  void clear() noexcept;

  void swap(Variant& other) noexcept;

  // Note, unlike TypeName(), TypeId() does not return the TypeIndex
  // of the original type when a TensorValueDataProto is stored as the
  // value.  In this case, it returns the TypeIndex of TensorValueDataProto.
  TypeIndex TypeId() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_2(mht_2_v, 404, "", "./tensorflow/core/framework/variant.h", "TypeId");

    const TypeIndex VoidTypeIndex = TypeIndex::Make<void>();
    if (is_empty()) {
      return VoidTypeIndex;
    }
    return GetValue()->TypeId();
  }

  std::string DebugString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_3(mht_3_v, 415, "", "./tensorflow/core/framework/variant.h", "DebugString");

    return strings::StrCat("Variant<type: ", TypeName(),
                           " value: ", SummarizeValue(), ">");
  }

  std::string SummarizeValue() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_4(mht_4_v, 423, "", "./tensorflow/core/framework/variant.h", "SummarizeValue");

    return is_empty() ? "[empty]" : GetValue()->DebugString();
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  T* get() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_5(mht_5_v, 433, "", "./tensorflow/core/framework/variant.h", "get");

    const TypeIndex TTypeIndex = TypeIndex::Make<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(static_cast<Variant::Value<T>*>(GetValue())->value);
  }

  // Returns a pointer to the stored value if it is type T, or nullptr
  // otherwise.
  template <typename T>
  const T* get() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_6(mht_6_v, 445, "", "./tensorflow/core/framework/variant.h", "get");

    const TypeIndex TTypeIndex = TypeIndex::Make<T>();
    if (is_empty() || (TTypeIndex != TypeId())) return nullptr;
    return std::addressof(
        static_cast<const Variant::Value<T>*>(GetValue())->value);
  }

  // Returns TypeNameVariant(value).
  //
  // In the special case that a serialized Variant is stored (value
  // is a VariantTensorDataProto), returns value.TypeName(), the
  // TypeName field stored in the VariantTensorDataProto buffer.
  std::string TypeName() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_7(mht_7_v, 460, "", "./tensorflow/core/framework/variant.h", "TypeName");

    if (is_empty()) {
      return "";
    }
    return GetValue()->TypeName();
  }

  // Serialize the contents of the stored object into `data`.
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_8(mht_8_v, 471, "", "./tensorflow/core/framework/variant.h", "Encode");

    if (!is_empty()) {
      GetValue()->Encode(data);
    }
  }

  // Deserialize `data` and update the stored object.
  bool Decode(VariantTensorData data);

  // Helper methods to directly serialize/deserialize from strings.
  void Encode(std::string* buf) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_9(mht_9_v, 484, "", "./tensorflow/core/framework/variant.h", "Encode");

    if (!is_empty()) {
      GetValue()->Encode(buf);
    }
  }
  bool Decode(std::string buf) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("buf: \"" + buf + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_10(mht_10_v, 493, "", "./tensorflow/core/framework/variant.h", "Decode");

    if (!is_empty()) {
      return GetValue()->Decode(std::move(buf));
    }
    return true;
  }

  template <typename VT>
  static constexpr bool CanInlineType() {
    return ((sizeof(Value<VT>) <= InlineValue::kMaxValueSize) &&
            (alignof(Value<VT>) <= kMaxInlineValueAlignSize));
  }

 private:
  struct in_place_t {};
  static constexpr in_place_t InPlace() { return in_place_t{}; }

  struct ValueInterface {
    virtual ~ValueInterface() = default;
    virtual TypeIndex TypeId() const = 0;
    virtual void* RawPtr() = 0;
    virtual const void* RawPtr() const = 0;
    virtual std::unique_ptr<ValueInterface> Clone() const = 0;
    virtual void CloneInto(ValueInterface* memory) const = 0;
    virtual void MoveAssign(ValueInterface* memory) = 0;
    virtual void MoveInto(ValueInterface* memory) = 0;
    virtual std::string TypeName() const = 0;
    virtual std::string DebugString() const = 0;
    virtual void Encode(VariantTensorData* data) const = 0;
    virtual bool Decode(VariantTensorData data) = 0;
    virtual void Encode(std::string* buf) const = 0;
    virtual bool Decode(std::string data) = 0;
  };

  template <typename T>
  struct Value final : ValueInterface {
    template <class... Args>
    explicit Value(in_place_t /*tag*/, Args&&... args)
        : value(std::forward<Args>(args)...) {}

    // NOTE(ebrevdo): Destructor must be explicitly defined for CUDA to happily
    // build `alignof(Variant<void*>)`.
    ~Value() final = default;

    TypeIndex TypeId() const final {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_11(mht_11_v, 540, "", "./tensorflow/core/framework/variant.h", "TypeId");

      const TypeIndex value_type_index =
          TypeIndex::Make<typename std::decay<T>::type>();
      return value_type_index;
    }

    void* RawPtr() final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_12(mht_12_v, 549, "", "./tensorflow/core/framework/variant.h", "RawPtr");
 return &value; }

    const void* RawPtr() const final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_13(mht_13_v, 554, "", "./tensorflow/core/framework/variant.h", "RawPtr");
 return &value; }

    std::unique_ptr<ValueInterface> Clone() const final {
      return absl::make_unique<Value>(InPlace(), value);
    }

    void MoveAssign(ValueInterface* memory) final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_14(mht_14_v, 563, "", "./tensorflow/core/framework/variant.h", "MoveAssign");

      CHECK(TypeId() == memory->TypeId())
          << TypeId().name() << " vs. " << memory->TypeId().name();
      static_cast<Value*>(memory)->value = std::move(value);
    }

    void CloneInto(ValueInterface* memory) const final {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_15(mht_15_v, 572, "", "./tensorflow/core/framework/variant.h", "CloneInto");

      new (memory) Value(InPlace(), value);
    }

    void MoveInto(ValueInterface* memory) final {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_16(mht_16_v, 579, "", "./tensorflow/core/framework/variant.h", "MoveInto");

      new (memory) Value(InPlace(), std::move(value));
    }

    std::string TypeName() const final {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_17(mht_17_v, 586, "", "./tensorflow/core/framework/variant.h", "TypeName");
 return TypeNameVariant(value); }

    std::string DebugString() const final {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_18(mht_18_v, 591, "", "./tensorflow/core/framework/variant.h", "DebugString");
 return DebugStringVariant(value); }

    void Encode(VariantTensorData* data) const final {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_19(mht_19_v, 596, "", "./tensorflow/core/framework/variant.h", "Encode");

      EncodeVariant(value, data);
    }

    bool Decode(VariantTensorData data) final {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_20(mht_20_v, 603, "", "./tensorflow/core/framework/variant.h", "Decode");

      return DecodeVariant(&data, &value);
    }

    void Encode(std::string* buf) const final {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_21(mht_21_v, 610, "", "./tensorflow/core/framework/variant.h", "Encode");
 EncodeVariant(value, buf); }

    bool Decode(std::string buf) final {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("buf: \"" + buf + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_22(mht_22_v, 616, "", "./tensorflow/core/framework/variant.h", "Decode");
 return DecodeVariant(&buf, &value); }

    T value;
  };
  static constexpr int kMaxInlineValueAlignSize = alignof(Value<void*>);

  using HeapValue = std::unique_ptr<ValueInterface>;

  struct InlineValue {
    // We try to size InlineValue so that sizeof(Variant) <= 64 and it can fit
    // into the aligned space of a TensorBuffer.
    static constexpr int kMaxValueSize = (64 - /*some extra padding=*/8);

    typedef char ValueDataArray[kMaxValueSize];
    alignas(kMaxInlineValueAlignSize) ValueDataArray value_data;

    // Tag is used for deducing the right type when constructing a Value in
    // place.
    template <typename VT>
    struct Tag {};

    template <typename VT, class... Args>
    explicit InlineValue(Tag<VT> /*tag*/, Args&&... args) noexcept {
      Value<VT>* inline_value_data = reinterpret_cast<Value<VT>*>(value_data);
      new (inline_value_data) Value<VT>(InPlace(), std::forward<Args>(args)...);
    }

    InlineValue(const InlineValue& other) noexcept {
      other.AsValueInterface()->CloneInto(AsValueInterface());
    }

    InlineValue(InlineValue&& other) noexcept {
      other.AsValueInterface()->MoveInto(AsValueInterface());
    }

    void ResetMemory() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_23(mht_23_v, 654, "", "./tensorflow/core/framework/variant.h", "ResetMemory");
 AsValueInterface()->~ValueInterface(); }

    InlineValue& operator=(const InlineValue& other) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_24(mht_24_v, 659, "", "./tensorflow/core/framework/variant.h", "=");

      if (&other == this) return *this;
      ResetMemory();
      other.AsValueInterface()->CloneInto(AsValueInterface());
      return *this;
    }

    InlineValue& operator=(InlineValue&& other) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_25(mht_25_v, 669, "", "./tensorflow/core/framework/variant.h", "=");

      if (&other == this) return *this;
      if (AsValueInterface()->TypeId() == other.AsValueInterface()->TypeId()) {
        other.AsValueInterface()->MoveAssign(AsValueInterface());
      } else {
        ResetMemory();
        other.AsValueInterface()->MoveInto(AsValueInterface());
      }
      return *this;
    }

    ValueInterface* AsValueInterface() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_26(mht_26_v, 683, "", "./tensorflow/core/framework/variant.h", "AsValueInterface");

      return reinterpret_cast<ValueInterface*>(value_data);
    }

    const ValueInterface* AsValueInterface() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_27(mht_27_v, 690, "", "./tensorflow/core/framework/variant.h", "AsValueInterface");

      return reinterpret_cast<const ValueInterface*>(value_data);
    }

    ~InlineValue() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_28(mht_28_v, 697, "", "./tensorflow/core/framework/variant.h", "~InlineValue");
 ResetMemory(); }
  };

  union {
    HeapValue heap_value_;
    InlineValue inline_value_;
  };
  // is_inline_ provides discrimination between which member of the prior union
  // is currently within it's lifetime. To switch from one member to the other,
  // the destructor must be called on the currently alive member before calling
  // the constructor on the other member. In effect, a member is expected to be
  // live at any given time and that member is tracked via this boolean.
  bool is_inline_;

  bool IsInlineValue() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_29(mht_29_v, 714, "", "./tensorflow/core/framework/variant.h", "IsInlineValue");
 return is_inline_; }

  // ResetMemory causes the destructor of the currently active member of the
  // union to be run. This must be follwed with a placement new call on the
  // member whose lifetime is to start. Additionally, is_inline_ needs to be set
  // accordingly. ResetAndSetInline and ResetAndSetHeap are simple helper
  // functions for performing the actions that are required to follow.
  void ResetMemory() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_30(mht_30_v, 724, "", "./tensorflow/core/framework/variant.h", "ResetMemory");

    if (IsInlineValue()) {
      inline_value_.~InlineValue();
    } else {
      heap_value_.~HeapValue();
    }
  }

  // ResetAndSetInline clears the current state and then constructs a new value
  // inline with the provided arguments.
  template <typename... Args>
  void ResetAndSetInline(Args&&... args) noexcept {
    ResetMemory();
    new (&inline_value_) InlineValue(std::forward<Args>(args)...);
    is_inline_ = true;
  }

  // ResetAndSetHeap clears the current state then constructs a new value on the
  // heap with the provided arguments.
  template <typename... Args>
  void ResetAndSetHeap(Args&&... args) noexcept {
    ResetMemory();
    new (&heap_value_) HeapValue(std::forward<Args>(args)...);
    is_inline_ = false;
  }

  ValueInterface* GetValue() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_31(mht_31_v, 753, "", "./tensorflow/core/framework/variant.h", "GetValue");

    if (IsInlineValue()) {
      return inline_value_.AsValueInterface();
    } else {
      return heap_value_.get();
    }
  }

  const ValueInterface* GetValue() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_32(mht_32_v, 764, "", "./tensorflow/core/framework/variant.h", "GetValue");

    if (IsInlineValue()) {
      return inline_value_.AsValueInterface();
    } else {
      return heap_value_.get();
    }
  }

  // PRECONDITION: Called on construction or ResetMemory() has been called
  // before this method.
  template <typename VT, typename T>
  void InsertValue(T&& value) {
    if (IsInlineValue()) {
      new (&inline_value_)
          InlineValue(InlineValue::Tag<VT>{}, std::forward<T>(value));
    } else {
      new (&heap_value_) HeapValue(
          absl::make_unique<Value<VT>>(InPlace(), std::forward<T>(value)));
    }
  }
};

// Make sure that a Variant object can reside in a 64-byte aligned Tensor
// buffer.
static_assert(sizeof(Variant) <= 64,
              "Expected internal representation to be 64 bytes.");

inline Variant::Variant(const Variant& other)
    : is_inline_(other.IsInlineValue()) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_33(mht_33_v, 795, "", "./tensorflow/core/framework/variant.h", "Variant::Variant");

  if (IsInlineValue()) {
    new (&inline_value_) InlineValue(other.inline_value_);
  } else {
    new (&heap_value_)
        HeapValue(other.heap_value_ ? other.heap_value_->Clone() : nullptr);
  }
}

inline Variant::Variant(Variant&& other) noexcept
    : is_inline_(other.IsInlineValue()) {
  if (IsInlineValue()) {
    new (&inline_value_) InlineValue(std::move(other.inline_value_));
  } else {
    new (&heap_value_) HeapValue(std::move(other.heap_value_));
  }
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_move_constructible<VT>::value,
                                  void>::type*>
inline Variant::Variant(T&& value) : is_inline_(CanInlineType<VT>()) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_34(mht_34_v, 820, "", "./tensorflow/core/framework/variant.h", "Variant::Variant");

  InsertValue<VT>(std::forward<T>(value));
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_copy_constructible<VT>::value,
                                  void>::type*>
inline Variant::Variant(const T& value) : is_inline_(CanInlineType<VT>()) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_35(mht_35_v, 831, "", "./tensorflow/core/framework/variant.h", "Variant::Variant");

  InsertValue<VT>(value);
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_move_constructible<VT>::value,
                                  void>::type*>
inline Variant& Variant::operator=(T&& value) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_36(mht_36_v, 842, "", "./tensorflow/core/framework/variant.h", "=");

  ResetMemory();
  is_inline_ = CanInlineType<VT>();
  InsertValue<VT>(std::forward<T>(value));
  return *this;
}

template <typename T, typename VT,
          typename std::enable_if<!std::is_same<Variant, VT>::value &&
                                      std::is_copy_constructible<VT>::value,
                                  void>::type*>
inline Variant& Variant::operator=(const T& value) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTh mht_37(mht_37_v, 856, "", "./tensorflow/core/framework/variant.h", "=");

  ResetMemory();
  is_inline_ = CanInlineType<VT>();
  InsertValue<VT>(value);
  return *this;
}

inline void Variant::clear() noexcept {
  // We set the internal unique_ptr to nullptr so that we preserve the
  // invariant that one of the two states must be set at all times. nullptr
  // indicates that the variant is empty.
  ResetAndSetHeap(/*pointer=*/nullptr);
}

inline void Variant::swap(Variant& other) noexcept {
  if (is_empty()) {
    if (other.IsInlineValue()) {
      ResetAndSetInline(std::move(other.inline_value_));
    } else {
      ResetAndSetHeap(std::move(other.heap_value_));
    }
    other.clear();
  } else if (other.is_empty()) {
    if (IsInlineValue()) {
      other.ResetAndSetInline(std::move(inline_value_));
    } else {
      other.ResetAndSetHeap(std::move(heap_value_));
    }
    clear();
  } else {  // Both Variants have values.
    if (other.IsInlineValue() && IsInlineValue()) {
      std::swap(inline_value_, other.inline_value_);
    } else if (!other.IsInlineValue() && !IsInlineValue()) {
      std::swap(heap_value_, other.heap_value_);
    } else if (other.IsInlineValue() && !IsInlineValue()) {
      HeapValue v = std::move(heap_value_);
      ResetAndSetInline(std::move(other.inline_value_));
      other.ResetAndSetHeap(std::move(v));
    } else {  // !other.IsInlineValue() && IsInlineValue()
      HeapValue v = std::move(other.heap_value_);
      other.ResetAndSetInline(std::move(inline_value_));
      ResetAndSetHeap(std::move(v));
    }
  }
}

template <>
void* Variant::get();

template <>
const void* Variant::get() const;

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_H_
