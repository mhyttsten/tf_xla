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
/// @file object.h
/// @brief Object hierarchy for the TensorFlow C++ API. All "objects" are
/// derived from the `Handle` class. Instances of `Handle` are referred to as
/// "handles". All handles have a tagged value.
///
/// Example Usage:
/// Object runtime = GetRuntime("tfrt");
/// Object module = runtime.Get("Import")("cool_mobilenet")
/// runtime.Get("Tensor")(Tuple(5,5,5), 3.3);
/// Object test = CreateModule("test");
/// test.Set("cool_function", callable);
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_OBJECT_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_OBJECT_H_
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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh() {
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
#include <utility>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {

using TaggedValue = impl::TaggedValue;
class Handle;

// Necessary forward declare.
template <class T>
Handle Convert(T value);

/// @brief Base Handle class that wraps TaggedValue data. All data creation and
/// manipulation should done using Handle instances. Users should not be working
/// with TaggedValues directly.

/// The `Handle` class contains a TaggedValue in the `value_` member, which
/// contains the underlying data. An object belonging to `Foo`, a derived class
/// of `Handle`, can be referred to as a `Foo` handle.
///
/// It is important that all derived classes do not add any new data fields.
/// This ensures that it is always safe to slice down (i.e. assign an object of
/// a derived class to the base class) a handle to the base Handle class.
class Handle {
 public:
  /// Default constructor, which initializes a TaggedValue with type NONE.
  Handle() : value_(TaggedValue::None()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_0(mht_0_v, 231, "", "./tensorflow/cc/experimental/libtf/object.h", "Handle");
}

 public:
  /// Constructs a handle from a TaggedValue.
  explicit Handle(TaggedValue value) : value_(std::move(value)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_1(mht_1_v, 238, "", "./tensorflow/cc/experimental/libtf/object.h", "Handle");
}
  // explicit Handle(TaggedValue value, Handle* class_input)
  //     : value_(std::move(value)), class_(class_input) {}
  // const Handle& type() { return *class_; }

 protected:
  /// The wrapped TaggedValue.
  TaggedValue value_;
  // effectively a "weak reference" to intern'd class value.
  // types are compared by comparing pointer values here.
  // Handle* class_;  // effectively a "weak reference" to intern'd class value.

  /// The Integer handle.
  friend class Integer;
  /// The Float handle.
  friend class Float;
  /// The String handle.
  friend class String;
  /// The Object handle.
  friend class Object;
  /// The List handle.
  friend class List;
  /// The Dictionary handle.
  friend class Dictionary;
  /// The Tuple handle.
  friend class Tuple;
  /// The Callable handle.
  friend class Callable;
  /// The Tensor handle.
  friend class Tensor;
  /// Converts a Handle instance to an instance of a derived class `T`.
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
  /// Infrastructure for converting a TaggedValue tuple function signature to an
  /// unpacked variable list.
  template <typename Fn, class TRET, class... ArgsOut>
  friend class UneraseCallHelper;
};

// Forward declare.
template <class T>
tensorflow::StatusOr<T> Cast(Handle handle);

/// @brief The None class for holding TaggedValues of type NONE.
class None final : public Handle {
 public:
  /// Creates a handle that wraps a NONE TaggedValue.
  None() : Handle(TaggedValue::None()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_2(mht_2_v, 288, "", "./tensorflow/cc/experimental/libtf/object.h", "None");
}

 private:
  explicit None(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_3(mht_3_v, 294, "", "./tensorflow/cc/experimental/libtf/object.h", "None");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The String class for holding TaggedValues of type STRING.
class String final : public Handle {
 public:
  /// Creates a handle that wraps a STRING TaggedValue.
  explicit String(const char* s) : Handle(TaggedValue(s)) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("s: \"" + (s == nullptr ? std::string("nullptr") : std::string((char*)s)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_4(mht_4_v, 307, "", "./tensorflow/cc/experimental/libtf/object.h", "String");
}
  /// Returns the underlying TaggedValue string.
  const char* get() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_5(mht_5_v, 312, "", "./tensorflow/cc/experimental/libtf/object.h", "get");
 return value_.s(); }

 private:
  // Private since it is in general unsafe.
  explicit String(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_6(mht_6_v, 319, "", "./tensorflow/cc/experimental/libtf/object.h", "String");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The `Object` class modeled after Python "objects".
///
/// An `Object` uses a TaggedValue dictionary to store its attributes. The
/// "__parent__" attribute is reserved.
class Object : public Handle {
 public:
  /// Constructs a handle that acts as an object.
  Object() : Handle(TaggedValue::Dict()) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_7(mht_7_v, 334, "", "./tensorflow/cc/experimental/libtf/object.h", "Object");
}
  /// Retrieves the key of the object's parent.
  static const String& ParentKey();

  /// @brief Gets an object member attribute`key`.
  ///
  /// If the `key` is not found in the object, the object's "__parent__"
  /// attribute is then searched.
  ///
  /// @tparam T The desired return type.
  /// @param key The key to look up.
  /// @return `StatusOr` wrapping the key's value.
  template <class T = Handle>
  tensorflow::StatusOr<T> Get(const String& key) {
    auto& dict = value_.dict();
    auto it = dict.find(key.value_);
    if (it != dict.end()) {
      return Cast<T>(Handle(it->second));
    } else {
      // Lookup in object stored by reference in attribute  "__parent__".
      auto it_class = dict.find(ParentKey().value_);
      if (it_class != dict.end()) {
        auto& class_dict_maybe = it_class->second;
        if (class_dict_maybe.type() == TaggedValue::DICT) {
          auto& dict = class_dict_maybe.dict();
          auto it = dict.find(key.value_);
          if (it != value_.dict().end()) {
            return Cast<T>(Handle(it->second));
          }
        }
      }
    }
    return tensorflow::errors::NotFound("Key not in dictionary.");
  }

  /// Sets `key` attribute with the underlying value of `h`.
  void Set(const String& key, Handle h) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_8(mht_8_v, 373, "", "./tensorflow/cc/experimental/libtf/object.h", "Set");

    value_.dict()[key.value_] = std::move(h.value_);
  }

  /// Removes `key` from the object's attributes.
  void Unset(const String& key) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_9(mht_9_v, 381, "", "./tensorflow/cc/experimental/libtf/object.h", "Unset");
 value_.dict().erase(key.value_); }
  // TODO(b/): Adding dir() is in the future.
 private:
  // Private since it is in general unsafe.
  explicit Object(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_10(mht_10_v, 388, "", "./tensorflow/cc/experimental/libtf/object.h", "Object");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Dictionary class for holding TaggedValues of type DICT.
class Dictionary final : public Handle {
 public:
  /// Constructs a handle that wraps a DICT TaggedValue.
  Dictionary() : Handle(TaggedValue::Dict()) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_11(mht_11_v, 400, "", "./tensorflow/cc/experimental/libtf/object.h", "Dictionary");
}
  // TODO(aselle): make this private to preserve invariant.

  /// Retrieves `key` with type `T`.
  template <class T>
  tensorflow::StatusOr<T> Get(const Handle& key) {
    auto it = value_.dict().find(key.value_);
    if (it != value_.dict().end()) return Cast<T>(Handle(it->second));
    return tensorflow::errors::NotFound("Key not in dictionary.");
  }
  /// Sets `key` with value `value`.
  void Set(const String& key, Handle value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_12(mht_12_v, 414, "", "./tensorflow/cc/experimental/libtf/object.h", "Set");

    value_.dict()[key.value_] = std::move(value.value_);
  }
  /// Sets `key` with value `value`.
  void Set(const Handle& key, Handle value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_13(mht_13_v, 421, "", "./tensorflow/cc/experimental/libtf/object.h", "Set");

    value_.dict()[key.value_] = std::move(value.value_);
  }
  /// Retrieves size of dictionary.
  size_t size() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_14(mht_14_v, 428, "", "./tensorflow/cc/experimental/libtf/object.h", "size");
 return value_.dict().size(); }

 private:
  // Private since it is in general unsafe.
  explicit Dictionary(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_15(mht_15_v, 435, "", "./tensorflow/cc/experimental/libtf/object.h", "Dictionary");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Integer class for holding TaggedValues of type INT.
class Integer final : public Handle {
 public:
  /// Creates a handle that wraps an INT TaggedValue.
  explicit Integer(Handle h) : Handle(h.value_) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_16(mht_16_v, 447, "", "./tensorflow/cc/experimental/libtf/object.h", "Integer");
}
  /// Creates a handle that wraps an INT TaggedValue.
  explicit Integer(int64_t i) : Handle(TaggedValue(i)) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_17(mht_17_v, 452, "", "./tensorflow/cc/experimental/libtf/object.h", "Integer");
}
  /// Retrieves the underlying integer value.
  int64_t get() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_18(mht_18_v, 457, "", "./tensorflow/cc/experimental/libtf/object.h", "get");
 return value_.i64().get(); }

 private:
  // Private since it is in general unsafe.
  explicit Integer(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_19(mht_19_v, 464, "", "./tensorflow/cc/experimental/libtf/object.h", "Integer");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Float class for holding TaggedValues of type FLOAT.
class Float final : public Handle {
 public:
  /// Constructs a Float handle that wraps a FLOAT TaggedValue.
  explicit Float(Handle h) : Handle(h.value_) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_20(mht_20_v, 476, "", "./tensorflow/cc/experimental/libtf/object.h", "Float");
}
  /// Constructs a Float handle that wraps a FLOAT TaggedValue.
  explicit Float(float i) : Handle(TaggedValue(i)) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_21(mht_21_v, 481, "", "./tensorflow/cc/experimental/libtf/object.h", "Float");
}
  /// Retrieves the underlying float value.
  float get() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_22(mht_22_v, 486, "", "./tensorflow/cc/experimental/libtf/object.h", "get");
 return value_.f32().get(); }

 private:
  // Private since it is in general unsafe.
  explicit Float(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_23(mht_23_v, 493, "", "./tensorflow/cc/experimental/libtf/object.h", "Float");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The Tensor class for holding TaggedValues of type TENSOR.
class Tensor final : public Handle {
 public:
  /// Constructs a Tensor handle from a Handle that wraps a TENSOR TaggedValue.
  explicit Tensor(Handle h) : Handle(h.value_) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_24(mht_24_v, 505, "", "./tensorflow/cc/experimental/libtf/object.h", "Tensor");
}

  /// @brief Retrieves the value of the Tensor handle.

  /// @param data Buffer in which to copy contents of the handle.
  /// @throws InvalidArgument Raises error if `data` is of invalid size.
  template <class T>
  tensorflow::Status GetValue(absl::Span<T> data) const;

 private:
  // Private since it is in general unsafe.
  explicit Tensor(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_25(mht_25_v, 519, "", "./tensorflow/cc/experimental/libtf/object.h", "Tensor");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

template <class T>
tensorflow::Status Tensor::GetValue(absl::Span<T> data) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_26(mht_26_v, 528, "", "./tensorflow/cc/experimental/libtf/object.h", "Tensor::GetValue");

  tensorflow::AbstractTensorPtr t;
  {
    const auto abstract_t = value_.tensor().get();
    if (!tensorflow::ImmediateExecutionTensorHandle::classof(abstract_t)) {
      return tensorflow::errors::InvalidArgument(
          "Attempting to get value of non eager tensor.");
    }
    auto imm_t =
        static_cast<tensorflow::ImmediateExecutionTensorHandle*>(abstract_t);
    tensorflow::Status status;
    t.reset(imm_t->Resolve(&status));
    if (!status.ok()) {
      return status;
    }
  }
  if (data.size() != t->NumElements()) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Mismatched number of elements: \n", "Expected: ", data.size(), "\n",
        "Actual: ", t->NumElements(), "\n"));
  }
  memcpy(data.data(), t->Data(), t->ByteSize());
  return tensorflow::Status::OK();
}

/// @brief The Tuple class for holding TaggedValues of type TUPLE.
class Tuple : public Handle {
 public:
  /// Constructs a Tuple handle.
  template <class... T>
  explicit Tuple(T... args) : Handle(TaggedValue::Tuple()) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_27(mht_27_v, 561, "", "./tensorflow/cc/experimental/libtf/object.h", "Handle");

    add(args...);
  }

  /// Retrieves value at index `i`.
  template <class T>
  tensorflow::StatusOr<T> Get(size_t i) {
    if (i >= value_.tuple().size())
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    return Cast<T>(Handle(value_.tuple()[i]));
  }

  /// Retrieves number of elements.
  size_t size() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_28(mht_28_v, 577, "", "./tensorflow/cc/experimental/libtf/object.h", "size");
 return value_.tuple().size(); }

 private:
  // Add an item to a tuple. Should only be done by special construction
  // like Callables (which are a friend).
  void add() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_29(mht_29_v, 585, "", "./tensorflow/cc/experimental/libtf/object.h", "add");
}
  template <class T, class... T2>
  void add(T arg, T2... args) {
    value_.tuple().emplace_back(Convert(arg).value_);
    add(args...);
  }

  // Private since it is in general unsafe.
  explicit Tuple(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_30(mht_30_v, 596, "", "./tensorflow/cc/experimental/libtf/object.h", "Tuple");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The List class for holding TaggedValues of type LIST.
class List final : public Handle {
 public:
  /// Constructs a List handle.
  template <class... T>
  explicit List(T... args) : Handle(TaggedValue::List()) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_31(mht_31_v, 609, "", "./tensorflow/cc/experimental/libtf/object.h", "Handle");
}
  /// Retrieves value at index `i`.
  template <class T>
  tensorflow::StatusOr<T> Get(size_t i) {
    if (i >= size()) {
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    }
    return Cast<T>(Handle(value_.list()[i]));
  }

  /// Sets value `h` at index `i`.
  tensorflow::Status Set(size_t i, Handle h) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_32(mht_32_v, 623, "", "./tensorflow/cc/experimental/libtf/object.h", "Set");

    if (i >= size()) {
      return tensorflow::errors::InvalidArgument("Out of bounds index.");
    }
    value_.list()[i] = std::move(h.value_);
    return tensorflow::Status::OK();
  }

  /// Appends `arg` to list.
  template <class T>
  void append(T arg) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_33(mht_33_v, 636, "", "./tensorflow/cc/experimental/libtf/object.h", "append");

    value_.list().emplace_back(Convert(arg).value_);
  }
  /// Retrieves size of list.
  size_t size() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_34(mht_34_v, 643, "", "./tensorflow/cc/experimental/libtf/object.h", "size");
 return value_.list().size(); }

 private:
  // Private since it is in general unsafe.
  explicit List(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_35(mht_35_v, 650, "", "./tensorflow/cc/experimental/libtf/object.h", "List");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

/// @brief The `KeywordArg` class for storing keyword arguments as name value
/// pairs.
class KeywordArg {
 public:
  explicit KeywordArg(const char* s) : key_(String(s)), value_() {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("s: \"" + (s == nullptr ? std::string("nullptr") : std::string((char*)s)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_36(mht_36_v, 663, "", "./tensorflow/cc/experimental/libtf/object.h", "KeywordArg");
}

  template <class T>
  KeywordArg& operator=(const T obj) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_37(mht_37_v, 669, "", "./tensorflow/cc/experimental/libtf/object.h", "=");

    value_ = Convert(obj);
    return *this;
  }

  friend class Callable;

 private:
  String key_;
  Handle value_;
};

/// @brief The Callable class for creating callables.
class Callable final : public Handle {
 private:
  // Collect arguments for call
  void CollectArgs(Tuple& args, Dictionary& kwargs, int idx) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_38(mht_38_v, 688, "", "./tensorflow/cc/experimental/libtf/object.h", "CollectArgs");
}
  template <typename T, typename... Types>
  void CollectArgs(Tuple& args, Dictionary& kwargs, int idx, T v,
                   Types... vars) {
    const Handle& o = Convert(v);
    args.value_.tuple().emplace_back(o.value_);
    CollectArgs(args, kwargs, idx + 1, vars...);
  }
  template <typename... Types>
  void CollectArgs(Tuple& args, Dictionary& kwargs, int idx, KeywordArg v,
                   Types... vars) {
    kwargs.Set(v.key_, v.value_);
    CollectArgs(args, kwargs, idx + 1, vars...);
  }

 public:
  /// @brief Calls the wrapped TaggedValue function on a variable argument
  /// list.
  template <typename TReturn = Handle, typename... Types>
  tensorflow::StatusOr<TReturn> Call(Types... vars) {
    Dictionary kwargs = Dictionary();
    Tuple args;
    CollectArgs(args, kwargs, 0, vars...);
    auto maybe_value =
        value_.func()(std::move(args.value_), std::move(kwargs.value_));
    if (!maybe_value.ok()) {
      return maybe_value.status();
    }
    return Cast<TReturn>(Handle(maybe_value.ValueOrDie()));
  }

 public:
  // TODO(aselle): need to find a way to write test w/o this being public.
  // Private since it is in general unsafe.
  explicit Callable(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_39(mht_39_v, 725, "", "./tensorflow/cc/experimental/libtf/object.h", "Callable");
}
  template <class T>
  friend tensorflow::StatusOr<T> Cast(Handle handle);
};

namespace internal {
/// @brief The Capsule class for holding pointers.
class Capsule final : public Handle {
 public:
  /// Statically cast the TaggedValue capsule to type `T`.
  template <class T>
  T cast() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_40(mht_40_v, 739, "", "./tensorflow/cc/experimental/libtf/object.h", "cast");

    return static_cast<T>(value_.capsule());
  }

 private:
  // Private since it is in general unsafe.
  explicit Capsule(TaggedValue v) : Handle(std::move(v)) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_41(mht_41_v, 748, "", "./tensorflow/cc/experimental/libtf/object.h", "Capsule");
}
  template <class T>
  friend tensorflow::StatusOr<T> tf::libtf::Cast(Handle handle);
};
}  // namespace internal

/// @defgroup Util Functions for type conversion
///
/// @brief Functions for retrieving and converting Handle types.
/// @{

/// Retrieves tagged type of `T` handle.
template <class T>
inline TaggedValue::Type TypeToTaggedType() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_42(mht_42_v, 764, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType");
}
/// Retrieves tagged type of base class handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Handle>() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_43(mht_43_v, 770, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Handle>");

  return TaggedValue::Type::NONE;
}
/// Retrieves tagged type of None handle.
template <>
inline TaggedValue::Type TypeToTaggedType<None>() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_44(mht_44_v, 778, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<None>");

  return TaggedValue::Type::NONE;
}
/// Retrieves tagged type of String handle.
template <>
inline TaggedValue::Type TypeToTaggedType<String>() {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_45(mht_45_v, 786, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<String>");

  return TaggedValue::Type::STRING;
}
/// Retrieves tagged type of Callable handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Callable>() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_46(mht_46_v, 794, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Callable>");

  return TaggedValue::Type::FUNC;
}
/// Retrieves tagged type of Integer handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Integer>() {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_47(mht_47_v, 802, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Integer>");

  return TaggedValue::Type::INT64;
}
/// Retrieves tagged type of Float handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Float>() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_48(mht_48_v, 810, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Float>");

  return TaggedValue::Type::FLOAT32;
}
/// Retrieves tagged type of Object handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Object>() {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_49(mht_49_v, 818, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Object>");

  return TaggedValue::Type::DICT;
}
/// Retrieves tagged type of Dictionary handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Dictionary>() {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_50(mht_50_v, 826, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Dictionary>");

  return TaggedValue::Type::DICT;
}
/// Retrieves tagged type of List handle.
template <>
inline TaggedValue::Type TypeToTaggedType<List>() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_51(mht_51_v, 834, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<List>");

  return TaggedValue::Type::LIST;
}
/// Retrieves tagged type of Tensor handle.
template <>
inline TaggedValue::Type TypeToTaggedType<Tensor>() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_52(mht_52_v, 842, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<Tensor>");

  return TaggedValue::Type::TENSOR;
}
/// Retrieves tagged type of Capsule handle.
template <>
inline TaggedValue::Type TypeToTaggedType<internal::Capsule>() {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_53(mht_53_v, 850, "", "./tensorflow/cc/experimental/libtf/object.h", "TypeToTaggedType<internal::Capsule>");

  return TaggedValue::Type::CAPSULE;
}
// TODO(unknown): fully populate

/// @brief Casts a handle to type `T`
///
/// @param handle The handle to cast.
/// @tparam T The target handle type.
/// @exception InvalidArgument Raises error if the underlying TaggedValue type
/// of `handle` is not equivalent to `T`.
template <class T>
tensorflow::StatusOr<T> Cast(Handle handle) {
  if (handle.value_.type() == TypeToTaggedType<T>() ||
      std::is_same<T, Handle>::value)
    return T((std::move(handle.value_)));
  return tensorflow::errors::InvalidArgument("Incompatible cast.");
}

// Converters for C++ primitives like float and int to handles. Allows callable
// calls and list appends to be more idiomatic.

/// Converts a C++ const char* to a String handle.
template <>
inline Handle Convert(const char* value) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_54(mht_54_v, 878, "", "./tensorflow/cc/experimental/libtf/object.h", "Convert");

  return String(value);
}
/// Converts a C++ int32_t to an Integer handle.
template <>
inline Handle Convert(int32_t value) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_55(mht_55_v, 886, "", "./tensorflow/cc/experimental/libtf/object.h", "Convert");

  return Integer(value);
}
/// Converts a C++ int64_t to an Integer handle.
template <>
inline Handle Convert(int64_t value) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_56(mht_56_v, 894, "", "./tensorflow/cc/experimental/libtf/object.h", "Convert");

  return Integer(value);
}
/// Converts a C++ float to an Integer handle.
template <>
inline Handle Convert(float value) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_57(mht_57_v, 902, "", "./tensorflow/cc/experimental/libtf/object.h", "Convert");

  return Float(value);
}
/// Converts a value with primitive type T to a Handle.
template <class T>
inline Handle Convert(T value) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_58(mht_58_v, 910, "", "./tensorflow/cc/experimental/libtf/object.h", "Convert");

  return Handle(std::move(value));
}

/// @}

// in the future it will be possible to make additional hard typed APIs
// by generating code by introspecting objects.

// Here's a code gen'd example
// The dynamic structure can be turned into it.
/*
class Tf : Object {
  Tensor ones(Tensor shape, String dtype);
  // ...
}
*/

// Adapter to allow users to define Callables. Use TFLIB_CALLABLE_ADAPTOR
// instead.
template <typename TF, typename TReturn, typename... TFuncArgs>
class CallableWrapper;

// Template extracts arguments from a lambda function. This base
// class definition inherits from a another specialization in order. We use
// this top level template to extract the function pointer associated with
// the created lambda functor class.
template <typename TLambda>
class CallableWrapperUnpackArgs
    : public CallableWrapperUnpackArgs<decltype(&TLambda::operator())> {
 public:
  CallableWrapperUnpackArgs(TLambda fn, const char* name)
      : CallableWrapperUnpackArgs<decltype(&TLambda::operator())>(fn, name) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_59(mht_59_v, 946, "", "./tensorflow/cc/experimental/libtf/object.h", "CallableWrapperUnpackArgs");
}
};

// This specialization unpacks the arguments from a normal function pointer.
template <typename TReturn, typename... TFuncArgs>
class CallableWrapperUnpackArgs<TReturn (*)(TFuncArgs...)>
    : public CallableWrapper<TReturn (*)(TFuncArgs...), TReturn, TFuncArgs...> {
  using Fn = TReturn (*)(TFuncArgs...);

 public:
  CallableWrapperUnpackArgs(Fn fn, const char* name)
      : CallableWrapper<Fn, TReturn, TFuncArgs...>(fn, name) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_60(mht_60_v, 961, "", "./tensorflow/cc/experimental/libtf/object.h", "CallableWrapperUnpackArgs");
}
};

// This is the second stage of extracting the arguments from lambda function.
// NOTE: CallableWrapper's first template argument is the type of the
// function or functor (not the member pointer).
template <typename TClass, typename TReturn, typename... TFuncArgs>
class CallableWrapperUnpackArgs<TReturn (TClass::*)(TFuncArgs...) const>
    : public CallableWrapper<TClass, TReturn, TFuncArgs...> {
  using Fn = TClass;

 public:
  CallableWrapperUnpackArgs(Fn fn, const char* name)
      : CallableWrapper<Fn, TReturn, TFuncArgs...>(fn, name) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_61(mht_61_v, 978, "", "./tensorflow/cc/experimental/libtf/object.h", "CallableWrapperUnpackArgs");
}
};

template <class Fn, typename TReturn, class... ArgsOut>
class UneraseCallHelper;

// UneraseCallHelper::Call allows transforming all the incoming arguments
// from a TaggedValue tuple to a variadic list of args.  The class template
// starts as a list of argument types and ends empty. The static member
// template starts empty and ends with the unerased types of the signature.

// Base case (all arguments are processed, so call the function TFunc.
template <class Fn, typename TReturn>
class UneraseCallHelper<Fn, TReturn> {
 public:
  template <typename... ArgsOut>
  static tensorflow::StatusOr<TaggedValue> Call(const char* name, Fn functor_,
                                                int argument_index,
                                                const TaggedValue& args_in,
                                                ArgsOut... args) {
    // Call concrete type function
    TReturn ret = functor_(args...);
    return ret.value_;
  }
};

// Unpack a single argument case. Each argument is then cast.
template <class Fn, typename TReturn, class TSignatureArg,
          class... TSignatureRest>
class UneraseCallHelper<Fn, TReturn, TSignatureArg, TSignatureRest...> {
 public:
  template <typename... TArgsOut>
  static tensorflow::StatusOr<TaggedValue> Call(const char* name, Fn fn,
                                                int argument_index,
                                                TaggedValue& args_in,
                                                TArgsOut... args) {
    Handle h(std::move(args_in.tuple()[argument_index]));
    tensorflow::StatusOr<TSignatureArg> x = Cast<TSignatureArg>(std::move(h));
    if (!x.ok())
      return tensorflow::errors::InvalidArgument(
          std::string("Function ") + name + " Arg " +
          std::to_string(argument_index) +
          " cannot be cast to desired signature type ");
    return UneraseCallHelper<Fn, TReturn, TSignatureRest...>::template Call(
        name, fn, argument_index + 1, args_in, args..., *x);
  }
};

// Template specialization that allows extracting arguments from a C function
// pointer.
template <class Fn, typename TReturn, typename... TFuncArgs>
class CallableWrapper {
 private:
  Fn functor_;
  const char* name_;

 public:
  explicit CallableWrapper(Fn fn, const char* name)
      : functor_(fn), name_(name) {
   std::vector<std::string> mht_62_v;
   mht_62_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSobjectDTh mht_62(mht_62_v, 1040, "", "./tensorflow/cc/experimental/libtf/object.h", "CallableWrapper");
}

  // Entry point of the Adaptor functor. Note args, and kwargs are attempted
  // to be moved.
  tensorflow::StatusOr<TaggedValue> operator()(TaggedValue args,
                                               TaggedValue kwargs) {
    constexpr size_t argument_count = sizeof...(TFuncArgs);
    if (argument_count != args.tuple().size())
      return tensorflow::errors::InvalidArgument(
          std::string("Function ") + name_ + " expected " +
          std::to_string(argument_count) + " args.");
    return UneraseCallHelper<Fn, TReturn, TFuncArgs...>::Call(name_, functor_,
                                                              0, args);
  }
};

// Wrap a function that uses object handles as arguments and return types
// with one that takes TaggedValues. For example:
// Tuple Pack(Integer, Float, String);
// TaggedValue callable = TFLIB_CALLABLE_ADAPTOR(Pack);
#define TFLIB_CALLABLE_ADAPTOR(x) ::tf::libtf::CreateCallableAdaptor(x, #x)

template <class TF>
TaggedValue CreateCallableAdaptor(TF x, const char* name) {
  return TaggedValue((CallableWrapperUnpackArgs<TF>(x, name)));
}

}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_OBJECT_H_
