/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_FRAMEWORK_OPS_H_
#define TENSORFLOW_CC_FRAMEWORK_OPS_H_
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
class MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh() {
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


#include <type_traits>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

/// @defgroup core Core Tensorflow API

class Output;

/// @addtogroup core
/// @{

/// Represents a node in the computation graph.
class Operation {
 public:
  Operation() : node_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_0(mht_0_v, 208, "", "./tensorflow/cc/framework/ops.h", "Operation");
}
  explicit Operation(Node* n);

  int32 num_inputs() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_1(mht_1_v, 214, "", "./tensorflow/cc/framework/ops.h", "num_inputs");
 return node_->num_inputs(); }
  DataType input_type(int32_t o) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_2(mht_2_v, 218, "", "./tensorflow/cc/framework/ops.h", "input_type");
 return node_->input_type(o); }
  Output input(int32_t i) const;

  int32 num_outputs() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_3(mht_3_v, 224, "", "./tensorflow/cc/framework/ops.h", "num_outputs");
 return node_->num_outputs(); }
  DataType output_type(int32_t o) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_4(mht_4_v, 228, "", "./tensorflow/cc/framework/ops.h", "output_type");
 return node_->output_type(o); }
  Output output(int32_t i) const;

  Node* node() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_5(mht_5_v, 234, "", "./tensorflow/cc/framework/ops.h", "node");
 return node_; }

  uint64 hash(int32_t index) const;

  bool operator==(const Operation& other) const { return node_ == other.node_; }

 private:
  typedef std::vector<std::pair<Node*, int32>> Inputs;
  static Inputs GetInputs(Node* node);

  Inputs inputs_;
  Node* node_;
};

/// Represents a tensor value produced by an Operation.
class Output {
 public:
  Output() = default;
  explicit Output(Node* n) : op_(n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_6(mht_6_v, 255, "", "./tensorflow/cc/framework/ops.h", "Output");
}
  Output(Node* n, int32_t index) : op_(n), index_(index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_7(mht_7_v, 259, "", "./tensorflow/cc/framework/ops.h", "Output");
}
  Output(const Operation& op, int32_t index) : op_(op), index_(index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_8(mht_8_v, 263, "", "./tensorflow/cc/framework/ops.h", "Output");
}

  Operation op() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_9(mht_9_v, 268, "", "./tensorflow/cc/framework/ops.h", "op");
 return op_; }
  Node* node() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_10(mht_10_v, 272, "", "./tensorflow/cc/framework/ops.h", "node");
 return op().node(); }
  int32 index() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_11(mht_11_v, 276, "", "./tensorflow/cc/framework/ops.h", "index");
 return index_; }
  DataType type() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_12(mht_12_v, 280, "", "./tensorflow/cc/framework/ops.h", "type");
 return op_.output_type(index_); }
  std::string name() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_13(mht_13_v, 284, "", "./tensorflow/cc/framework/ops.h", "name");

    return strings::StrCat(node()->name(), ":", index());
  }
  bool operator==(const Output& other) const {
    return op_ == other.op_ && index_ == other.index_;
  }

  uint64 hash() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_14(mht_14_v, 294, "", "./tensorflow/cc/framework/ops.h", "hash");
 return op_.hash(index_); }

 private:
  Operation op_ = Operation(nullptr);
  int32 index_ = 0;
};

/// Hash class that can be used for e.g. storing Outputs in an unordered_map
struct OutputHash {
  std::size_t operator()(const Output& output) const {
    return Hash64Combine(std::hash<Node*>()(output.node()),
                         std::hash<int32>()(output.index()));
  }
};

/// Represents a tensor value that can be used as an operand to an Operation.
class Input {
 public:
  /// Initializer enables constructing an Input object from various kinds of C++
  /// constants such as simple primitive constants and nested initializer lists
  /// representing a multi-dimensional array. Initializer constructors are all
  /// templates, so the aforementioned kinds of C++ constants can be used to
  /// construct an Initializer. Initializer stores the value it got constructed
  /// with in a Tensor object.
  struct Initializer {
    /// Construct from a scalar value of an arithmetic type or a type that can
    /// be converted to a string (eg. a string literal).
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(const T& v) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_15(mht_15_v, 327, "", "./tensorflow/cc/framework/ops.h", "Initializer");
  // NOLINT(runtime/explicit)
      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(), TensorShape());
      t.flat<RealT>()(0) = RealT(v);
      tensor = t;
    }

    Initializer(const Tensor& t) : tensor(t) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_16(mht_16_v, 337, "", "./tensorflow/cc/framework/ops.h", "Initializer");
}  // NOLINT(runtime/explicit)

    /// Construct from a scalar value and an explicit shape
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(const T& v, const TensorShape& shape) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_17(mht_17_v, 346, "", "./tensorflow/cc/framework/ops.h", "Initializer");

      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(), shape);
      for (int64_t i = 0; i < t.NumElements(); ++i) {
        t.flat<RealT>()(i) = RealT(v);
      }
      tensor = t;
    }

    /// Construct from a initializer list of scalars (a one-dimensional tensor).
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(
        const std::initializer_list<T>& v) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_18(mht_18_v, 363, "", "./tensorflow/cc/framework/ops.h", "Initializer");
  // NOLINT(runtime/explicit)
      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(),
               TensorShape{static_cast<int>(v.size())});
      std::copy_n(v.begin(), v.size(), t.flat<RealT>().data());
      tensor = t;
    }

    /// Construct from a initializer list of scalars and an explicit shape.
    template <typename T, typename = typename std::enable_if<
                              std::is_arithmetic<T>::value ||
                              std::is_convertible<T, std::string>::value>::type>
    Initializer(const std::initializer_list<T>& v, const TensorShape& shape) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_19(mht_19_v, 378, "", "./tensorflow/cc/framework/ops.h", "Initializer");

      typedef typename RealType<T>::type RealT;
      Tensor t(DataTypeToEnum<RealT>::v(), shape);
      if (t.NumElements() != static_cast<int64_t>(v.size())) {
        status = errors::InvalidArgument(
            "Cannot construct a tensor with ", t.NumElements(),
            " from an initializer list with ", v.size(), " elements");
        return;
      }
      std::copy_n(v.begin(), v.size(), t.flat<RealT>().data());
      tensor = t;
    }

    /// Construct a multi-dimensional tensor from a nested initializer
    /// list. Note that C++ syntax allows nesting of arbitrarily typed
    /// initializer lists, so such invalid initializers cannot be disallowed at
    /// compile time. This function performs checks to make sure that the nested
    /// initializer list is indeed a valid multi-dimensional tensor.
    Initializer(const std::initializer_list<Initializer>& v);

    // START_SKIP_DOXYGEN
    template <typename T, bool = std::is_convertible<T, std::string>::value>
    struct RealType {
      typedef tstring type;
    };

    template <typename T>
    struct RealType<T, false> {
      typedef T type;
    };
    // END_SKIP_DOXYGEN

    TensorProto AsTensorProto() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_20(mht_20_v, 413, "", "./tensorflow/cc/framework/ops.h", "AsTensorProto");

      TensorProto tensor_proto;
      if (tensor.NumElements() > 1) {
        tensor.AsProtoTensorContent(&tensor_proto);
      } else {
        tensor.AsProtoField(&tensor_proto);
      }
      return tensor_proto;
    }

    Status status;
    Tensor tensor;
  };

  /// All of Input's constructors are implicit. Input can be implicitly
  /// constructed from the following objects :
  /// * Output: This is so that the output of an Operation can be directly used
  ///   as the input to a op wrapper, which takes Inputs.
  /// * A scalar, or a multi-dimensional tensor specified as a recursive
  ///   initializer list. This enables directly passing constants as
  ///   inputs to op wrappers.
  /// * A Tensor object.
  Input(const Output& o) : output_(o) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_21(mht_21_v, 438, "", "./tensorflow/cc/framework/ops.h", "Input");
}  // NOLINT(runtime/explicit)

  template <typename T, typename = typename std::enable_if<
                            std::is_arithmetic<T>::value ||
                            std::is_convertible<T, std::string>::value>::type>
  Input(const T& v)  // NOLINT(runtime/explicit)
      : Input(Initializer(v)) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_22(mht_22_v, 447, "", "./tensorflow/cc/framework/ops.h", "Input");
}

  Input(const Initializer& init)  // NOLINT(runtime/explicit)
      : status_(init.status),
        tensor_(init.tensor) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_23(mht_23_v, 454, "", "./tensorflow/cc/framework/ops.h", "Input");
}

  Input(const Tensor& t)  // NOLINT(runtime/explicit)
      : status_(Status::OK()),
        tensor_(t) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_24(mht_24_v, 461, "", "./tensorflow/cc/framework/ops.h", "Input");
}

  Input(const std::initializer_list<Initializer>&
            init) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_25(mht_25_v, 467, "", "./tensorflow/cc/framework/ops.h", "Input");
  // NOLINT(runtime/explicit)
    for (const auto& i : init) {
      if (!i.status.ok()) {
        status_ = i.status;
        return;
      }
    }
    tensor_ = Initializer(init).tensor;
  }

  /// Constructor specifying a node name, index and datatype. This should only
  /// be used for specifying a backward edge, needed by control flow.
  Input(const std::string& name, int32_t i, DataType dt)
      : node_name_(name), index_(i), data_type_(dt) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_26(mht_26_v, 484, "", "./tensorflow/cc/framework/ops.h", "Input");
}

  Node* node() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_27(mht_27_v, 489, "", "./tensorflow/cc/framework/ops.h", "node");
 return output_.node(); }
  std::string node_name() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_28(mht_28_v, 493, "", "./tensorflow/cc/framework/ops.h", "node_name");
 return node_name_; }
  int32 index() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_29(mht_29_v, 497, "", "./tensorflow/cc/framework/ops.h", "index");
 return node_name_.empty() ? output_.index() : index_; }
  DataType data_type() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_30(mht_30_v, 501, "", "./tensorflow/cc/framework/ops.h", "data_type");
 return data_type_; }
  Status status() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_31(mht_31_v, 505, "", "./tensorflow/cc/framework/ops.h", "status");
 return status_; }
  const Tensor& tensor() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_32(mht_32_v, 509, "", "./tensorflow/cc/framework/ops.h", "tensor");
 return tensor_; }

 private:
  Status status_;
  Output output_ = Output(Operation(nullptr), 0);
  Tensor tensor_;
  const std::string node_name_ = "";
  int32 index_ = 0;
  DataType data_type_ = DT_INVALID;
};

/// A type for representing the output of ops that produce more than one output,
/// or a list of tensors.
typedef std::vector<Output> OutputList;

/// A type for representing the input to ops that require a list of tensors.
class InputList {
 public:
  /// Implicitly convert a list of outputs to a list of inputs. This is useful
  /// to write code such as ops::Concat(ops::Split(x, 4)).
  InputList(const OutputList& out) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_33(mht_33_v, 532, "", "./tensorflow/cc/framework/ops.h", "InputList");
  // NOLINT(runtime/explicit)
    for (auto const& x : out) {
      inputs_.push_back(x);
    }
  }

  InputList(
      const std::initializer_list<Input>& inputs)  // NOLINT(runtime/explicit)
      : inputs_(inputs.begin(), inputs.end()) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_34(mht_34_v, 543, "", "./tensorflow/cc/framework/ops.h", "InputList");
}

  InputList(const tensorflow::gtl::ArraySlice<Input>&
                inputs)  // NOLINT(runtime/explicit)
      : inputs_(inputs.begin(), inputs.end()) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_35(mht_35_v, 550, "", "./tensorflow/cc/framework/ops.h", "InputList");
}

  InputList(
      const std::initializer_list<Output>& out) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_36(mht_36_v, 556, "", "./tensorflow/cc/framework/ops.h", "InputList");
  // NOLINT(runtime/explicit)
    for (auto const& x : out) {
      inputs_.push_back(x);
    }
  }

  typename std::vector<Input>::iterator begin() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_37(mht_37_v, 565, "", "./tensorflow/cc/framework/ops.h", "begin");
 return inputs_.begin(); }
  typename std::vector<Input>::iterator end() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_38(mht_38_v, 569, "", "./tensorflow/cc/framework/ops.h", "end");
 return inputs_.end(); }
  typename std::vector<Input>::const_iterator begin() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_39(mht_39_v, 573, "", "./tensorflow/cc/framework/ops.h", "begin");

    return inputs_.begin();
  }
  typename std::vector<Input>::const_iterator end() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTh mht_40(mht_40_v, 579, "", "./tensorflow/cc/framework/ops.h", "end");

    return inputs_.end();
  }

 private:
  std::vector<Input> inputs_;
};

/// @}

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_OPS_H_
