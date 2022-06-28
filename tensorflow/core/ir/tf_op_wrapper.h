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

#ifndef TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
#define TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
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
class MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh {
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
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh() {
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


#include <cstddef>

#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"

namespace mlir {
namespace detail {
// This class iterates over the control dependencies of the values.
template <typename ValueIteratorT>
class ControlRetIterator final
    : public llvm::mapped_iterator_base<ControlRetIterator<ValueIteratorT>,
                                        ValueIteratorT, Value> {
 public:
  using llvm::mapped_iterator_base<ControlRetIterator<ValueIteratorT>,
                                   ValueIteratorT, Value>::mapped_iterator_base;

  Value mapElement(Value value) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/ir/tf_op_wrapper.h", "mapElement");

    return value.getType().isa<tf_type::ControlType>()
               ? value
               : tfg::LookupControlDependency(value);
  }
};
}  // namespace detail

namespace tfg {

// Wrapper class exposing convenience methods to manipulate TensorFlow graph
// nodes uniformly.
class TFOp {
 public:
  // Wrap an operation. The operation can be null. The constructor must be
  // marked as implicit to support `llvm::dyn_cast`.
  TFOp(Operation *op = nullptr);  // NOLINT

  explicit TFOp(Operation &op) : TFOp(&op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_1(mht_1_v, 230, "", "./tensorflow/core/ir/tf_op_wrapper.h", "TFOp");
}

  // Support LLVM-style RTTI.
  static bool classof(Operation *op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_2(mht_2_v, 236, "", "./tensorflow/core/ir/tf_op_wrapper.h", "classof");

    return isa<TFGraphDialect>(op->getDialect());
  }

  // Get the wrapped operation.
  Operation *getOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_3(mht_3_v, 244, "", "./tensorflow/core/ir/tf_op_wrapper.h", "getOperation");
 return op_; }

  // Returns a pointer to the TensorFlow Graph Dialect. It nevers returns
  // nullptr.
  TFGraphDialect *getDialect() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_4(mht_4_v, 251, "", "./tensorflow/core/ir/tf_op_wrapper.h", "getDialect");

    return cast<TFGraphDialect>(op_->getDialect());
  }

  // Split the operands into data and control operands.
  std::pair<OperandRange, OperandRange> splitOperands() {
    ControlType ctl_type = getDialect()->getControlType();
    return SplitDataAndControlValues(op_->getOperands(), ctl_type);
  }

  // Returns the regular operands, the control operands will be excluded.
  OperandRange getNonControlOperands() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_5(mht_5_v, 265, "", "./tensorflow/core/ir/tf_op_wrapper.h", "getNonControlOperands");
 return splitOperands().first; }

  // The control operands are always after the regular inputs.
  OperandRange getControlOperands() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_6(mht_6_v, 271, "", "./tensorflow/core/ir/tf_op_wrapper.h", "getControlOperands");
 return splitOperands().second; }

  // Returns the control token produced by this operation.
  Value controlRet() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_7(mht_7_v, 277, "", "./tensorflow/core/ir/tf_op_wrapper.h", "controlRet");
 return op_->getResult(op_->getNumResults() - 1); }

  // Returns the non-control results produced by this operation.
  ResultRange getNonControlResults() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_8(mht_8_v, 283, "", "./tensorflow/core/ir/tf_op_wrapper.h", "getNonControlResults");

    return op_->getResults().slice(0, op_->getNumResults() - 1);
  }

  // Returns the node name for this operation.
  StringAttr nameAttr();
  StringRef name();
  // Set a new node name for this operation.
  void setName(const Twine &name);
  void setName(StringAttr name);

  // Returns the requested device, which is also the "device" field in a
  // GraphDef.
  StringAttr requestedDeviceAttr();
  StringRef requestedDevice();
  // Set a new requested device for this operation.
  void setRequestedDevice(const Twine &requested_device);
  void setRequestedDevice(StringAttr requested_device);

  // Returns the assigned device, this field is set by placer in general.
  StringAttr assignedDeviceAttr();
  StringRef assignedDevice();
  // Set a new assigned device for this operation.
  void setAssignedDevice(const Twine &assigned_device);
  void setAssignedDevice(StringAttr assigned_device);

  // Returns the assigned TPU cluster name.
  StringAttr tpuReplicate();
  // Set the assigned TPU cluster name.
  void setTpuReplicate(StringAttr tpu_replicate);

  // Returns the device, preferring the assigned device if set, and the
  // requested device otherwise.
  StringAttr deviceAttr() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_9(mht_9_v, 319, "", "./tensorflow/core/ir/tf_op_wrapper.h", "deviceAttr");

    StringAttr device = assignedDeviceAttr();
    if (device) return device;
    return requestedDeviceAttr();
  }
  StringRef device() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_10(mht_10_v, 327, "", "./tensorflow/core/ir/tf_op_wrapper.h", "device");

    StringAttr device_attr = deviceAttr();
    if (device_attr) return device_attr.getValue();
    return "";
  }

  // Forward `->` to the underlying operation, exposing the `Operation` methods.
  Operation *operator->() { return op_; }
  Operation &operator*() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_11(mht_11_v, 338, "", "./tensorflow/core/ir/tf_op_wrapper.h", "*");
 return *op_; }

  // Converts to true if there is a wrapped operation.
  explicit operator bool() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_12(mht_12_v, 344, "", "./tensorflow/core/ir/tf_op_wrapper.h", "bool");
 return op_; }

 private:
  // The wrapped operation.
  Operation *op_;
};

// A range iterator to get the control tokens associated with a value range.
// This range allows to wrap a ValueRange (or an OperandRange) and iterates on
// the control token associated to the producer of each value. For example, if
// you wrap the operands of an operation:
//     OperandControlRetRange range = op->getOperands();
// iterating this range will yield the control edges from each of the operations
// (or block arguments) producing these operands.
template <typename ValueRangeT>
class ControlRetRange final
    : public llvm::iterator_range<
          ::mlir::detail::ControlRetIterator<typename ValueRangeT::iterator>> {
 public:
  using Base = llvm::iterator_range<
      ::mlir::detail::ControlRetIterator<typename ValueRangeT::iterator>>;
  explicit ControlRetRange(ValueRangeT c) : Base(c.begin(), c.end()) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_13(mht_13_v, 368, "", "./tensorflow/core/ir/tf_op_wrapper.h", "ControlRetRange");
}

  /// Return the value at the given index.
  Value operator[](size_t index) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_14(mht_14_v, 374, "", "./tensorflow/core/ir/tf_op_wrapper.h", "lambda");

    assert(index < size() && "invalid index into value range");
    return *(this->begin() + index);
  }

  // Return the size of this range.
  size_t size() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_15(mht_15_v, 383, "", "./tensorflow/core/ir/tf_op_wrapper.h", "size");
 return llvm::size(*this); }

  // Return first value in the range.
  Value front() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTh mht_16(mht_16_v, 389, "", "./tensorflow/core/ir/tf_op_wrapper.h", "front");
 return (*this)[0]; }

  // Compare this range with another.
  template <typename OtherT>
  bool operator==(const OtherT &other) const {
    return llvm::size(*this) == llvm::size(other) &&
           std::equal(this->begin(), this->end(), other.begin());
  }
  template <typename OtherT>
  bool operator!=(const OtherT &other) const {
    return !(*this == other);
  }
};

using OperandControlRetRange = ControlRetRange<OperandRange>;
using ValueControlRetRange = ControlRetRange<ValueRange>;

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
