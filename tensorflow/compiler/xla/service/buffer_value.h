/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_VALUE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_VALUE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh() {
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
#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// Abstract class describing a value used by one of the dataflow analyses -
// TuplePointsToAnalysis or HloDataflowAnalysis.
// TODO(b/78906445) Delete this class when TuplePointsToAnalysis is unused.
//
// XLA arrays are trivially a single BufferValue. Tuples are made up of more
// than one BufferValue: an BufferValue for the pointer vector, and an
// BufferValue for each child element.
//
// Every BufferValue is defined by a particular instruction and most
// instructions define only a single BufferValue. Instructions which define a
// single BufferValue include array-shaped instructions such as Add but also
// includes Tuple-shaped instructions such as Tuple. The Tuple instruction
// defines a single BufferValue which is a vector of pointers to the values
// containing the Tuple instruction's operands. Though the result of the Tuple
// instruction includes multiple values only the top-level BufferValue (the
// vector of pointers) is defined by the Tuple instruction. The values
// containing the tuple elements are defined by earlier instructions, usually
// the operands of the Tuple instruction.
//
// Instructions which construct both the tuple *and* the tuple elements define
// more than one BufferValue. This includes (at least) tuple-shaped Constant,
// Parameter, Infeed and While instructions. These tuple-shaped instructions do
// not assemble a tuple from existing BufferValues like the Tuple instruction
// does, but rather define all the BufferValues in the tuple.
//
// Some instructions, such as Bitcast, define no buffers. These instructions
// simply forward buffers from their operands.
//
// The BufferValue object describes which HLO instruction defines a buffer and
// where within that instruction's output shape the buffer is defined. The
// location within the output shape is indicated by BufferValue::index() which
// is defined identically to the index used in ShapeUtil::GetSubshape().
// Examples:
//
// %add = Add(%foo, %bar)
// %tuple_constant = Constant({1, {42, 43}})
//
// %add defines a single array-shaped buffer BufferValue(%add, {}) which holds
// the array result of the add operation. The nested-tuple-shaped
// %tuple_constant defines 5 buffers described by the following BufferValue
// objects:
//
//   BufferValue(%tuple_constant, {})      // "Top-level" buffer: vector of
//                                         //  pointers to BufferValues at
//                                         //  indices {0} and {1}
//   BufferValue(%tuple_constant, {0})     // Holds value "1"
//   BufferValue(%tuple_constant, {1})     // Holds nested tuple: vector of
//                                         //  pointers to BufferValues at
//                                         //  indices {1, 0} and {1, 1}
//   BufferValue(%tuple_constant, {1, 0})  // Holds value "42"
//   BufferValue(%tuple_constant, {1, 1})  // Holds value "43"

class BufferValue {
 public:
  using Color = int64_t;

  // Id is a unique identifier for the BufferValue to facilitate efficient
  // collections of BufferValues with stable iteration order.
  using Id = int64_t;

  // Functions which return the size and alignment of a logical buffer in bytes.
  using SizeFunction = std::function<int64_t(const BufferValue&)>;
  using AlignmentFunction = std::function<int64_t(BufferValue::Color)>;

  // Prevent value being copied, allowing comparison by pointer,
  BufferValue(const BufferValue&) = delete;
  BufferValue& operator=(const BufferValue&) = delete;
  // ... but allow moves.
  BufferValue(BufferValue&&) = default;
  BufferValue& operator=(BufferValue&&) = default;

  virtual ~BufferValue() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh mht_0(mht_0_v, 272, "", "./tensorflow/compiler/xla/service/buffer_value.h", "~BufferValue");
}

  Id id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh mht_1(mht_1_v, 277, "", "./tensorflow/compiler/xla/service/buffer_value.h", "id");
 return id_; }

  // Return the instruction that defines the buffer.
  virtual HloInstruction* instruction() const = 0;

  // Return the index within the output of the instruction where the buffer is
  // defined. Index used defined as in ShapeUtil::GetSubshape()
  virtual const ShapeIndex& index() const = 0;

  // Return the color of the BufferValue. Differently colored buffers can not be
  // parts of the same allocation.
  ABSL_DEPRECATED("Use Layout::memory_space instead.")
  Color color() const {
    CHECK_NE(color_, kInvalidColor)
        << "Should not query the color of a buffer that was never colored";
    return color_;
  }

  ABSL_DEPRECATED("Use Layout::memory_space instead.")
  void set_color(Color color) {
    CHECK_NE(color, kInvalidColor)
        << "Should not set the color of a buffer to the invalid color";
    color_ = color;
  }

  ABSL_DEPRECATED("Use Layout::memory_space instead.")
  bool has_color() const { return color_ != kInvalidColor; }

  // Return the shape of the buffer. This reference points into the shape field
  // of the instruction defining the buffer.  Therefore, the returned shape will
  // contain the layout of instruction, if any.
  virtual const Shape& shape() const = 0;

  // Returns true if this buffer is the top-level output buffer of the defining
  // HLO instruction. This is equivalent to index == {}.
  bool IsTopLevel() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh mht_2(mht_2_v, 315, "", "./tensorflow/compiler/xla/service/buffer_value.h", "IsTopLevel");
 return index().empty(); }

  // Whether this buffer contains a tuple.
  bool IsTuple() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh mht_3(mht_3_v, 321, "", "./tensorflow/compiler/xla/service/buffer_value.h", "IsTuple");
 return is_tuple_; }

  // Whether this buffer contains an array.
  bool IsArray() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh mht_4(mht_4_v, 327, "", "./tensorflow/compiler/xla/service/buffer_value.h", "IsArray");
 return is_array_; }

  bool operator<(const BufferValue& other) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_valueDTh mht_5(mht_5_v, 332, "", "./tensorflow/compiler/xla/service/buffer_value.h", "operator<");
 return id_ < other.id_; }

  virtual std::string ToString() const = 0;

  // TODO(lauj) rename LogicalBufferProto to BufferValueProto.
  LogicalBufferProto ToProto(const SizeFunction& size_fn) const;

  // Returns the LogicalBufferProto::Location that serializes the given
  // instruction and index.
  static LogicalBufferProto::Location ToLocationProto(
      const HloInstruction& instruction, const ShapeIndex& index);

  const Color kInvalidColor = -1;

 protected:
  BufferValue(HloInstruction* instruction, const ShapeIndex& index, Id id);

 private:
  // The defining instruction and index are not stored here; they can be found
  // in the LogicalBuffer and HloValue subclasses. This class exists only to
  // support migrations from TuplePointsToAnalysis to HloDataflowAnalysis, by
  // allowing abstract use of LogicalBuffer or HloValue. After those migrations
  // are complete, this class should be deleted (b/78906445). Because we plan to
  // delete LogicalBuffer and this class, we don't refactor all the shared
  // features from LogicalBuffer and HloValue into this class.
  Id id_ : 62;
  bool is_array_ : 1;
  bool is_tuple_ : 1;
  Color color_ = kInvalidColor;
};

std::ostream& operator<<(std::ostream& out, const BufferValue& buffer);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_VALUE_H_
