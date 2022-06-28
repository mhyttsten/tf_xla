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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_BUFFER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh() {
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


#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// A container which can hold one or more HloValues. An HLO buffer abstractly
// represents the allocation which HLO instructions write into and read
// from. Generally there is a one-to-one correspondence between HloBuffers and
// HloValue where each HloValue in the module is held in a unique HloBuffer. An
// exception is the while instruction which updates the loop state in-place. In
// this case, we have a single HloBuffer for each HloPosition in the loop state,
// but multiple HloValues. For example:
//
//   %init = ...
//   %while = While(%init, body, condition)
//
//  body:
//   %body_param = Param(0)
//     ...
//   %body_root = ...
//
//  condition:
//   %cond_param = Param(0)
//     ...
//
// For simplicity, assume that %while is array-shaped. In this case, we have a
// single HloBuffer which holds the following HloValues: HloValue{%init},
// HloValue{%while}, HloValue{%body_param}, HloValue{%body_root}, and
// HloValue{%cond_param}.
//
// HloBuffers may appear at different HloPositions in the module mirroring the
// same property of HloValues. For example:
//
//   %sub = Sub(...)
//   %add = Add(...)
//   %tuple = Tuple(%add, %sub)
//   %gte = GetTupleElement(%tuple, 0)
//
// In this case, the HloBuffer containing %add appears at the following
// positions: HloPosition{%add, {}}, HloPosition{%tuple, {0}}, and
// HloPosition{%gte, {}}.
//
// Different HloPositions which share the same HloBuffer indicate mandatory
// aliasing in the HLO module. These positions must share the same memory
// allocation for correctness (the backends rely on this property). This differs
// from incidental aliasing introduced by memory reuse in BufferAssignment where
// different instructions may happen to get the same allocation.
class HloBuffer {
 public:
  using Id = int64_t;

  // Predicate comparing HloBuffers by increasing id, useful for std::sort.
  static bool IdLessThan(const HloBuffer* a, const HloBuffer* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_0(mht_0_v, 247, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "IdLessThan");

    return a->id() < b->id();
  }

  // Predicate comparing HloBuffers by equal id, useful for std::unique.
  static bool IdEqual(const HloBuffer* a, const HloBuffer* b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_1(mht_1_v, 255, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "IdEqual");

    return a->id() == b->id();
  }

  HloBuffer(Id id, std::vector<const HloValue*> values)
      : id_(id), values_(std::move(values)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_2(mht_2_v, 263, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "HloBuffer");
}

  // Return the unique identifier for this HloBuffer.
  Id id() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_3(mht_3_v, 269, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "id");
 return id_; }

  // Return all values contained in this buffer.
  const std::vector<const HloValue*>& values() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_4(mht_4_v, 275, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "values");
 return values_; }

  // Memory space color. Used to indicate the memory space that the hlo buffer
  // needs to live in.
  BufferValue::Color color() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_5(mht_5_v, 282, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "color");

    // Invariant: All values in the buffer should have the same color.
    BufferValue::Color result = values()[0]->color();
    for (const HloValue* value : values()) {
      DCHECK_EQ(result, value->color());
    }
    return result;
  }

  // Return the unique HLO value in the buffer. CHECK fails if the buffer does
  // not contain exactly one value.
  const HloValue& GetUniqueValue() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_bufferDTh mht_6(mht_6_v, 296, "", "./tensorflow/compiler/xla/service/hlo_buffer.h", "GetUniqueValue");

    CHECK_EQ(values_.size(), 1);
    return *values_[0];
  }

  std::vector<HloPosition> ComputePositions() const;

  std::string ToString() const;

  bool operator==(const HloBuffer& other) const;
  bool operator!=(const HloBuffer& other) const { return !(*this == other); }

 private:
  // Unique identifier for this HloBuffer.
  Id id_;

  // The set of values contained in this buffer. Vector contains no duplicates
  // and is sorted stably by HloValue::Id.
  std::vector<const HloValue*> values_;
};

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_BUFFER_H_
