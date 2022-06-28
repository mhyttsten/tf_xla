/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh() {
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


#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
// This is a set data structure that provides a deterministic iteration order.
// The iteration order of elements only depends on the sequence of
// inserts/deletes, so as long as the inserts/deletes happen in the same
// sequence, the set will have the same iteration order.
//
// Assumes that T can be cheaply copied for simplicity.
template <typename T>
class OrderedSet {
 public:
  // Inserts `value` into the ordered set.  Returns true if the value was not
  // present in the set before the insertion.
  bool Insert(T value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/graphcycles/ordered_set.h", "Insert");

    bool new_insertion =
        value_to_index_.insert({value, value_sequence_.size()}).second;
    if (new_insertion) {
      value_sequence_.push_back(value);
    }
    return new_insertion;
  }

  // Removes `value` from the set.  Assumes `value` is already present in the
  // set.
  void Erase(T value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/service/graphcycles/ordered_set.h", "Erase");

    auto it = value_to_index_.find(value);
    DCHECK(it != value_to_index_.end());

    // Since we don't want to move values around in `value_sequence_` we swap
    // the value in the last position and with value to be deleted and then
    // pop_back.
    value_to_index_[value_sequence_.back()] = it->second;
    std::swap(value_sequence_[it->second], value_sequence_.back());
    value_sequence_.pop_back();
    value_to_index_.erase(it);
  }

  void Reserve(size_t new_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh mht_2(mht_2_v, 236, "", "./tensorflow/compiler/xla/service/graphcycles/ordered_set.h", "Reserve");

    value_to_index_.reserve(new_size);
    value_sequence_.reserve(new_size);
  }

  void Clear() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh mht_3(mht_3_v, 244, "", "./tensorflow/compiler/xla/service/graphcycles/ordered_set.h", "Clear");

    value_to_index_.clear();
    value_sequence_.clear();
  }

  bool Contains(T value) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh mht_4(mht_4_v, 252, "", "./tensorflow/compiler/xla/service/graphcycles/ordered_set.h", "Contains");
 return value_to_index_.contains(value); }
  size_t Size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgraphcyclesPSordered_setDTh mht_5(mht_5_v, 256, "", "./tensorflow/compiler/xla/service/graphcycles/ordered_set.h", "Size");
 return value_sequence_.size(); }

  absl::Span<T const> GetSequence() const { return value_sequence_; }

 private:
  // The stable order that we maintain through insertions and deletions.
  std::vector<T> value_sequence_;

  // Maps values to their indices in `value_sequence_`.
  absl::flat_hash_map<T, int> value_to_index_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
