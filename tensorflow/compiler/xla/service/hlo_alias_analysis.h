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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Analysis which allocates HloBuffers to HloValues.
class HloAliasAnalysis {
 public:
  // The callgraph of the given HloModule must be flattened
  // (xla::FlattenCallGraph) prior to running the analysis.
  static StatusOr<std::unique_ptr<HloAliasAnalysis>> Run(
      const HloModule* module,
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr);

  std::string ToString() const;

  // Return the buffer containing the given value.
  const HloBuffer& GetBufferContainingValue(const HloValue& value) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "GetBufferContainingValue");

    return *value_to_buffer_.at(&value);
  }
  HloBuffer& GetBufferContainingValue(const HloValue& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "GetBufferContainingValue");

    return *value_to_buffer_.at(&value);
  }

  // Return the HloBuffer with the given ID.
  const HloBuffer& GetBuffer(HloBuffer::Id buffer_id) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_2(mht_2_v, 232, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "GetBuffer");

    return buffers_.at(buffer_id);
  }
  HloBuffer& GetBuffer(HloBuffer::Id buffer_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_3(mht_3_v, 238, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "GetBuffer");

    return buffers_.at(buffer_id);
  }

  // Returns the unique buffer at the given position. CHECK fails if the buffer
  // set at that position does not contain exactly one buffer.
  const HloBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                                     const ShapeIndex& index = {}) const;
  HloBuffer& GetUniqueBufferAt(const HloInstruction* instruction,
                               const ShapeIndex& index = {});

  // Compute the set of buffers at the given instruction and index and return as
  // a vector. This set is exactly the union of the buffers containing the
  // HloValues at this position.
  std::vector<const HloBuffer*> ComputeBuffersAt(
      const HloInstruction* instruction, const ShapeIndex& index = {}) const;

  // Return a vector of all HloBuffers stabily sorted by HloBuffer::Id. This
  // vector is lazily computed. Mutating operations on HloAliasAnalysis may
  // invalidate the underlying vector requiring recomputation.
  const std::vector<HloBuffer>& buffers() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_4(mht_4_v, 261, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "buffers");
 return buffers_; }

  // Returns the underlying dataflow analysis used by this alias analysis.
  HloDataflowAnalysis& dataflow_analysis() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_5(mht_5_v, 267, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "dataflow_analysis");
 return *dataflow_analysis_; }

  // Returns true if a buffer lives out of the module.
  bool BufferLivesOut(const HloBuffer& buffer) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_6(mht_6_v, 273, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "BufferLivesOut");

    return live_out_buffers_.contains(&buffer);
  }

  // Returns true if a hlo value lives out of the module.
  bool ValueLivesOut(const HloValue& value) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_alias_analysisDTh mht_7(mht_7_v, 281, "", "./tensorflow/compiler/xla/service/hlo_alias_analysis.h", "ValueLivesOut");

    return live_out_buffers_.contains(&GetBufferContainingValue(value));
  }

  std::vector<const HloBuffer*> LiveOutBuffers() const {
    std::vector<const HloBuffer*> results(live_out_buffers_.begin(),
                                          live_out_buffers_.end());
    absl::c_sort(results, HloBuffer::IdLessThan);
    return results;
  }

 protected:
  explicit HloAliasAnalysis(const HloModule* module);

  // Verify various invariants of the alias analysis.
  Status Verify() const;

  const HloModule* module_;

  // A set of buffers that live out the module.
  absl::flat_hash_set<const HloBuffer*> live_out_buffers_;

  // The underlying dataflow analysis used by this alias analysis.
  std::unique_ptr<HloDataflowAnalysis> dataflow_analysis_;

  // A map indicating which buffer a value is contained in.
  absl::flat_hash_map<const HloValue*, HloBuffer*> value_to_buffer_;

  // A lazily constructed vector containing all HloBuffers sorted by
  // HloBuffer::Id.
  std::vector<HloBuffer> buffers_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_ALIAS_ANALYSIS_H_
