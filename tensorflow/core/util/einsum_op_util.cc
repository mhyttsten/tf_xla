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
class MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc() {
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

#include "tensorflow/core/util/einsum_op_util.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

Status ValidateEinsumEquation(const string& equation,
                              gtl::InlinedVector<string, 2>* input_subscripts,
                              string* output_subscript) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("equation: \"" + equation + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/util/einsum_op_util.cc", "ValidateEinsumEquation");

  gtl::InlinedVector<string, 2> inputs_and_output_subscripts =
      absl::StrSplit(equation, "->");
  if (inputs_and_output_subscripts.size() != 2) {
    return errors::InvalidArgument(
        "Expecting exactly one '->' in einsum equation: ", equation);
  }
  *output_subscript = std::move(inputs_and_output_subscripts[1]);
  *input_subscripts =
      absl::StrSplit(std::move(inputs_and_output_subscripts[0]), ',');
  if (input_subscripts->size() != 1 && input_subscripts->size() != 2) {
    return errors::InvalidArgument(
        "Expecting 1 or 2 input subscripts in equation '", equation,
        "' but got: ", input_subscripts->size());
  }
  return Status::OK();
}

// Returns the EinsumDimensionType given whether the corresponding label is
// present in exactly one input subscript (is_unique) and whether it is absent
// from the output subscripts (is_removed). Does not handle broadcasting
// dimensions.
EinsumDimensionType GetDimensionType(bool is_removed, bool is_unique) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/util/einsum_op_util.cc", "GetDimensionType");

  if (!is_removed && !is_unique)
    return kBatch;
  else if (!is_removed && is_unique)
    return kFree;
  else if (is_removed && !is_unique)
    return kContract;
  else  // is_removed && is_unique
    return kReduce;
}

// Maps the character labels to consecutive integers.
void MapToLabels(const string& subscript, Labels* labels,
                 absl::flat_hash_map<char, int>* label_mapping) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("subscript: \"" + subscript + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/util/einsum_op_util.cc", "MapToLabels");

  for (int i = 0; i < subscript.size(); ++i) {
    const char label_char = subscript[i];
    if (label_char == '.') {
      labels->push_back(kEllipsisLabel);
      i += 2;  // Skip next 2 characters as well.
      continue;
    }
    if (!label_mapping->contains(label_char)) {
      const int next_label = label_mapping->size();
      (*label_mapping)[label_char] = next_label;
    }
    const int mapped_label = (*label_mapping)[label_char];
    labels->push_back(mapped_label);
  }
}

Status ParseEinsumEquation(const string& equation, OperandLabels* input_labels,
                           Labels* output_labels,
                           std::vector<EinsumDimensionType>* label_types,
                           OperandLabelCounts* input_label_counts,
                           LabelCounts* output_label_counts,
                           gtl::InlinedVector<bool, 2>* input_has_ellipsis,
                           bool* output_has_ellipsis) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("equation: \"" + equation + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSeinsum_op_utilDTcc mht_3(mht_3_v, 269, "", "./tensorflow/core/util/einsum_op_util.cc", "ParseEinsumEquation");

  gtl::InlinedVector<string, 2> input_str;
  string output_str;
  TF_RETURN_IF_ERROR(ValidateEinsumEquation(equation, &input_str, &output_str));

  // Temporary map from single character labels to (consecutive) integer labels.
  absl::flat_hash_map<char, int> label_mapping;
  int num_inputs = input_str.size();
  input_labels->resize(num_inputs);

  // Map from single characters to integer labels.
  for (int i = 0; i < num_inputs; ++i) {
    MapToLabels(input_str[i], &input_labels->at(i), &label_mapping);
  }
  MapToLabels(output_str, output_labels, &label_mapping);

  // Compute counts for input and output labels.
  int num_labels = label_mapping.size();
  input_label_counts->resize(num_inputs);
  input_has_ellipsis->resize(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_label_counts->at(i).resize(num_labels);
    input_has_ellipsis->at(i) = false;
    for (const int label : input_labels->at(i)) {
      if (label != kEllipsisLabel)
        input_label_counts->at(i)[label] += 1;
      else
        input_has_ellipsis->at(i) = true;
    }
  }
  output_label_counts->resize(num_labels);
  *output_has_ellipsis = false;
  for (const int label : *output_labels) {
    if (label != kEllipsisLabel)
      output_label_counts->at(label) += 1;
    else
      *output_has_ellipsis = true;
  }

  // Map each label to a unique EinsumDimensionType.
  label_types->resize(num_labels);
  for (int label = 0; label < num_labels; ++label) {
    if (label == kEllipsisLabel) continue;
    bool removed = (*output_label_counts)[label] == 0;
    bool unique = num_inputs == 1 || (*input_label_counts)[0][label] == 0 ||
                  (*input_label_counts)[1][label] == 0;
    (*label_types)[label] = GetDimensionType(removed, unique);
  }
  return Status::OK();
}

}  // namespace tensorflow
