/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// An abstraction to pick from one of N elements with a specified
// weight per element.
//
// The weight for a given element can be changed in O(lg N) time
// An element can be picked in O(lg N) time.
//
// Uses O(N) bytes of memory.
//
// Alternative: distribution-sampler.h allows O(1) time picking, but no weight
// adjustment after construction.

#ifndef TENSORFLOW_LIB_RANDOM_WEIGHTED_PICKER_H_
#define TENSORFLOW_LIB_RANDOM_WEIGHTED_PICKER_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh() {
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


#include <assert.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

class SimplePhilox;

class WeightedPicker {
 public:
  // REQUIRES   N >= 0
  // Initializes the elements with a weight of one per element
  explicit WeightedPicker(int N);

  // Releases all resources
  ~WeightedPicker();

  // Pick a random element with probability proportional to its weight.
  // If total weight is zero, returns -1.
  int Pick(SimplePhilox* rnd) const;

  // Deterministically pick element x whose weight covers the
  // specified weight_index.
  // Returns -1 if weight_index is not in the range [ 0 .. total_weight()-1 ]
  int PickAt(int32_t weight_index) const;

  // Get the weight associated with an element
  // REQUIRES 0 <= index < N
  int32 get_weight(int index) const;

  // Set the weight associated with an element
  // REQUIRES weight >= 0.0f
  // REQUIRES 0 <= index < N
  void set_weight(int index, int32_t weight);

  // Get the total combined weight of all elements
  int32 total_weight() const;

  // Get the number of elements in the picker
  int num_elements() const;

  // Set weight of each element to "weight"
  void SetAllWeights(int32_t weight);

  // Resizes the picker to N and
  // sets the weight of each element i to weight[i].
  // The sum of the weights should not exceed 2^31 - 2
  // Complexity O(N).
  void SetWeightsFromArray(int N, const int32* weights);

  // REQUIRES   N >= 0
  //
  // Resize the weighted picker so that it has "N" elements.
  // Any newly added entries have zero weight.
  //
  // Note: Resizing to a smaller size than num_elements() will
  // not reclaim any memory.  If you wish to reduce memory usage,
  // allocate a new WeightedPicker of the appropriate size.
  //
  // It is efficient to use repeated calls to Resize(num_elements() + 1)
  // to grow the picker to size X (takes total time O(X)).
  void Resize(int N);

  // Grow the picker by one and set the weight of the new entry to "weight".
  //
  // Repeated calls to Append() in order to grow the
  // picker to size X takes a total time of O(X lg(X)).
  // Consider using SetWeightsFromArray instead.
  void Append(int32_t weight);

 private:
  // We keep a binary tree with N leaves.  The "i"th leaf contains
  // the weight of the "i"th element.  An internal node contains
  // the sum of the weights of its children.
  int N_;           // Number of elements
  int num_levels_;  // Number of levels in tree (level-0 is root)
  int32** level_;   // Array that holds nodes per level

  // Size of each level
  static int LevelSize(int level) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh mht_0(mht_0_v, 281, "", "./tensorflow/core/lib/random/weighted_picker.h", "LevelSize");
 return 1 << level; }

  // Rebuild the tree weights using the leaf weights
  void RebuildTreeWeights();

  TF_DISALLOW_COPY_AND_ASSIGN(WeightedPicker);
};

inline int32 WeightedPicker::get_weight(int index) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh mht_1(mht_1_v, 292, "", "./tensorflow/core/lib/random/weighted_picker.h", "WeightedPicker::get_weight");

  DCHECK_GE(index, 0);
  DCHECK_LT(index, N_);
  return level_[num_levels_ - 1][index];
}

inline int32 WeightedPicker::total_weight() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh mht_2(mht_2_v, 301, "", "./tensorflow/core/lib/random/weighted_picker.h", "WeightedPicker::total_weight");
 return level_[0][0]; }

inline int WeightedPicker::num_elements() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTh mht_3(mht_3_v, 306, "", "./tensorflow/core/lib/random/weighted_picker.h", "WeightedPicker::num_elements");
 return N_; }

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_WEIGHTED_PICKER_H_
