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
class MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc() {
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

#include "tensorflow/core/lib/random/weighted_picker.h"

#include <string.h>
#include <algorithm>

#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace random {

WeightedPicker::WeightedPicker(int N) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::WeightedPicker");

  CHECK_GE(N, 0);
  N_ = N;

  // Find the number of levels
  num_levels_ = 1;
  while (LevelSize(num_levels_ - 1) < N) {
    num_levels_++;
  }

  // Initialize the levels
  level_ = new int32*[num_levels_];
  for (int l = 0; l < num_levels_; l++) {
    level_[l] = new int32[LevelSize(l)];
  }

  SetAllWeights(1);
}

WeightedPicker::~WeightedPicker() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::~WeightedPicker");

  for (int l = 0; l < num_levels_; l++) {
    delete[] level_[l];
  }
  delete[] level_;
}

static int32 UnbiasedUniform(SimplePhilox* r, int32_t n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/lib/random/weighted_picker.cc", "UnbiasedUniform");

  CHECK_LE(0, n);
  const uint32 range = ~static_cast<uint32>(0);
  if (n == 0) {
    return r->Rand32() * n;
  } else if (0 == (n & (n - 1))) {
    // N is a power of two, so just mask off the lower bits.
    return r->Rand32() & (n - 1);
  } else {
    // Reject all numbers that skew the distribution towards 0.

    // Rand32's output is uniform in the half-open interval [0, 2^{32}).
    // For any interval [m,n), the number of elements in it is n-m.

    uint32 rem = (range % n) + 1;
    uint32 rnd;

    // rem = ((2^{32}-1) \bmod n) + 1
    // 1 <= rem <= n

    // NB: rem == n is impossible, since n is not a power of 2 (from
    // earlier check).

    do {
      rnd = r->Rand32();  // rnd uniform over [0, 2^{32})
    } while (rnd < rem);  // reject [0, rem)
    // rnd is uniform over [rem, 2^{32})
    //
    // The number of elements in the half-open interval is
    //
    //  2^{32} - rem = 2^{32} - ((2^{32}-1) \bmod n) - 1
    //               = 2^{32}-1 - ((2^{32}-1) \bmod n)
    //               = n \cdot \lfloor (2^{32}-1)/n \rfloor
    //
    // therefore n evenly divides the number of integers in the
    // interval.
    //
    // The function v \rightarrow v % n takes values from [bias,
    // 2^{32}) to [0, n).  Each integer in the range interval [0, n)
    // will have exactly \lfloor (2^{32}-1)/n \rfloor preimages from
    // the domain interval.
    //
    // Therefore, v % n is uniform over [0, n).  QED.

    return rnd % n;
  }
}

int WeightedPicker::Pick(SimplePhilox* rnd) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_3(mht_3_v, 278, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::Pick");

  if (total_weight() == 0) return -1;

  // using unbiased uniform distribution to avoid bias
  // toward low elements resulting from a possible use
  // of big weights.
  return PickAt(UnbiasedUniform(rnd, total_weight()));
}

int WeightedPicker::PickAt(int32_t weight_index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_4(mht_4_v, 290, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::PickAt");

  if (weight_index < 0 || weight_index >= total_weight()) return -1;

  int32_t position = weight_index;
  int index = 0;

  for (int l = 1; l < num_levels_; l++) {
    // Pick left or right child of "level_[l-1][index]"
    const int32_t left_weight = level_[l][2 * index];
    if (position < left_weight) {
      // Descend to left child
      index = 2 * index;
    } else {
      // Descend to right child
      index = 2 * index + 1;
      position -= left_weight;
    }
  }
  CHECK_GE(index, 0);
  CHECK_LT(index, N_);
  CHECK_LE(position, level_[num_levels_ - 1][index]);
  return index;
}

void WeightedPicker::set_weight(int index, int32_t weight) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_5(mht_5_v, 317, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::set_weight");

  assert(index >= 0);
  assert(index < N_);

  // Adjust the sums all the way up to the root
  const int32_t delta = weight - get_weight(index);
  for (int l = num_levels_ - 1; l >= 0; l--) {
    level_[l][index] += delta;
    index >>= 1;
  }
}

void WeightedPicker::SetAllWeights(int32_t weight) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_6(mht_6_v, 332, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::SetAllWeights");

  // Initialize leaves
  int32* leaves = level_[num_levels_ - 1];
  for (int i = 0; i < N_; i++) leaves[i] = weight;
  for (int i = N_; i < LevelSize(num_levels_ - 1); i++) leaves[i] = 0;

  // Now sum up towards the root
  RebuildTreeWeights();
}

void WeightedPicker::SetWeightsFromArray(int N, const int32* weights) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_7(mht_7_v, 345, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::SetWeightsFromArray");

  Resize(N);

  // Initialize leaves
  int32* leaves = level_[num_levels_ - 1];
  for (int i = 0; i < N_; i++) leaves[i] = weights[i];
  for (int i = N_; i < LevelSize(num_levels_ - 1); i++) leaves[i] = 0;

  // Now sum up towards the root
  RebuildTreeWeights();
}

void WeightedPicker::RebuildTreeWeights() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_8(mht_8_v, 360, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::RebuildTreeWeights");

  for (int l = num_levels_ - 2; l >= 0; l--) {
    int32* level = level_[l];
    int32* children = level_[l + 1];
    for (int i = 0; i < LevelSize(l); i++) {
      level[i] = children[2 * i] + children[2 * i + 1];
    }
  }
}

void WeightedPicker::Append(int32_t weight) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_9(mht_9_v, 373, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::Append");

  Resize(num_elements() + 1);
  set_weight(num_elements() - 1, weight);
}

void WeightedPicker::Resize(int new_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSrandomPSweighted_pickerDTcc mht_10(mht_10_v, 381, "", "./tensorflow/core/lib/random/weighted_picker.cc", "WeightedPicker::Resize");

  CHECK_GE(new_size, 0);
  if (new_size <= LevelSize(num_levels_ - 1)) {
    // The new picker fits in the existing levels.

    // First zero out any of the weights that are being dropped so
    // that the levels are correct (only needed when shrinking)
    for (int i = new_size; i < N_; i++) {
      set_weight(i, 0);
    }

    // We do not need to set any new weights when enlarging because
    // the unneeded entries always have weight zero.
    N_ = new_size;
    return;
  }

  // We follow the simple strategy of just copying the old
  // WeightedPicker into a new WeightedPicker.  The cost is
  // O(N) regardless.
  assert(new_size > N_);
  WeightedPicker new_picker(new_size);
  int32* dst = new_picker.level_[new_picker.num_levels_ - 1];
  int32* src = this->level_[this->num_levels_ - 1];
  memcpy(dst, src, sizeof(dst[0]) * N_);
  memset(dst + N_, 0, sizeof(dst[0]) * (new_size - N_));
  new_picker.RebuildTreeWeights();

  // Now swap the two pickers
  std::swap(new_picker.N_, this->N_);
  std::swap(new_picker.num_levels_, this->num_levels_);
  std::swap(new_picker.level_, this->level_);
  assert(this->N_ == new_size);
}

}  // namespace random
}  // namespace tensorflow
