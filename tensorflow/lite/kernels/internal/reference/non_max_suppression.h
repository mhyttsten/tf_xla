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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_NON_MAX_SUPPRESSION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_NON_MAX_SUPPRESSION_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSnon_max_suppressionDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSnon_max_suppressionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSnon_max_suppressionDTh() {
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


#include <algorithm>
#include <cmath>
#include <deque>
#include <queue>

namespace tflite {
namespace reference_ops {

// A pair of diagonal corners of the box.
struct BoxCornerEncoding {
  float y1;
  float x1;
  float y2;
  float x2;
};

inline float ComputeIntersectionOverUnion(const float* boxes, const int i,
                                          const int j) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSnon_max_suppressionDTh mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/internal/reference/non_max_suppression.h", "ComputeIntersectionOverUnion");

  auto& box_i = reinterpret_cast<const BoxCornerEncoding*>(boxes)[i];
  auto& box_j = reinterpret_cast<const BoxCornerEncoding*>(boxes)[j];
  const float box_i_y_min = std::min<float>(box_i.y1, box_i.y2);
  const float box_i_y_max = std::max<float>(box_i.y1, box_i.y2);
  const float box_i_x_min = std::min<float>(box_i.x1, box_i.x2);
  const float box_i_x_max = std::max<float>(box_i.x1, box_i.x2);
  const float box_j_y_min = std::min<float>(box_j.y1, box_j.y2);
  const float box_j_y_max = std::max<float>(box_j.y1, box_j.y2);
  const float box_j_x_min = std::min<float>(box_j.x1, box_j.x2);
  const float box_j_x_max = std::max<float>(box_j.x1, box_j.x2);

  const float area_i =
      (box_i_y_max - box_i_y_min) * (box_i_x_max - box_i_x_min);
  const float area_j =
      (box_j_y_max - box_j_y_min) * (box_j_x_max - box_j_x_min);
  if (area_i <= 0 || area_j <= 0) return 0.0;
  const float intersection_ymax = std::min<float>(box_i_y_max, box_j_y_max);
  const float intersection_xmax = std::min<float>(box_i_x_max, box_j_x_max);
  const float intersection_ymin = std::max<float>(box_i_y_min, box_j_y_min);
  const float intersection_xmin = std::max<float>(box_i_x_min, box_j_x_min);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

// Implements (Single-Class) Soft NMS (with Gaussian weighting).
// Supports functionality of TensorFlow ops NonMaxSuppressionV4 & V5.
// Reference: "Soft-NMS - Improving Object Detection With One Line of Code"
//            [Bodla et al, https://arxiv.org/abs/1704.04503]
// Implementation adapted from the TensorFlow NMS code at
// tensorflow/core/kernels/non_max_suppression_op.cc.
//
// Arguments:
//  boxes: box encodings in format [y1, x1, y2, x2], shape: [num_boxes, 4]
//  num_boxes: number of candidates
//  scores: scores for candidate boxes, in the same order. shape: [num_boxes]
//  max_output_size: the maximum number of selections.
//  iou_threshold: Intersection-over-Union (IoU) threshold for NMS
//  score_threshold: All candidate scores below this value are rejected
//  soft_nms_sigma: Soft NMS parameter, used for decaying scores
//
// Outputs:
//  selected_indices: all the selected indices. Underlying array must have
//    length >= max_output_size. Cannot be null.
//  selected_scores: scores of selected indices. Defer from original value for
//    Soft NMS. If not null, array must have length >= max_output_size.
//  num_selected_indices: Number of selections. Only these many elements are
//    set in selected_indices, selected_scores. Cannot be null.
//
// Assumes inputs are valid (for eg, iou_threshold must be >= 0).
inline void NonMaxSuppression(const float* boxes, const int num_boxes,
                              const float* scores, const int max_output_size,
                              const float iou_threshold,
                              const float score_threshold,
                              const float soft_nms_sigma, int* selected_indices,
                              float* selected_scores,
                              int* num_selected_indices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSnon_max_suppressionDTh mht_1(mht_1_v, 265, "", "./tensorflow/lite/kernels/internal/reference/non_max_suppression.h", "NonMaxSuppression");

  struct Candidate {
    int index;
    float score;
    int suppress_begin_index;
  };

  // Priority queue to hold candidates.
  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSnon_max_suppressionDTh mht_2(mht_2_v, 276, "", "./tensorflow/lite/kernels/internal/reference/non_max_suppression.h", "lambda");

    return bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  // Populate queue with candidates above the score threshold.
  for (int i = 0; i < num_boxes; ++i) {
    if (scores[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, scores[i], 0}));
    }
  }

  *num_selected_indices = 0;
  int num_outputs = std::min(static_cast<int>(candidate_priority_queue.size()),
                             max_output_size);
  if (num_outputs == 0) return;

  // NMS loop.
  float scale = 0;
  if (soft_nms_sigma > 0.0) {
    scale = -0.5 / soft_nms_sigma;
  }
  while (*num_selected_indices < num_outputs &&
         !candidate_priority_queue.empty()) {
    Candidate next_candidate = candidate_priority_queue.top();
    const float original_score = next_candidate.score;
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if `next_candidate` should be suppressed. We also enforce a property
    // that a candidate can be suppressed by another candidate no more than
    // once via `suppress_begin_index` which tracks which previously selected
    // boxes have already been compared against next_candidate prior to a given
    // iteration.  These previous selected boxes are then skipped over in the
    // following loop.
    bool should_hard_suppress = false;
    for (int j = *num_selected_indices - 1;
         j >= next_candidate.suppress_begin_index; --j) {
      const float iou = ComputeIntersectionOverUnion(
          boxes, next_candidate.index, selected_indices[j]);

      // First decide whether to perform hard suppression.
      if (iou >= iou_threshold) {
        should_hard_suppress = true;
        break;
      }

      // Suppress score if NMS sigma > 0.
      if (soft_nms_sigma > 0.0) {
        next_candidate.score =
            next_candidate.score * std::exp(scale * iou * iou);
      }

      // If score has fallen below score_threshold, it won't be pushed back into
      // the queue.
      if (next_candidate.score <= score_threshold) break;
    }
    // If `next_candidate.score` has not dropped below `score_threshold`
    // by this point, then we know that we went through all of the previous
    // selections and can safely update `suppress_begin_index` to
    // `selected.size()`. If on the other hand `next_candidate.score`
    // *has* dropped below the score threshold, then since `suppress_weight`
    // always returns values in [0, 1], further suppression by items that were
    // not covered in the above for loop would not have caused the algorithm
    // to select this item. We thus do the same update to
    // `suppress_begin_index`, but really, this element will not be added back
    // into the priority queue.
    next_candidate.suppress_begin_index = *num_selected_indices;

    if (!should_hard_suppress) {
      if (next_candidate.score == original_score) {
        // Suppression has not occurred, so select next_candidate.
        selected_indices[*num_selected_indices] = next_candidate.index;
        if (selected_scores) {
          selected_scores[*num_selected_indices] = next_candidate.score;
        }
        ++*num_selected_indices;
      }
      if (next_candidate.score > score_threshold) {
        // Soft suppression might have occurred and current score is still
        // greater than score_threshold; add next_candidate back onto priority
        // queue.
        candidate_priority_queue.push(next_candidate);
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_NON_MAX_SUPPRESSION_H_
