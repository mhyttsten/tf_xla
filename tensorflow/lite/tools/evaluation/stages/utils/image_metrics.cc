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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h"

#include <algorithm>
#include <cmath>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"

namespace tflite {
namespace evaluation {
namespace image {

float Box2D::Length(const Box2D::Interval& a) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::Length");

  return std::max(0.f, a.max - a.min);
}

float Box2D::Intersection(const Box2D::Interval& a, const Box2D::Interval& b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::Intersection");

  return Length(Interval{std::max(a.min, b.min), std::min(a.max, b.max)});
}

float Box2D::Area() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_2(mht_2_v, 210, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::Area");
 return Length(x) * Length(y); }

float Box2D::Intersection(const Box2D& other) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_3(mht_3_v, 215, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::Intersection");

  return Intersection(x, other.x) * Intersection(y, other.y);
}

float Box2D::Union(const Box2D& other) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_4(mht_4_v, 222, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::Union");

  return Area() + other.Area() - Intersection(other);
}

float Box2D::IoU(const Box2D& other) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_5(mht_5_v, 229, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::IoU");

  const float total = Union(other);
  if (total > 0) {
    return Intersection(other) / total;
  } else {
    return 0.0;
  }
}

float Box2D::Overlap(const Box2D& other) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_6(mht_6_v, 241, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "Box2D::Overlap");

  const float intersection = Intersection(other);
  return intersection > 0 ? intersection / Area() : 0.0;
}

float AveragePrecision::FromPRCurve(const std::vector<PR>& pr,
                                    std::vector<PR>* pr_out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_7(mht_7_v, 250, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "AveragePrecision::FromPRCurve");

  // Because pr[...] are ordered by recall, iterate backward to compute max
  // precision. p(r) = max_{r' >= r} p(r') for r in 0.0, 0.1, 0.2, ..., 0.9,
  // 1.0. Then, take the average of (num_recal_points) quantities.
  float p = 0;
  float sum = 0;
  int r_level = opts_.num_recall_points;
  for (int i = pr.size() - 1; i >= 0; --i) {
    const PR& item = pr[i];
    if (i > 0) {
      if (item.r < pr[i - 1].r) {
        LOG(ERROR) << "recall points are not in order: " << pr[i - 1].r << ", "
                   << item.r;
        return 0;
      }
    }

    // Because r takes values opts_.num_recall_points, opts_.num_recall_points -
    // 1, ..., 0, the following condition is checking whether item.r crosses r /
    // opts_.num_recall_points. I.e., 1.0, 0.90, ..., 0.01, 0.0.  We don't use
    // float to represent r because 0.01 is not representable precisely.
    while (item.r * opts_.num_recall_points < r_level) {
      const float recall =
          static_cast<float>(r_level) / opts_.num_recall_points;
      if (r_level < 0) {
        LOG(ERROR) << "Number of recall points should be > 0";
        return 0;
      }
      sum += p;
      r_level -= 1;
      if (pr_out != nullptr) {
        pr_out->emplace_back(p, recall);
      }
    }
    p = std::max(p, item.p);
  }
  for (; r_level >= 0; --r_level) {
    const float recall = static_cast<float>(r_level) / opts_.num_recall_points;
    sum += p;
    if (pr_out != nullptr) {
      pr_out->emplace_back(p, recall);
    }
  }
  return sum / (1 + opts_.num_recall_points);
}

float AveragePrecision::FromBoxes(const std::vector<Detection>& groundtruth,
                                  const std::vector<Detection>& prediction,
                                  std::vector<PR>* pr_out) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTcc mht_8(mht_8_v, 301, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc", "AveragePrecision::FromBoxes");

  // Index ground truth boxes based on imageid.
  absl::flat_hash_map<int64_t, std::list<Detection>> gt;
  int num_gt = 0;
  for (auto& box : groundtruth) {
    gt[box.imgid].push_back(box);
    if (!box.difficult && box.ignore == kDontIgnore) {
      ++num_gt;
    }
  }

  if (num_gt == 0) {
    return NAN;
  }

  // Sort all predicted boxes by their scores in a non-ascending order.
  std::vector<Detection> pd = prediction;
  std::sort(pd.begin(), pd.end(), [](const Detection& a, const Detection& b) {
    return a.score > b.score;
  });

  // Computes p-r for every prediction.
  std::vector<PR> pr;
  int correct = 0;
  int num_pd = 0;
  for (int i = 0; i < pd.size(); ++i) {
    const Detection& b = pd[i];
    auto* g = &gt[b.imgid];
    auto best = g->end();
    float best_iou = -INFINITY;
    for (auto it = g->begin(); it != g->end(); ++it) {
      const auto iou = b.box.IoU(it->box);
      if (iou > best_iou) {
        best = it;
        best_iou = iou;
      }
    }
    if ((best != g->end()) && (best_iou >= opts_.iou_threshold)) {
      if (best->difficult) {
        continue;
      }
      switch (best->ignore) {
        case kDontIgnore: {
          ++correct;
          ++num_pd;
          g->erase(best);
          pr.push_back({static_cast<float>(correct) / num_pd,
                        static_cast<float>(correct) / num_gt});
          break;
        }
        case kIgnoreOneMatch: {
          g->erase(best);
          break;
        }
        case kIgnoreAllMatches: {
          break;
        }
      }
    } else {
      ++num_pd;
      pr.push_back({static_cast<float>(correct) / num_pd,
                    static_cast<float>(correct) / num_gt});
    }
  }
  return FromPRCurve(pr, pr_out);
}

}  // namespace image
}  // namespace evaluation
}  // namespace tflite
