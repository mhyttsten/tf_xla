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

// Collection of scoring classes that can be extended and provided to the
// CTCBeamSearchDecoder to incorporate additional scoring logic (such as a
// language model).
//
// To build a custom scorer extend and implement the pure virtual methods from
// BeamScorerInterface. The default CTC decoding behavior is implemented
// through BaseBeamScorer.

// Copied from tensorflow/core/util/ctc/ctc_beam_scorer.h
// TODO(b/111524997): Remove this file.
#ifndef TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_SCORER_H_
#define TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_SCORER_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh() {
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


#include "tensorflow/lite/kernels/ctc/ctc_beam_entry.h"

namespace tflite {
namespace custom {
namespace ctc {

// Base implementation of a beam scorer used by default by the decoder that can
// be subclassed and provided as an argument to CTCBeamSearchDecoder, if complex
// scoring is required. Its main purpose is to provide a thin layer for
// integrating language model scoring easily.
template <typename CTCBeamState>
class BaseBeamScorer {
 public:
  virtual ~BaseBeamScorer() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/ctc/ctc_beam_scorer.h", "~BaseBeamScorer");
}
  // State initialization.
  virtual void InitializeState(CTCBeamState* root) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/ctc/ctc_beam_scorer.h", "InitializeState");
}
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  virtual void ExpandState(const CTCBeamState& from_state, int from_label,
                           CTCBeamState* to_state, int to_label) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh mht_2(mht_2_v, 224, "", "./tensorflow/lite/kernels/ctc/ctc_beam_scorer.h", "ExpandState");
}
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  virtual void ExpandStateEnd(CTCBeamState* state) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh mht_3(mht_3_v, 231, "", "./tensorflow/lite/kernels/ctc/ctc_beam_scorer.h", "ExpandStateEnd");
}
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  virtual float GetStateExpansionScore(const CTCBeamState& state,
                                       float previous_score) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh mht_4(mht_4_v, 243, "", "./tensorflow/lite/kernels/ctc/ctc_beam_scorer.h", "GetStateExpansionScore");

    return previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  virtual float GetStateEndExpansionScore(const CTCBeamState& state) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_scorerDTh mht_5(mht_5_v, 254, "", "./tensorflow/lite/kernels/ctc/ctc_beam_scorer.h", "GetStateEndExpansionScore");

    return 0;
  }
};

}  // namespace ctc
}  // namespace custom
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_SCORER_H_
