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
class MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This test illustrates how to make use of the CTCBeamSearchDecoder using a
// custom BeamScorer and BeamState based on a dictionary with a few artificial
// words.
#include "tensorflow/core/util/ctc/ctc_beam_search.h"

#include <cmath>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace {

template <class T>
using TestData = std::vector<std::vector<std::vector<T>>>;

// The HistoryBeamState is used to keep track of the current candidate and
// caches the expansion score (needed by the scorer).
template <class T>
struct HistoryBeamState {
  T score;
  std::vector<int> labels;
};

// DictionaryBeamScorer essentially favors candidates that can still become
// dictionary words. As soon as a beam candidate is not a dictionary word or
// a prefix of a dictionary word it gets a low probability at each step.
//
// The dictionary itself is hard-coded a static const variable of the class.
template <class T, class BeamState>
class DictionaryBeamScorer
    : public tensorflow::ctc::BaseBeamScorer<T, BeamState> {
 public:
  DictionaryBeamScorer()
      : tensorflow::ctc::BaseBeamScorer<T, BeamState>(),
        dictionary_({{3}, {3, 1}}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "DictionaryBeamScorer");
}

  void InitializeState(BeamState* root) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "InitializeState");
 root->score = 0; }

  void ExpandState(const BeamState& from_state, int from_label,
                   BeamState* to_state, int to_label) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ExpandState");

    // Keep track of the current complete candidate by storing the labels along
    // the expansion path in the beam state.
    to_state->labels.push_back(to_label);
    SetStateScoreAccordingToDict(to_state);
  }

  void ExpandStateEnd(BeamState* state) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ExpandStateEnd");

    SetStateScoreAccordingToDict(state);
  }

  T GetStateExpansionScore(const BeamState& state,
                           T previous_score) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_4(mht_4_v, 248, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "GetStateExpansionScore");

    return previous_score + state.score;
  }

  T GetStateEndExpansionScore(const BeamState& state) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "GetStateEndExpansionScore");

    return state.score;
  }

  // Simple dictionary used when scoring the beams to check if they are prefixes
  // of dictionary words (see SetStateScoreAccordingToDict below).
  const std::vector<std::vector<int>> dictionary_;

 private:
  void SetStateScoreAccordingToDict(BeamState* state) const;
};

template <class T, class BeamState>
void DictionaryBeamScorer<T, BeamState>::SetStateScoreAccordingToDict(
    BeamState* state) const {
  // Check if the beam can still be a dictionary word (e.g. prefix of one).
  const std::vector<int>& candidate = state->labels;
  for (int w = 0; w < dictionary_.size(); ++w) {
    const std::vector<int>& word = dictionary_[w];
    // If the length of the current beam is already larger, skip.
    if (candidate.size() > word.size()) {
      continue;
    }
    if (std::equal(word.begin(), word.begin() + candidate.size(),
                   candidate.begin())) {
      state->score = std::log(T(1.0));
      return;
    }
  }
  // At this point, the candidate certainly can't be in the dictionary.
  state->score = std::log(T(0.01));
}

template <class T>
void ctc_beam_search_decoding_with_and_without_dictionary() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_6(mht_6_v, 292, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ctc_beam_search_decoding_with_and_without_dictionary");

  const int batch_size = 1;
  const int timesteps = 5;
  const int top_paths = 3;
  const int num_classes = 6;

  // Plain decoder using hibernating beam search algorithm.
  typename tensorflow::ctc::CTCBeamSearchDecoder<T>::DefaultBeamScorer
      default_scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T> decoder(num_classes, 10 * top_paths,
                                                   &default_scorer);

  // Dictionary decoder, allowing only two dictionary words : {3}, {3, 1}.
  DictionaryBeamScorer<T, HistoryBeamState<T>> dictionary_scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T, HistoryBeamState<T>>
      dictionary_decoder(num_classes, top_paths, &dictionary_scorer);

  // Raw data containers (arrays of floats64, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  T input_data_mat[timesteps][batch_size][num_classes] = {
      {{0, 0.6, 0, 0.4, 0, 0}},
      {{0, 0.5, 0, 0.5, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}},
      {{0, 0.4, 0, 0.6, 0, 0}}};

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < timesteps; ++t) {
    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < num_classes; ++c) {
        input_data_mat[t][b][c] = std::log(input_data_mat[t][b][c]);
      }
    }
  }

  // Plain output, without any additional scoring.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> expected_output =
      {
          {{1, 3}, {1, 3, 1}, {3, 1, 3}},
      };

  // Dictionary outputs: preference for dictionary candidates. The
  // second-candidate is there, despite it not being a dictionary word, due to
  // stronger probability in the input to the decoder.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_dict_output = {
          {{3}, {1, 3}, {3, 1}},
      };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>
      inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : outputs) {
    output.resize(batch_size);
  }
  T score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> scores(
      &score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output[0][path]);
  }

  // Prepare dictionary outputs.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> dict_outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : dict_outputs) {
    output.resize(batch_size);
  }
  EXPECT_TRUE(
      dictionary_decoder.Decode(seq_len, inputs, &dict_outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(dict_outputs[path][0], expected_dict_output[0][path]);
  }
}

template <class T>
void ctc_beam_search_decoding_all_beam_elements_have_finite_scores() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_7(mht_7_v, 385, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ctc_beam_search_decoding_all_beam_elements_have_finite_scores");

  const int batch_size = 1;
  const int timesteps = 1;
  const int top_paths = 3;
  const int num_classes = 6;

  // Plain decoder using hibernating beam search algorithm.
  typename tensorflow::ctc::CTCBeamSearchDecoder<T>::DefaultBeamScorer
      default_scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T> decoder(num_classes, top_paths,
                                                   &default_scorer);

  // Raw data containers (arrays of floats64, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  T input_data_mat[timesteps][batch_size][num_classes] = {
      {{0.4, 0.3, 0, 0, 0, 0.5}}};

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>
      inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : outputs) {
    output.resize(batch_size);
  }
  T score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> scores(
      &score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  // Make sure all scores are finite.
  for (int path = 0; path < top_paths; ++path) {
    LOG(INFO) << "path " << path;
    EXPECT_FALSE(std::isinf(score[0][path]));
  }
}

// A beam decoder to test label selection. It simply models N labels with
// rapidly dropping off log-probability.

typedef int LabelState;  // The state is simply the final label.

template <class T>
class RapidlyDroppingLabelScorer
    : public tensorflow::ctc::BaseBeamScorer<T, LabelState> {
 public:
  void InitializeState(LabelState* root) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_8(mht_8_v, 444, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "InitializeState");
}

  void ExpandState(const LabelState& from_state, int from_label,
                   LabelState* to_state, int to_label) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_9(mht_9_v, 450, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ExpandState");

    *to_state = to_label;
  }

  void ExpandStateEnd(LabelState* state) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_10(mht_10_v, 457, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ExpandStateEnd");
}

  T GetStateExpansionScore(const LabelState& state,
                           T previous_score) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_11(mht_11_v, 463, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "GetStateExpansionScore");

    // Drop off rapidly for later labels.
    const T kRapidly = 100;
    return previous_score - kRapidly * state;
  }

  T GetStateEndExpansionScore(const LabelState& state) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_12(mht_12_v, 472, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "GetStateEndExpansionScore");

    return T(0);
  }
};

template <class T>
void ctc_beam_search_label_selection() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_search_testDTcc mht_13(mht_13_v, 481, "", "./tensorflow/core/util/ctc/ctc_beam_search_test.cc", "ctc_beam_search_label_selection");

  const int batch_size = 1;
  const int timesteps = 3;
  const int top_paths = 5;
  const int num_classes = 6;

  // Decoder which drops off log-probabilities for labels 0 >> 1 >> 2 >> 3.
  RapidlyDroppingLabelScorer<T> scorer;
  tensorflow::ctc::CTCBeamSearchDecoder<T, LabelState> decoder(
      num_classes, top_paths, &scorer);

  // Raw data containers (arrays of floats64, ints, etc.).
  int sequence_lengths[batch_size] = {timesteps};
  // Log probabilities, slightly preferring later labels, this decision
  // should be overridden by the scorer which strongly prefers earlier labels.
  // The last one is empty label, and for simplicity  we give it an extremely
  // high cost to ignore it. We also use the first label to break up the
  // repeated label sequence.
  T input_data_mat[timesteps][batch_size][num_classes] = {
      {{-1e6, 1, 2, 3, 4, -1e6}},
      {{1e6, 0, 0, 0, 0, -1e6}},  // force label 0 to break up repeated
      {{-1e6, 1.1, 2.2, 3.3, 4.4, -1e6}},
  };

  // Expected output without label selection
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_default_output = {
          {{1, 0, 1}, {1, 0, 2}, {2, 0, 1}, {1, 0, 3}, {2, 0, 2}},
      };

  // Expected output with label selection limiting to 2 items
  // this is suboptimal because only labels 3 and 4 were allowed to be seen.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_output_size2 = {
          {{3, 0, 3}, {3, 0, 4}, {4, 0, 3}, {4, 0, 4}, {3}},
      };

  // Expected output with label width of 2.0. This would permit three labels at
  // the first timestep, but only two at the last.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output>
      expected_output_width2 = {
          {{2, 0, 3}, {2, 0, 4}, {3, 0, 3}, {3, 0, 4}, {4, 0, 3}},
      };

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], batch_size);
  std::vector<
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>>
      inputs;
  inputs.reserve(timesteps);
  for (int t = 0; t < timesteps; ++t) {
    inputs.emplace_back(&input_data_mat[t][0][0], batch_size, num_classes);
  }

  // Prepare containers for output and scores.
  std::vector<typename tensorflow::ctc::CTCDecoder<T>::Output> outputs(
      top_paths);
  for (typename tensorflow::ctc::CTCDecoder<T>::Output& output : outputs) {
    output.resize(batch_size);
  }
  T score[batch_size][top_paths] = {{0.0}};
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> scores(
      &score[0][0], batch_size, top_paths);

  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_default_output[0][path]);
  }

  // Try label selection size 2
  decoder.SetLabelSelectionParameters(2, T(-1));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output_size2[0][path]);
  }

  // Try label selection width 2.0
  decoder.SetLabelSelectionParameters(0, T(2.0));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output_width2[0][path]);
  }

  // Try both size 2 and width 2.0: the former is more constraining, so
  // it's equivalent to that.
  decoder.SetLabelSelectionParameters(2, T(2.0));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_output_size2[0][path]);
  }

  // Size 4 and width > 3.3 are equivalent to no label selection
  decoder.SetLabelSelectionParameters(4, T(3.3001));
  EXPECT_TRUE(decoder.Decode(seq_len, inputs, &outputs, &scores).ok());
  for (int path = 0; path < top_paths; ++path) {
    EXPECT_EQ(outputs[path][0], expected_default_output[0][path]);
  }
}

TEST(CtcBeamSearch, FloatDecodingWithAndWithoutDictionary) {
  ctc_beam_search_decoding_with_and_without_dictionary<float>();
}

TEST(CtcBeamSearch, DoubleDecodingWithAndWithoutDictionary) {
  ctc_beam_search_decoding_with_and_without_dictionary<double>();
}

TEST(CtcBeamSearch, FloatAllBeamElementsHaveFiniteScores) {
  ctc_beam_search_decoding_all_beam_elements_have_finite_scores<float>();
}

TEST(CtcBeamSearch, DoubleAllBeamElementsHaveFiniteScores) {
  ctc_beam_search_decoding_all_beam_elements_have_finite_scores<double>();
}

TEST(CtcBeamSearch, FloatLabelSelection) {
  ctc_beam_search_label_selection<float>();
}

TEST(CtcBeamSearch, DoubleLabelSelection) {
  ctc_beam_search_label_selection<double>();
}

}  // namespace
