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
class MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPStest_streaming_accuracyDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPStest_streaming_accuracyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPStest_streaming_accuracyDTcc() {
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
/*

Tool to create accuracy statistics from running an audio recognition model on a
continuous stream of samples.

This is designed to be an environment for running experiments on new models and
settings to understand the effects they will have in a real application. You
need to supply it with a long audio file containing sounds you want to recognize
and a text file listing the labels of each sound along with the time they occur.
With this information, and a frozen model, the tool will process the audio
stream, apply the model, and keep track of how many mistakes and successes the
model achieved.

The matched percentage is the number of sounds that were correctly classified,
as a percentage of the total number of sounds listed in the ground truth file.
A correct classification is when the right label is chosen within a short time
of the expected ground truth, where the time tolerance is controlled by the
'time_tolerance_ms' command line flag.

The wrong percentage is how many sounds triggered a detection (the classifier
figured out it wasn't silence or background noise), but the detected class was
wrong. This is also a percentage of the total number of ground truth sounds.

The false positive percentage is how many sounds were detected when there was
only silence or background noise. This is also expressed as a percentage of the
total number of ground truth sounds, though since it can be large it may go
above 100%.

The easiest way to get an audio file and labels to test with is by using the
'generate_streaming_test_wav' script. This will synthesize a test file with
randomly placed sounds and background noise, and output a text file with the
ground truth.

If you want to test natural data, you need to use a .wav with the same sample
rate as your model (often 16,000 samples per second), and note down where the
sounds occur in time. Save this information out as a comma-separated text file,
where the first column is the label and the second is the time in seconds from
the start of the file that it occurs.

Here's an example of how to run the tool:

bazel run tensorflow/examples/speech_commands:test_streaming_accuracy -- \
--wav=/tmp/streaming_test_bg.wav \
--graph=/tmp/conv_frozen.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--ground_truth=/tmp/streaming_test_labels.txt --verbose \
--clip_duration_ms=1000 --detection_threshold=0.70 --average_window_ms=500 \
--suppression_ms=500 --time_tolerance_ms=1500

 */

#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/examples/speech_commands/accuracy_utils.h"
#include "tensorflow/examples/speech_commands/recognize_commands.h"

// These are all common classes it's handy to reference with no namespace.
using ::int64_t;
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::uint16;
using tensorflow::uint32;

namespace {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("graph_file_name: \"" + graph_file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPStest_streaming_accuracyDTcc mht_0(mht_0_v, 269, "", "./tensorflow/examples/speech_commands/test_streaming_accuracy.cc", "LoadGraph");

  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPStest_streaming_accuracyDTcc mht_1(mht_1_v, 291, "", "./tensorflow/examples/speech_commands/test_streaming_accuracy.cc", "ReadLabelsFile");

  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file '", file_name,
                                        "' not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  return Status::OK();
}

}  // namespace

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPStest_streaming_accuracyDTcc mht_2(mht_2_v, 310, "", "./tensorflow/examples/speech_commands/test_streaming_accuracy.cc", "main");

  string wav = "";
  string graph = "";
  string labels = "";
  string ground_truth = "";
  string input_data_name = "decoded_sample_data:0";
  string input_rate_name = "decoded_sample_data:1";
  string output_name = "labels_softmax";
  int32_t clip_duration_ms = 1000;
  int32_t clip_stride_ms = 30;
  int32_t average_window_ms = 500;
  int32_t time_tolerance_ms = 750;
  int32_t suppression_ms = 1500;
  float detection_threshold = 0.7f;
  bool verbose = false;
  std::vector<Flag> flag_list = {
      Flag("wav", &wav, "audio file to be identified"),
      Flag("graph", &graph, "model to be executed"),
      Flag("labels", &labels, "path to file containing labels"),
      Flag("ground_truth", &ground_truth,
           "path to file containing correct times and labels of words in the "
           "audio as <word>,<timestamp in ms> lines"),
      Flag("input_data_name", &input_data_name,
           "name of input data node in model"),
      Flag("input_rate_name", &input_rate_name,
           "name of input sample rate node in model"),
      Flag("output_name", &output_name, "name of output node in model"),
      Flag("clip_duration_ms", &clip_duration_ms,
           "length of recognition window"),
      Flag("average_window_ms", &average_window_ms,
           "length of window to smooth results over"),
      Flag("time_tolerance_ms", &time_tolerance_ms,
           "maximum gap allowed between a recognition and ground truth"),
      Flag("suppression_ms", &suppression_ms,
           "how long to ignore others for after a recognition"),
      Flag("clip_stride_ms", &clip_stride_ms, "how often to run recognition"),
      Flag("detection_threshold", &detection_threshold,
           "what score is required to trigger detection of a word"),
      Flag("verbose", &verbose, "whether to log extra debugging information"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  std::vector<string> labels_list;
  Status read_labels_status = ReadLabelsFile(labels, &labels_list);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return -1;
  }

  std::vector<std::pair<string, int64_t>> ground_truth_list;
  Status read_ground_truth_status =
      tensorflow::ReadGroundTruthFile(ground_truth, &ground_truth_list);
  if (!read_ground_truth_status.ok()) {
    LOG(ERROR) << read_ground_truth_status;
    return -1;
  }

  string wav_string;
  Status read_wav_status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), wav, &wav_string);
  if (!read_wav_status.ok()) {
    LOG(ERROR) << read_wav_status;
    return -1;
  }
  std::vector<float> audio_data;
  uint32 sample_count;
  uint16 channel_count;
  uint32 sample_rate;
  Status decode_wav_status = tensorflow::wav::DecodeLin16WaveAsFloatVector(
      wav_string, &audio_data, &sample_count, &channel_count, &sample_rate);
  if (!decode_wav_status.ok()) {
    LOG(ERROR) << decode_wav_status;
    return -1;
  }
  if (channel_count != 1) {
    LOG(ERROR) << "Only mono .wav files can be used, but input has "
               << channel_count << " channels.";
    return -1;
  }

  const int64_t clip_duration_samples = (clip_duration_ms * sample_rate) / 1000;
  const int64_t clip_stride_samples = (clip_stride_ms * sample_rate) / 1000;
  Tensor audio_data_tensor(tensorflow::DT_FLOAT,
                           tensorflow::TensorShape({clip_duration_samples, 1}));

  Tensor sample_rate_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  sample_rate_tensor.scalar<int32>()() = sample_rate;

  tensorflow::RecognizeCommands recognize_commands(
      labels_list, average_window_ms, detection_threshold, suppression_ms);

  std::vector<std::pair<string, int64_t>> all_found_words;
  tensorflow::StreamingAccuracyStats previous_stats;

  const int64_t audio_data_end = (sample_count - clip_duration_samples);
  for (int64_t audio_data_offset = 0; audio_data_offset < audio_data_end;
       audio_data_offset += clip_stride_samples) {
    const float* input_start = &(audio_data[audio_data_offset]);
    const float* input_end = input_start + clip_duration_samples;
    std::copy(input_start, input_end, audio_data_tensor.flat<float>().data());

    // Actually run the audio through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_data_name, audio_data_tensor},
                                      {input_rate_name, sample_rate_tensor}},
                                     {output_name}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }

    const int64_t current_time_ms = (audio_data_offset * 1000) / sample_rate;
    string found_command;
    float score;
    bool is_new_command;
    Status recognize_status = recognize_commands.ProcessLatestResults(
        outputs[0], current_time_ms, &found_command, &score, &is_new_command);
    if (!recognize_status.ok()) {
      LOG(ERROR) << "Recognition processing failed: " << recognize_status;
      return -1;
    }

    if (is_new_command && (found_command != "_silence_")) {
      all_found_words.push_back({found_command, current_time_ms});
      if (verbose) {
        tensorflow::StreamingAccuracyStats stats;
        tensorflow::CalculateAccuracyStats(ground_truth_list, all_found_words,
                                           current_time_ms, time_tolerance_ms,
                                           &stats);
        int32_t false_positive_delta = stats.how_many_false_positives -
                                       previous_stats.how_many_false_positives;
        int32_t correct_delta = stats.how_many_correct_words -
                                previous_stats.how_many_correct_words;
        int32_t wrong_delta =
            stats.how_many_wrong_words - previous_stats.how_many_wrong_words;
        string recognition_state;
        if (false_positive_delta == 1) {
          recognition_state = " (False Positive)";
        } else if (correct_delta == 1) {
          recognition_state = " (Correct)";
        } else if (wrong_delta == 1) {
          recognition_state = " (Wrong)";
        } else {
          LOG(ERROR) << "Unexpected state in statistics";
        }
        LOG(INFO) << current_time_ms << "ms: " << found_command << ": " << score
                  << recognition_state;
        previous_stats = stats;
        tensorflow::PrintAccuracyStats(stats);
      }
    }
  }

  tensorflow::StreamingAccuracyStats stats;
  tensorflow::CalculateAccuracyStats(ground_truth_list, all_found_words, -1,
                                     time_tolerance_ms, &stats);
  tensorflow::PrintAccuracyStats(stats);

  return 0;
}
