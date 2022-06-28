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

// Copied from tensorflow/core/util/ctc/ctc_decoder.h
// TODO(b/111524997): Remove this file.
#ifndef TENSORFLOW_LITE_KERNELS_CTC_CTC_DECODER_H_
#define TENSORFLOW_LITE_KERNELS_CTC_CTC_DECODER_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh() {
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
#include <vector>

#include "third_party/eigen3/Eigen/Core"

namespace tflite {
namespace custom {
namespace ctc {

// The CTCDecoder is an abstract interface to be implemented when providing a
// decoding method on the timestep output of a RNN trained with CTC loss.
//
// The two types of decoding available are:
//   - greedy path, through the CTCGreedyDecoder
//   - beam search, through the CTCBeamSearchDecoder
class CTCDecoder {
 public:
  typedef Eigen::Map<const Eigen::ArrayXi> SequenceLength;
  typedef Eigen::Map<const Eigen::MatrixXf> Input;
  typedef std::vector<std::vector<int>> Output;
  typedef Eigen::Map<Eigen::MatrixXf> ScoreOutput;

  CTCDecoder(int num_classes, int batch_size, bool merge_repeated)
      : num_classes_(num_classes),
        blank_index_(num_classes - 1),
        batch_size_(batch_size),
        merge_repeated_(merge_repeated) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh mht_0(mht_0_v, 216, "", "./tensorflow/lite/kernels/ctc/ctc_decoder.h", "CTCDecoder");
}

  virtual ~CTCDecoder() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh mht_1(mht_1_v, 221, "", "./tensorflow/lite/kernels/ctc/ctc_decoder.h", "~CTCDecoder");
}

  // Dimensionality of the input/output is expected to be:
  //  - seq_len[b] - b = 0 to batch_size_
  //  - input[t].rows(b) - t = 0 to timesteps; b = 0 t batch_size_
  //  - output.size() specifies the number of beams to be returned.
  //  - scores(b, i) - b = 0 to batch_size; i = 0 to output.size()
  virtual bool Decode(const SequenceLength& seq_len,
                      const std::vector<Input>& input,
                      std::vector<Output>* output, ScoreOutput* scores) = 0;

  int batch_size() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh mht_2(mht_2_v, 235, "", "./tensorflow/lite/kernels/ctc/ctc_decoder.h", "batch_size");
 return batch_size_; }
  int num_classes() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh mht_3(mht_3_v, 239, "", "./tensorflow/lite/kernels/ctc/ctc_decoder.h", "num_classes");
 return num_classes_; }

 protected:
  int num_classes_;
  int blank_index_;
  int batch_size_;
  bool merge_repeated_;
};

// CTCGreedyDecoder is an implementation of the simple best path decoding
// algorithm, selecting at each timestep the most likely class at each timestep.
class CTCGreedyDecoder : public CTCDecoder {
 public:
  CTCGreedyDecoder(int num_classes, int batch_size, bool merge_repeated)
      : CTCDecoder(num_classes, batch_size, merge_repeated) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh mht_4(mht_4_v, 256, "", "./tensorflow/lite/kernels/ctc/ctc_decoder.h", "CTCGreedyDecoder");
}

  bool Decode(const CTCDecoder::SequenceLength& seq_len,
              const std::vector<CTCDecoder::Input>& input,
              std::vector<CTCDecoder::Output>* output,
              CTCDecoder::ScoreOutput* scores) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_decoderDTh mht_5(mht_5_v, 264, "", "./tensorflow/lite/kernels/ctc/ctc_decoder.h", "Decode");

    if (output->empty() || (*output)[0].size() < batch_size_) {
      return false;
    }
    if (scores->rows() < batch_size_ || scores->cols() == 0) {
      return false;
    }
    // For each batch entry, identify the transitions
    for (int b = 0; b < batch_size_; ++b) {
      int seq_len_b = seq_len[b];
      // Only writing to beam 0
      std::vector<int>& output_b = (*output)[0][b];

      int prev_class_ix = -1;
      (*scores)(b, 0) = 0;
      for (int t = 0; t < seq_len_b; ++t) {
        auto row = input[t].row(b);
        int max_class_ix;
        (*scores)(b, 0) += -row.maxCoeff(&max_class_ix);
        if (max_class_ix != blank_index_ &&
            !(merge_repeated_ && max_class_ix == prev_class_ix)) {
          output_b.push_back(max_class_ix);
        }
        prev_class_ix = max_class_ix;
      }
    }
    return true;
  }
};

}  // namespace ctc
}  // namespace custom
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CTC_CTC_DECODER_H_
