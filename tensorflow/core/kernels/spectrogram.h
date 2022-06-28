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

// Class for generating spectrogram slices from a waveform.
// Initialize() should be called before calls to other functions.  Once
// Initialize() has been called and returned true, The Compute*() functions can
// be called repeatedly with sequential input data (ie. the first element of the
// next input vector directly follows the last element of the previous input
// vector). Whenever enough audio samples are buffered to produce a
// new frame, it will be placed in output. Output is cleared on each
// call to Compute*(). This class is thread-unsafe, and should only be
// called from one thread at a time.
// With the default parameters, the output of this class should be very
// close to the results of the following MATLAB code:
// overlap_samples = window_length_samples - step_samples;
// window = hann(window_length_samples, 'periodic');
// S = abs(spectrogram(audio, window, overlap_samples)).^2;

#ifndef TENSORFLOW_CORE_KERNELS_SPECTROGRAM_H_
#define TENSORFLOW_CORE_KERNELS_SPECTROGRAM_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh() {
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


#include <complex>
#include <deque>
#include <vector>

#include "third_party/fft2d/fft.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class Spectrogram {
 public:
  Spectrogram() : initialized_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/spectrogram.h", "Spectrogram");
}
  ~Spectrogram() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/spectrogram.h", "~Spectrogram");
}

  // Initializes the class with a given window length and step length
  // (both in samples). Internally a Hann window is used as the window
  // function. Returns true on success, after which calls to Process()
  // are possible. window_length must be greater than 1 and step
  // length must be greater than 0.
  bool Initialize(int window_length, int step_length);

  // Initialize with an explicit window instead of a length.
  bool Initialize(const std::vector<double>& window, int step_length);

  // Reset internal variables.
  // Spectrogram keeps internal state: remaining input data from previous call.
  // As a result it can produce different number of frames when you call
  // ComputeComplexSpectrogram multiple times (even though input data
  // has the same size). As it is shown in
  // MultipleCallsToComputeComplexSpectrogramMayYieldDifferentNumbersOfFrames
  // in tensorflow/core/kernels/spectrogram_test.cc.
  // But if you need to compute Spectrogram on input data without keeping
  // internal state (and clear remaining input data from the previous call)
  // you have to call Reset() before computing Spectrogram.
  // For example in tensorflow/core/kernels/spectrogram_op.cc
  bool Reset();

  // Processes an arbitrary amount of audio data (contained in input)
  // to yield complex spectrogram frames. After a successful call to
  // Initialize(), Process() may be called repeatedly with new input data
  // each time.  The audio input is buffered internally, and the output
  // vector is populated with as many temporally-ordered spectral slices
  // as it is possible to generate from the input.  The output is cleared
  // on each call before the new frames (if any) are added.
  //
  // The template parameters can be float or double.
  template <class InputSample, class OutputSample>
  bool ComputeComplexSpectrogram(
      const std::vector<InputSample>& input,
      std::vector<std::vector<std::complex<OutputSample>>>* output);

  // This function works as the one above, but returns the power
  // (the L2 norm, or the squared magnitude) of each complex value.
  template <class InputSample, class OutputSample>
  bool ComputeSquaredMagnitudeSpectrogram(
      const std::vector<InputSample>& input,
      std::vector<std::vector<OutputSample>>* output);

  // Return reference to the window function used internally.
  const std::vector<double>& GetWindow() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh mht_2(mht_2_v, 269, "", "./tensorflow/core/kernels/spectrogram.h", "GetWindow");
 return window_; }

  // Return the number of frequency channels in the spectrogram.
  int output_frequency_channels() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogramDTh mht_3(mht_3_v, 275, "", "./tensorflow/core/kernels/spectrogram.h", "output_frequency_channels");
 return output_frequency_channels_; }

 private:
  template <class InputSample>
  bool GetNextWindowOfSamples(const std::vector<InputSample>& input,
                              int* input_start);
  void ProcessCoreFFT();

  int fft_length_;
  int output_frequency_channels_;
  int window_length_;
  int step_length_;
  bool initialized_;
  int samples_to_next_step_;

  std::vector<double> window_;
  std::vector<double> fft_input_output_;
  std::deque<double> input_queue_;

  // Working data areas for the FFT routines.
  std::vector<int> fft_integer_working_area_;
  std::vector<double> fft_double_working_area_;

  TF_DISALLOW_COPY_AND_ASSIGN(Spectrogram);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPECTROGRAM_H_
