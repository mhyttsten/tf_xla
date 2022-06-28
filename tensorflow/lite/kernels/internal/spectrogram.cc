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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc() {
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

#include "tensorflow/lite/kernels/internal/spectrogram.h"

#include <assert.h>
#include <math.h>

#include "third_party/fft2d/fft.h"

namespace tflite {
namespace internal {

using std::complex;

namespace {
// Returns the default Hann window function for the spectrogram.
void GetPeriodicHann(int window_length, std::vector<double>* window) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "GetPeriodicHann");

  // Some platforms don't have M_PI, so define a local constant here.
  const double pi = std::atan(1.0) * 4.0;
  window->resize(window_length);
  for (int i = 0; i < window_length; ++i) {
    (*window)[i] = 0.5 - 0.5 * cos((2.0 * pi * i) / window_length);
  }
}
}  // namespace

bool Spectrogram::Initialize(int window_length, int step_length) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "Spectrogram::Initialize");

  std::vector<double> window;
  GetPeriodicHann(window_length, &window);
  return Initialize(window, step_length);
}

inline int Log2Floor(uint32_t n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "Log2Floor");

  if (n == 0) return -1;
  int log = 0;
  uint32_t value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}

inline int Log2Ceiling(uint32_t n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_3(mht_3_v, 239, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "Log2Ceiling");

  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline uint32_t NextPowerOfTwo(uint32_t value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_4(mht_4_v, 250, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "NextPowerOfTwo");

  int exponent = Log2Ceiling(value);
  // DCHECK_LT(exponent, std::numeric_limits<uint32>::digits);
  return 1 << exponent;
}

bool Spectrogram::Initialize(const std::vector<double>& window,
                             int step_length) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_5(mht_5_v, 260, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "Spectrogram::Initialize");

  window_length_ = window.size();
  window_ = window;  // Copy window.
  if (window_length_ < 2) {
    // LOG(ERROR) << "Window length too short.";
    initialized_ = false;
    return false;
  }

  step_length_ = step_length;
  if (step_length_ < 1) {
    // LOG(ERROR) << "Step length must be positive.";
    initialized_ = false;
    return false;
  }

  fft_length_ = NextPowerOfTwo(window_length_);
  // CHECK(fft_length_ >= window_length_);
  output_frequency_channels_ = 1 + fft_length_ / 2;

  // Allocate 2 more than what rdft needs, so we can rationalize the layout.
  fft_input_output_.assign(fft_length_ + 2, 0.0);

  int half_fft_length = fft_length_ / 2;
  fft_double_working_area_.assign(half_fft_length, 0.0);
  fft_integer_working_area_.assign(2 + static_cast<int>(sqrt(half_fft_length)),
                                   0);
  // Set flag element to ensure that the working areas are initialized
  // on the first call to cdft.  It's redundant given the assign above,
  // but keep it as a reminder.
  fft_integer_working_area_[0] = 0;
  input_queue_.clear();
  samples_to_next_step_ = window_length_;
  initialized_ = true;
  return true;
}

template <class InputSample, class OutputSample>
bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<InputSample>& input,
    std::vector<std::vector<complex<OutputSample>>>* output) {
  if (!initialized_) {
    // LOG(ERROR) << "ComputeComplexSpectrogram() called before successful call
    // "
    //           << "to Initialize().";
    return false;
  }
  // CHECK(output);
  output->clear();
  int input_start = 0;
  while (GetNextWindowOfSamples(input, &input_start)) {
    // DCHECK_EQ(input_queue_.size(), window_length_);
    ProcessCoreFFT();  // Processes input_queue_ to fft_input_output_.
    // Add a new slice vector onto the output, to save new result to.
    output->resize(output->size() + 1);
    // Get a reference to the newly added slice to fill in.
    auto& spectrogram_slice = output->back();
    spectrogram_slice.resize(output_frequency_channels_);
    for (int i = 0; i < output_frequency_channels_; ++i) {
      // This will convert double to float if it needs to.
      spectrogram_slice[i] = complex<OutputSample>(
          fft_input_output_[2 * i], fft_input_output_[2 * i + 1]);
    }
  }
  return true;
}
// Instantiate it four ways:
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<complex<float>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<complex<float>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input,
    std::vector<std::vector<complex<double>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<complex<double>>>*);

template <class InputSample, class OutputSample>
bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<InputSample>& input,
    std::vector<std::vector<OutputSample>>* output) {
  if (!initialized_) {
    // LOG(ERROR) << "ComputeSquaredMagnitudeSpectrogram() called before "
    //           << "successful call to Initialize().";
    return false;
  }
  // CHECK(output);
  output->clear();
  int input_start = 0;
  while (GetNextWindowOfSamples(input, &input_start)) {
    // DCHECK_EQ(input_queue_.size(), window_length_);
    ProcessCoreFFT();  // Processes input_queue_ to fft_input_output_.
    // Add a new slice vector onto the output, to save new result to.
    output->resize(output->size() + 1);
    // Get a reference to the newly added slice to fill in.
    auto& spectrogram_slice = output->back();
    spectrogram_slice.resize(output_frequency_channels_);
    for (int i = 0; i < output_frequency_channels_; ++i) {
      // Similar to the Complex case, except storing the norm.
      // But the norm function is known to be a performance killer,
      // so do it this way with explicit real and imaginary temps.
      const double re = fft_input_output_[2 * i];
      const double im = fft_input_output_[2 * i + 1];
      // Which finally converts double to float if it needs to.
      spectrogram_slice[i] = re * re + im * im;
    }
  }
  return true;
}
// Instantiate it four ways:
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<float>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<float>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<double>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<double>>*);

// Return true if a full window of samples is prepared; manage the queue.
template <class InputSample>
bool Spectrogram::GetNextWindowOfSamples(const std::vector<InputSample>& input,
                                         int* input_start) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_6(mht_6_v, 387, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "Spectrogram::GetNextWindowOfSamples");

  auto input_it = input.begin() + *input_start;
  int input_remaining = input.end() - input_it;
  if (samples_to_next_step_ > input_remaining) {
    // Copy in as many samples are left and return false, no full window.
    input_queue_.insert(input_queue_.end(), input_it, input.end());
    *input_start += input_remaining;  // Increases it to input.size().
    samples_to_next_step_ -= input_remaining;
    return false;  // Not enough for a full window.
  } else {
    // Copy just enough into queue to make a new window, then trim the
    // front off the queue to make it window-sized.
    input_queue_.insert(input_queue_.end(), input_it,
                        input_it + samples_to_next_step_);
    *input_start += samples_to_next_step_;
    input_queue_.erase(
        input_queue_.begin(),
        input_queue_.begin() + input_queue_.size() - window_length_);
    // DCHECK_EQ(window_length_, input_queue_.size());
    samples_to_next_step_ = step_length_;  // Be ready for next time.
    return true;  // Yes, input_queue_ now contains exactly a window-full.
  }
}

void Spectrogram::ProcessCoreFFT() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSspectrogramDTcc mht_7(mht_7_v, 414, "", "./tensorflow/lite/kernels/internal/spectrogram.cc", "Spectrogram::ProcessCoreFFT");

  for (int j = 0; j < window_length_; ++j) {
    fft_input_output_[j] = input_queue_[j] * window_[j];
  }
  // Zero-pad the rest of the input buffer.
  for (int j = window_length_; j < fft_length_; ++j) {
    fft_input_output_[j] = 0.0;
  }
  const int kForwardFFT = 1;  // 1 means forward; -1 reverse.
  // This real FFT is a fair amount faster than using cdft here.
  rdft(fft_length_, kForwardFFT, &fft_input_output_[0],
       &fft_integer_working_area_[0], &fft_double_working_area_[0]);
  // Make rdft result look like cdft result;
  // unpack the last real value from the first position's imag slot.
  fft_input_output_[fft_length_] = fft_input_output_[1];
  fft_input_output_[fft_length_ + 1] = 0;
  fft_input_output_[1] = 0;
}

}  // namespace internal
}  // namespace tflite
