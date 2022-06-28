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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc() {
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

// This code resamples the FFT bins, and smooths then with triangle-shaped
// weights to create a mel-frequency filter bank. For filter i centered at f_i,
// there is a triangular weighting of the FFT bins that extends from
// filter f_i-1 (with a value of zero at the left edge of the triangle) to f_i
// (where the filter value is 1) to f_i+1 (where the filter values returns to
// zero).

// Note: this code fails if you ask for too many channels.  The algorithm used
// here assumes that each FFT bin contributes to at most two channels: the
// right side of a triangle for channel i, and the left side of the triangle
// for channel i+1.  If you ask for so many channels that some of the
// resulting mel triangle filters are smaller than a single FFT bin, these
// channels may end up with no contributing FFT bins.  The resulting mel
// spectrum output will have some channels that are always zero.

#include "tensorflow/lite/kernels/internal/mfcc_mel_filterbank.h"

#include <math.h>

namespace tflite {
namespace internal {

MfccMelFilterbank::MfccMelFilterbank() : initialized_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/internal/mfcc_mel_filterbank.cc", "MfccMelFilterbank::MfccMelFilterbank");
}

bool MfccMelFilterbank::Initialize(int input_length, double input_sample_rate,
                                   int output_channel_count,
                                   double lower_frequency_limit,
                                   double upper_frequency_limit) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/kernels/internal/mfcc_mel_filterbank.cc", "MfccMelFilterbank::Initialize");

  num_channels_ = output_channel_count;
  sample_rate_ = input_sample_rate;
  input_length_ = input_length;

  if (num_channels_ < 1) {
    // LOG(ERROR) << "Number of filterbank channels must be positive.";
    return false;
  }

  if (sample_rate_ <= 0) {
    // LOG(ERROR) << "Sample rate must be positive.";
    return false;
  }

  if (input_length < 2) {
    // LOG(ERROR) << "Input length must greater than 1.";
    return false;
  }

  if (lower_frequency_limit < 0) {
    // LOG(ERROR) << "Lower frequency limit must be nonnegative.";
    return false;
  }

  if (upper_frequency_limit <= lower_frequency_limit) {
    /// LOG(ERROR) << "Upper frequency limit must be greater than "
    //           << "lower frequency limit.";
    return false;
  }

  // An extra center frequency is computed at the top to get the upper
  // limit on the high side of the final triangular filter.
  center_frequencies_.resize(num_channels_ + 1);
  const double mel_low = FreqToMel(lower_frequency_limit);
  const double mel_hi = FreqToMel(upper_frequency_limit);
  const double mel_span = mel_hi - mel_low;
  const double mel_spacing = mel_span / static_cast<double>(num_channels_ + 1);
  for (int i = 0; i < num_channels_ + 1; ++i) {
    center_frequencies_[i] = mel_low + (mel_spacing * (i + 1));
  }

  // Always exclude DC; emulate HTK.
  const double hz_per_sbin =
      0.5 * sample_rate_ / static_cast<double>(input_length_ - 1);
  start_index_ = static_cast<int>(1.5 + (lower_frequency_limit / hz_per_sbin));
  end_index_ = static_cast<int>(upper_frequency_limit / hz_per_sbin);

  // Maps the input spectrum bin indices to filter bank channels/indices. For
  // each FFT bin, band_mapper tells us which channel this bin contributes to
  // on the right side of the triangle.  Thus this bin also contributes to the
  // left side of the next channel's triangle response.
  band_mapper_.resize(input_length_);
  int channel = 0;
  for (int i = 0; i < input_length_; ++i) {
    double melf = FreqToMel(i * hz_per_sbin);
    if ((i < start_index_) || (i > end_index_)) {
      band_mapper_[i] = -2;  // Indicate an unused Fourier coefficient.
    } else {
      while ((channel < num_channels_) &&
             (center_frequencies_[channel] < melf)) {
        ++channel;
      }
      band_mapper_[i] = channel - 1;  // Can be == -1
    }
  }

  // Create the weighting functions to taper the band edges.  The contribution
  // of any one FFT bin is based on its distance along the continuum between two
  // mel-channel center frequencies.  This bin contributes weights_[i] to the
  // current channel and 1-weights_[i] to the next channel.
  weights_.resize(input_length_);
  for (int i = 0; i < input_length_; ++i) {
    channel = band_mapper_[i];
    if ((i < start_index_) || (i > end_index_)) {
      weights_[i] = 0.0;
    } else {
      if (channel >= 0) {
        weights_[i] =
            (center_frequencies_[channel + 1] - FreqToMel(i * hz_per_sbin)) /
            (center_frequencies_[channel + 1] - center_frequencies_[channel]);
      } else {
        weights_[i] = (center_frequencies_[0] - FreqToMel(i * hz_per_sbin)) /
                      (center_frequencies_[0] - mel_low);
      }
    }
  }
  // Check the sum of FFT bin weights for every mel band to identify
  // situations where the mel bands are so narrow that they don't get
  // significant weight on enough (or any) FFT bins -- i.e., too many
  // mel bands have been requested for the given FFT size.
  std::vector<int> bad_channels;
  for (int c = 0; c < num_channels_; ++c) {
    float band_weights_sum = 0.0;
    for (int i = 0; i < input_length_; ++i) {
      if (band_mapper_[i] == c - 1) {
        band_weights_sum += (1.0 - weights_[i]);
      } else if (band_mapper_[i] == c) {
        band_weights_sum += weights_[i];
      }
    }
    // The lowest mel channels have the fewest FFT bins and the lowest
    // weights sum.  But given that the target gain at the center frequency
    // is 1.0, if the total sum of weights is 0.5, we're in bad shape.
    if (band_weights_sum < 0.5) {
      bad_channels.push_back(c);
    }
  }
  if (!bad_channels.empty()) {
    /*
    LOG(ERROR) << "Missing " << bad_channels.size() << " bands "
               << " starting at " << bad_channels[0]
               << " in mel-frequency design. "
               << "Perhaps too many channels or "
               << "not enough frequency resolution in spectrum. ("
               << "input_length: " << input_length
               << " input_sample_rate: " << input_sample_rate
               << " output_channel_count: " << output_channel_count
               << " lower_frequency_limit: " << lower_frequency_limit
               << " upper_frequency_limit: " << upper_frequency_limit;
               */
  }
  initialized_ = true;
  return true;
}

// Compute the mel spectrum from the squared-magnitude FFT input by taking the
// square root, then summing FFT magnitudes under triangular integration windows
// whose widths increase with frequency.
void MfccMelFilterbank::Compute(const std::vector<double> &input,
                                std::vector<double> *output) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc mht_2(mht_2_v, 348, "", "./tensorflow/lite/kernels/internal/mfcc_mel_filterbank.cc", "MfccMelFilterbank::Compute");

  if (!initialized_) {
    // LOG(ERROR) << "Mel Filterbank not initialized.";
    return;
  }

  if (input.size() <= end_index_) {
    // LOG(ERROR) << "Input too short to compute filterbank";
    return;
  }

  // Ensure output is right length and reset all values.
  output->assign(num_channels_, 0.0);

  for (int i = start_index_; i <= end_index_; i++) {  // For each FFT bin
    double spec_val = sqrt(input[i]);
    double weighted = spec_val * weights_[i];
    int channel = band_mapper_[i];
    if (channel >= 0)
      (*output)[channel] += weighted;  // Right side of triangle, downward slope
    channel++;
    if (channel < num_channels_)
      (*output)[channel] += spec_val - weighted;  // Left side of triangle
  }
}

double MfccMelFilterbank::FreqToMel(double freq) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSmfcc_mel_filterbankDTcc mht_3(mht_3_v, 377, "", "./tensorflow/lite/kernels/internal/mfcc_mel_filterbank.cc", "MfccMelFilterbank::FreqToMel");

  return 1127.0 * log1p(freq / 700.0);
}

}  // namespace internal
}  // namespace tflite
