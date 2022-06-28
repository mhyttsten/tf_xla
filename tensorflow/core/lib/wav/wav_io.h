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

// Functions to write audio in WAV format.

#ifndef TENSORFLOW_LIB_WAV_WAV_IO_H_
#define TENSORFLOW_LIB_WAV_WAV_IO_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSwavPSwav_ioDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSwavPSwav_ioDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSwavPSwav_ioDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace wav {

// Encode the provided interleaved buffer of audio as a signed 16-bit PCM
// little-endian WAV file.
//
// Example usage for 4 frames of an 8kHz stereo signal:
// First channel is -1, 1, -1, 1.
// Second channel is 0, 0, 0, 0.
//
// float audio_buffer[] = { -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
// string wav_string;
// if (EncodeAudioAsS16LEWav(audio_buffer, 8000, 2, 4, &wav_string).ok()) {
//   // Use wav_string.
// }
template <typename T>
Status EncodeAudioAsS16LEWav(const float* audio, size_t sample_rate,
                             size_t num_channels, size_t num_frames,
                             T* wav_string);

// Explicit instantiations defined in wav_io.cc.
extern template Status EncodeAudioAsS16LEWav<std::string>(
    const float* audio, size_t sample_rate, size_t num_channels,
    size_t num_frames, std::string* wav_string);
extern template Status EncodeAudioAsS16LEWav<tstring>(const float* audio,
                                                      size_t sample_rate,
                                                      size_t num_channels,
                                                      size_t num_frames,
                                                      tstring* wav_string);

// Decodes the little-endian signed 16-bit PCM WAV file data (aka LIN16
// encoding) into a float Tensor. The channels are encoded as the lowest
// dimension of the tensor, with the number of frames as the second. This means
// that a four frame stereo signal will have the shape [4, 2]. The sample rate
// is read from the file header, and an error is returned if the format is not
// supported.
// The results are output as floats within the range -1 to 1,
Status DecodeLin16WaveAsFloatVector(const std::string& wav_string,
                                    std::vector<float>* float_values,
                                    uint32* sample_count, uint16* channel_count,
                                    uint32* sample_rate);

// Everything below here is only exposed publicly for testing purposes.

// Handles moving the data index forward, validating the arguments, and avoiding
// overflow or underflow.
Status IncrementOffset(int old_offset, size_t increment, size_t max_size,
                       int* new_offset);

// This function is only exposed in the header for testing purposes, as a
// template that needs to be instantiated. Reads a typed numeric value from a
// stream of data.
template <class T>
Status ReadValue(const std::string& data, T* value, int* offset) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSwavPSwav_ioDTh mht_0(mht_0_v, 252, "", "./tensorflow/core/lib/wav/wav_io.h", "ReadValue");

  int new_offset;
  TF_RETURN_IF_ERROR(
      IncrementOffset(*offset, sizeof(T), data.size(), &new_offset));
  if (port::kLittleEndian) {
    memcpy(value, data.data() + *offset, sizeof(T));
  } else {
    *value = 0;
    const uint8* data_buf =
        reinterpret_cast<const uint8*>(data.data() + *offset);
    int shift = 0;
    for (int i = 0; i < sizeof(T); ++i, shift += 8) {
      *value = *value | (data_buf[i] << shift);
    }
  }
  *offset = new_offset;
  return Status::OK();
}

}  // namespace wav
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_WAV_WAV_IO_H_
