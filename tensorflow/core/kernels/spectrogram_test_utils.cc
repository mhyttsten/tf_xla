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
class MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc() {
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

#include "tensorflow/core/kernels/spectrogram_test_utils.h"

#include <math.h>
#include <stddef.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

bool ReadWaveFileToVector(const string& file_name, std::vector<double>* data) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "ReadWaveFileToVector");

  string wav_data;
  if (!ReadFileToString(Env::Default(), file_name, &wav_data).ok()) {
    LOG(ERROR) << "Wave file read failed for " << file_name;
    return false;
  }
  std::vector<float> decoded_data;
  uint32 decoded_sample_count;
  uint16 decoded_channel_count;
  uint32 decoded_sample_rate;
  if (!wav::DecodeLin16WaveAsFloatVector(
           wav_data, &decoded_data, &decoded_sample_count,
           &decoded_channel_count, &decoded_sample_rate)
           .ok()) {
    return false;
  }
  // Convert from float to double for the output value.
  data->resize(decoded_data.size());
  for (int i = 0; i < decoded_data.size(); ++i) {
    (*data)[i] = decoded_data[i];
  }
  return true;
}

bool ReadRawFloatFileToComplexVector(
    const string& file_name, int row_length,
    std::vector<std::vector<std::complex<double> > >* data) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "ReadRawFloatFileToComplexVector");

  data->clear();
  string data_string;
  if (!ReadFileToString(Env::Default(), file_name, &data_string).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  float real_out;
  float imag_out;
  const int kBytesPerValue = 4;
  CHECK_EQ(sizeof(real_out), kBytesPerValue);
  std::vector<std::complex<double> > data_row;
  int row_counter = 0;
  int offset = 0;
  const int end = data_string.size();
  while (offset < end) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    char arr[4];
    for (int i = 0; i < kBytesPerValue; ++i) {
      arr[3 - i] = *(data_string.data() + offset + i);
    }
    memcpy(&real_out, arr, kBytesPerValue);
    offset += kBytesPerValue;
    for (int i = 0; i < kBytesPerValue; ++i) {
      arr[3 - i] = *(data_string.data() + offset + i);
    }
    memcpy(&imag_out, arr, kBytesPerValue);
    offset += kBytesPerValue;
#else
    memcpy(&real_out, data_string.data() + offset, kBytesPerValue);
    offset += kBytesPerValue;
    memcpy(&imag_out, data_string.data() + offset, kBytesPerValue);
    offset += kBytesPerValue;
#endif
    if (row_counter >= row_length) {
      data->push_back(data_row);
      data_row.clear();
      row_counter = 0;
    }
    data_row.push_back(std::complex<double>(real_out, imag_out));
    ++row_counter;
  }
  if (row_counter >= row_length) {
    data->push_back(data_row);
  }
  return true;
}

void ReadCSVFileToComplexVectorOrDie(
    const string& file_name,
    std::vector<std::vector<std::complex<double> > >* data) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_2(mht_2_v, 286, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "ReadCSVFileToComplexVectorOrDie");

  data->clear();
  string data_string;
  if (!ReadFileToString(Env::Default(), file_name, &data_string).ok()) {
    LOG(FATAL) << "Failed to open file " << file_name;
    return;
  }
  std::vector<string> lines = str_util::Split(data_string, '\n');
  for (const string& line : lines) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::complex<double> > data_line;
    std::vector<string> values = str_util::Split(line, ',');
    for (std::vector<string>::const_iterator i = values.begin();
         i != values.end(); ++i) {
      // each element of values may be in the form:
      // 0.001+0.002i, 0.001, 0.001i, -1.2i, -1.2-3.2i, 1.5, 1.5e-03+21.0i
      std::vector<string> parts;
      // Find the first instance of + or - after the second character
      // in the string, that does not immediately follow an 'e'.
      size_t operator_index = i->find_first_of("+-", 2);
      if (operator_index < i->size() &&
          i->substr(operator_index - 1, 1) == "e") {
        operator_index = i->find_first_of("+-", operator_index + 1);
      }
      parts.push_back(i->substr(0, operator_index));
      if (operator_index < i->size()) {
        parts.push_back(i->substr(operator_index, string::npos));
      }

      double real_part = 0.0;
      double imaginary_part = 0.0;
      for (std::vector<string>::const_iterator j = parts.begin();
           j != parts.end(); ++j) {
        if (j->find_first_of("ij") != string::npos) {
          strings::safe_strtod(*j, &imaginary_part);
        } else {
          strings::safe_strtod(*j, &real_part);
        }
      }
      data_line.push_back(std::complex<double>(real_part, imaginary_part));
    }
    data->push_back(data_line);
  }
}

void ReadCSVFileToArrayOrDie(const string& filename,
                             std::vector<std::vector<float> >* array) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_3(mht_3_v, 338, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "ReadCSVFileToArrayOrDie");

  string contents;
  TF_CHECK_OK(ReadFileToString(Env::Default(), filename, &contents));
  std::vector<string> lines = str_util::Split(contents, '\n');
  contents.clear();

  array->clear();
  std::vector<float> values;
  for (int l = 0; l < lines.size(); ++l) {
    values.clear();
    std::vector<string> split_line = str_util::Split(lines[l], ",");
    for (const string& token : split_line) {
      float tmp;
      CHECK(strings::safe_strtof(token, &tmp));
      values.push_back(tmp);
    }
    array->push_back(values);
  }
}

bool WriteDoubleVectorToFile(const string& file_name,
                             const std::vector<double>& data) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_4(mht_4_v, 363, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "WriteDoubleVectorToFile");

  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < data.size(); ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteFloatVectorToFile(const string& file_name,
                            const std::vector<float>& data) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_5(mht_5_v, 389, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "WriteFloatVectorToFile");

  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < data.size(); ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteDoubleArrayToFile(const string& file_name, int size,
                            const double* data) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_6(mht_6_v, 415, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "WriteDoubleArrayToFile");

  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < size; ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteFloatArrayToFile(const string& file_name, int size,
                           const float* data) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_7(mht_7_v, 441, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "WriteFloatArrayToFile");

  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < size; ++i) {
    if (!file->Append(StringPiece(reinterpret_cast<const char*>(&(data[i])),
                                  sizeof(data[i])))
             .ok()) {
      LOG(ERROR) << "Failed to append to file " << file_name;
      return false;
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

bool WriteComplexVectorToRawFloatFile(
    const string& file_name,
    const std::vector<std::vector<std::complex<double> > >& data) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_8(mht_8_v, 468, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "WriteComplexVectorToRawFloatFile");

  std::unique_ptr<WritableFile> file;
  if (!Env::Default()->NewWritableFile(file_name, &file).ok()) {
    LOG(ERROR) << "Failed to open file " << file_name;
    return false;
  }
  for (int i = 0; i < data.size(); ++i) {
    for (int j = 0; j < data[i].size(); ++j) {
      const float real_part(real(data[i][j]));
      if (!file->Append(StringPiece(reinterpret_cast<const char*>(&real_part),
                                    sizeof(real_part)))
               .ok()) {
        LOG(ERROR) << "Failed to append to file " << file_name;
        return false;
      }

      const float imag_part(imag(data[i][j]));
      if (!file->Append(StringPiece(reinterpret_cast<const char*>(&imag_part),
                                    sizeof(imag_part)))
               .ok()) {
        LOG(ERROR) << "Failed to append to file " << file_name;
        return false;
      }
    }
  }
  if (!file->Close().ok()) {
    LOG(ERROR) << "Failed to close file " << file_name;
    return false;
  }
  return true;
}

void SineWave(int sample_rate, float frequency, float duration_seconds,
              std::vector<double>* data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspectrogram_test_utilsDTcc mht_9(mht_9_v, 504, "", "./tensorflow/core/kernels/spectrogram_test_utils.cc", "SineWave");

  data->clear();
  for (int i = 0; i < static_cast<int>(sample_rate * duration_seconds); ++i) {
    data->push_back(
        sin(2.0 * M_PI * i * frequency / static_cast<double>(sample_rate)));
  }
}

}  // namespace tensorflow
