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
class MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc() {
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
#include "tensorflow/core/util/memmapped_file_system_writer.h"

#include <algorithm>

namespace tensorflow {

Status MemmappedFileSystemWriter::InitializeToFile(Env* env,
                                                   const string& filename) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "MemmappedFileSystemWriter::InitializeToFile");

  auto status = env->NewWritableFile(filename, &output_file_);
  if (status.ok()) {
    output_file_offset_ = 0;
  }
  return status;
}

Status MemmappedFileSystemWriter::SaveTensor(const Tensor& tensor,
                                             const string& element_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("element_name: \"" + element_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "MemmappedFileSystemWriter::SaveTensor");

  if (!output_file_) {
    return errors::FailedPrecondition(
        "MemmappedEnvWritter: saving tensor into not opened file");
  }
  if (!MemmappedFileSystem::IsWellFormedMemmappedPackageFilename(
          element_name)) {
    return errors::InvalidArgument(
        "MemmappedEnvWritter: element_name is invalid: must have memmapped ",
        "package prefix ", MemmappedFileSystem::kMemmappedPackagePrefix,
        " and include [A-Za-z0-9_.]");
  }
  const auto tensor_data = tensor.tensor_data();
  if (tensor_data.empty()) {
    return errors::InvalidArgument(
        "MemmappedEnvWritter: saving tensor with 0 size");
  }
  // Adds pad for correct alignment after memmapping.
  TF_RETURN_IF_ERROR(AdjustAlignment(Allocator::kAllocatorAlignment));
  AddToDirectoryElement(element_name, tensor_data.size());
  const auto result = output_file_->Append(tensor_data);
  if (result.ok()) {
    output_file_offset_ += tensor_data.size();
  }
  return result;
}

Status MemmappedFileSystemWriter::SaveProtobuf(
    const protobuf::MessageLite& message, const string& element_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("element_name: \"" + element_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "MemmappedFileSystemWriter::SaveProtobuf");

  if (!output_file_) {
    return errors::FailedPrecondition(
        "MemmappedEnvWritter: saving protobuf into not opened file");
  }
  if (!MemmappedFileSystem::IsWellFormedMemmappedPackageFilename(
          element_name)) {
    return errors::InvalidArgument(
        "MemmappedEnvWritter: element_name is invalid: must have memmapped "
        "package prefix ",
        MemmappedFileSystem::kMemmappedPackagePrefix,
        " and include [A-Za-z0-9_.]");
  }
  const string encoded = message.SerializeAsString();
  AddToDirectoryElement(element_name, encoded.size());
  const auto res = output_file_->Append(encoded);
  if (res.ok()) {
    output_file_offset_ += encoded.size();
  }
  return res;
}

namespace {

StringPiece EncodeUint64LittleEndian(uint64 val, char* output_buffer) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("output_buffer: \"" + (output_buffer == nullptr ? std::string("nullptr") : std::string((char*)output_buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "EncodeUint64LittleEndian");

  for (unsigned int i = 0; i < sizeof(uint64); ++i) {
    output_buffer[i] = (val >> i * 8);
  }
  return {output_buffer, sizeof(uint64)};
}

}  // namespace

Status MemmappedFileSystemWriter::FlushAndClose() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_4(mht_4_v, 277, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "MemmappedFileSystemWriter::FlushAndClose");

  if (!output_file_) {
    return errors::FailedPrecondition(
        "MemmappedEnvWritter: flushing into not opened file");
  }
  const string dir = directory_.SerializeAsString();
  TF_RETURN_IF_ERROR(output_file_->Append(dir));

  // Write the directory offset.
  char buffer[sizeof(uint64)];
  TF_RETURN_IF_ERROR(output_file_->Append(
      EncodeUint64LittleEndian(output_file_offset_, buffer)));

  // Flush and close the file.
  TF_RETURN_IF_ERROR(output_file_->Flush());
  TF_RETURN_IF_ERROR(output_file_->Close());
  output_file_.reset();
  return Status::OK();
}

Status MemmappedFileSystemWriter::AdjustAlignment(uint64 alignment) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "MemmappedFileSystemWriter::AdjustAlignment");

  const uint64 alignment_rest = output_file_offset_ % alignment;
  const uint64 to_write_for_alignment =
      (alignment_rest == 0) ? 0 : alignment - (output_file_offset_ % alignment);
  static constexpr uint64 kFillerBufferSize = 16;
  const char kFillerBuffer[kFillerBufferSize] = {};
  for (uint64 rest = to_write_for_alignment; rest > 0;) {
    StringPiece sp(kFillerBuffer, std::min(rest, kFillerBufferSize));
    TF_RETURN_IF_ERROR(output_file_->Append(sp));
    rest -= sp.size();
    output_file_offset_ += sp.size();
  }
  return Status::OK();
}

void MemmappedFileSystemWriter::AddToDirectoryElement(const string& name,
                                                      uint64 length) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_writerDTcc mht_6(mht_6_v, 320, "", "./tensorflow/core/util/memmapped_file_system_writer.cc", "MemmappedFileSystemWriter::AddToDirectoryElement");

  MemmappedFileSystemDirectoryElement* new_directory_element =
      directory_.add_element();
  new_directory_element->set_offset(output_file_offset_);
  new_directory_element->set_name(name);
  new_directory_element->set_length(length);
}

}  // namespace tensorflow
