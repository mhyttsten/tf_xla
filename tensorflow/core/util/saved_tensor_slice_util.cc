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
class MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/ordered_code.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace checkpoint {

const char kSavedTensorSlicesKey[] = "";

string EncodeTensorNameSlice(const string& name, const TensorSlice& slice) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/util/saved_tensor_slice_util.cc", "EncodeTensorNameSlice");

  string buffer;
  // All the tensor slice keys will start with a 0
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, 0);
  tensorflow::strings::OrderedCode::WriteString(&buffer, name);
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, slice.dims());
  for (int d = 0; d < slice.dims(); ++d) {
    // A trivial extent (meaning we take EVERYTHING) will default to -1 for both
    // start and end. These will be properly parsed.
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.start(d));
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.length(d));
  }
  return buffer;
}

Status DecodeTensorNameSlice(const string& code, string* name,
                             tensorflow::TensorSlice* slice) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("code: \"" + code + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/util/saved_tensor_slice_util.cc", "DecodeTensorNameSlice");

  StringPiece src(code);
  uint64 x;
  if (!tensorflow::strings::OrderedCode::ReadNumIncreasing(&src, &x)) {
    return errors::Internal("Failed to parse the leading number: src = ", src);
  }
  if (x != 0) {
    return errors::Internal(
        "The leading number should always be 0 for any valid key: src = ", src);
  }
  if (!tensorflow::strings::OrderedCode::ReadString(&src, name)) {
    return errors::Internal("Failed to parse the tensor name: src = ", src);
  }
  if (!tensorflow::strings::OrderedCode::ReadNumIncreasing(&src, &x)) {
    return errors::Internal("Failed to parse the tensor rank: src = ", src);
  }
  if (x == 0) {
    return errors::Internal("Expecting positive rank of the tensor, got ", x,
                            ", src = ", src);
  }
  if (x >= kint32max) {
    return errors::Internal("Too many elements ", x);
  }
  slice->SetFullSlice(x);
  for (int d = 0; d < static_cast<int32>(x); ++d) {
    // We expected 2x integers
    int64_t start, length;
    if (!tensorflow::strings::OrderedCode::ReadSignedNumIncreasing(&src,
                                                                   &start)) {
      return errors::Internal("Failed to parse start: src = ", src);
    }
    if (!tensorflow::strings::OrderedCode::ReadSignedNumIncreasing(&src,
                                                                   &length)) {
      return errors::Internal("Failed to parse length: src = ", src);
    }
    if (length >= 0) {
      // a non-trivial extent
      slice->set_start(d, start);
      slice->set_length(d, length);
    }
  }
  return Status::OK();
}

Status ParseShapeAndSlice(const string& shape_and_slice, TensorShape* shape,
                          TensorSlice* slice, TensorShape* shape_slice) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("shape_and_slice: \"" + shape_and_slice + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTcc mht_2(mht_2_v, 269, "", "./tensorflow/core/util/saved_tensor_slice_util.cc", "ParseShapeAndSlice");

  CHECK(!shape_and_slice.empty());
  // Syntax: dim0 dim1 dim2 ... <slice string>
  // Where slice string is defined in core/framework/tensor_slice.h
  std::vector<string> splits = str_util::Split(shape_and_slice, ' ');

  // Must have at least 2 strings.
  if (splits.size() < 2) {
    return errors::InvalidArgument(
        "Need least two elements in shape_and_slice specification: ",
        shape_and_slice);
  }

  // The last split is the slice specification.
  slice->Clear();
  auto status = slice->Parse(splits.back(), slice);
  if (!status.ok()) return status;

  // The first n-1 are the shape specification.
  splits.pop_back();
  shape->Clear();
  for (const auto& s : splits) {
    int64_t dim;
    if (!strings::safe_strto64(s, &dim)) {
      return errors::InvalidArgument(
          "Non numerical dimension in shape_and_slice: ", shape_and_slice);
    }
    shape->AddDim(dim);
  }

  // The specified slice must be compatible with the specified shape.
  return slice->SliceTensorShape(*shape, shape_slice);
}

}  // namespace checkpoint

}  // namespace tensorflow
