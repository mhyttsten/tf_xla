/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OBJECT_ACCESSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OBJECT_ACCESSOR_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSobject_accessorDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSobject_accessorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSobject_accessorDTh() {
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


#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// This rewrite handles access to objects both reads and writes.
//
// The following syntax is supported to access objects:
//
//   READ:
//     vec4 value = $data[i]$;
//       where data is a buffer or 1D texture
//     vec4 value = $data[i,j]$;
//       where data is 2D texture
//     vec4 value = $data[i,j,k]$;
//       where data is 3D texture
//
//   WRITE:
//     $data[i] = value$;
//       where data is a buffer or 1D texture
//     $data[i,j] = value$;
//       where data is 2D texture
//     $data[i,j,k] = value$;
//       where data is 3D texture
//
// Accessor supports all types (gvecN) as well as float16.
//
// TODO(akulik): support field in data[x,y,z].x
//
class ObjectAccessor : public InlineRewrite {
 public:
  ObjectAccessor(bool is_mali, VariableAccessor* variable_accessor)
      : ObjectAccessor(is_mali, /*sampler_textures=*/false, variable_accessor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSobject_accessorDTh mht_0(mht_0_v, 227, "", "./tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h", "ObjectAccessor");

  }

  ObjectAccessor(bool is_mali, bool sampler_textures,
                 VariableAccessor* variable_accessor)
      : is_mali_(is_mali),
        sampler_textures_(sampler_textures),
        variable_accessor_(variable_accessor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSobject_accessorDTh mht_1(mht_1_v, 237, "", "./tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h", "ObjectAccessor");
}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final;

  // Return true if object was successfully added.
  bool AddObject(const std::string& name, Object object);

  // Returns objects declarations that need to be added in a shader's code.
  std::string GetObjectDeclarations() const;

  // Returns functions declarations that need to be added in a shader's code.
  // These functions are used by code accessing objects.
  std::string GetFunctionsDeclarations() const;

  // Returns a collection of registered objects
  std::vector<Object> GetObjects() const;

 private:
  RewriteStatus RewriteRead(absl::string_view location, std::string* output);

  RewriteStatus RewriteWrite(absl::string_view location,
                             absl::string_view value, std::string* output);

  std::map<std::string, Object> name_to_object_;

  const bool is_mali_;
  const bool sampler_textures_;
  VariableAccessor* variable_accessor_;
};

// Implementation details below.

namespace object_accessor_internal {

// Refers to an element in an object.
struct IndexedElement {
  absl::string_view object_name;
  std::vector<absl::string_view> indices;
};

// Splits name[index1, index2...] into 'name' and {'index1', 'index2'...}.
IndexedElement ParseElement(absl::string_view input);

}  // namespace object_accessor_internal
}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_OBJECT_ACCESSOR_H_
