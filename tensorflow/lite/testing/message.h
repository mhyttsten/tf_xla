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
#ifndef TENSORFLOW_LITE_TESTING_MESSAGE_H_
#define TENSORFLOW_LITE_TESTING_MESSAGE_H_
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
class MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh {
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
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh() {
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
#include <string>
#include <vector>

namespace tflite {
namespace testing {

// A Message is a textual protobuf-like structure that looks like:
//    tag {
//      f : "values"
//      child {
//        a : 1
//       }
//    }
// This class provides the framework for processing message but does not
// associate any particular behavior to fields and submessage. In order
// to properly parse a stream this class must be derived.
class Message {
 public:
  // Reads a stream, tokenizes it and create a new message under the given
  // top-level message. Returns true if the parsing succeeded.
  static bool Read(std::istream* input, Message* message);

  Message() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_0(mht_0_v, 210, "", "./tensorflow/lite/testing/message.h", "Message");
}
  virtual ~Message() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_1(mht_1_v, 214, "", "./tensorflow/lite/testing/message.h", "~Message");
}

  // Called when a new field is found. For example, when:
  //   f : "values"
  // is found, it triggers:
  //   SetField("f", "values");
  virtual void SetField(const std::string& name, const std::string& value) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_2(mht_2_v, 225, "", "./tensorflow/lite/testing/message.h", "SetField");
}

  // Called when a submessage is started. For example, when:
  //   child {
  // is found, it triggers
  //   AddChild("child");
  // If nullptr is returned, the contents of the submessage will be ignored.
  // Otherwise, the returned Message will be used to handle new fields and new
  // submessages. The caller should not take ownership of the returned pointer.
  virtual Message* AddChild(const std::string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_3(mht_3_v, 238, "", "./tensorflow/lite/testing/message.h", "AddChild");
 return nullptr; }

  // Called when a submessage is completed, that is, whenever a '}' is found.
  virtual void Finish() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_4(mht_4_v, 244, "", "./tensorflow/lite/testing/message.h", "Finish");
}

 protected:
  // Takes ownership of the given pointer. Subclasses can use this method if
  // they don't want to implement their own ownership semantics.
  Message* Store(Message* n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_5(mht_5_v, 252, "", "./tensorflow/lite/testing/message.h", "Store");

    children_.emplace_back(n);
    return n;
  }

  // Returns a list of all owned submessages.
  const std::vector<std::unique_ptr<Message>>& Children() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStestingPSmessageDTh mht_6(mht_6_v, 261, "", "./tensorflow/lite/testing/message.h", "Children");

    return children_;
  }

 private:
  std::vector<std::unique_ptr<Message>> children_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_MESSAGE_H_
