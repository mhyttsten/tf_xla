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
class MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc() {
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

#include "tensorflow/core/lib/io/iterator.h"

namespace tensorflow {
namespace table {

Iterator::Iterator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_0(mht_0_v, 190, "", "./tensorflow/core/lib/io/iterator.cc", "Iterator::Iterator");

  cleanup_.function = nullptr;
  cleanup_.next = nullptr;
}

Iterator::~Iterator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_1(mht_1_v, 198, "", "./tensorflow/core/lib/io/iterator.cc", "Iterator::~Iterator");

  if (cleanup_.function != nullptr) {
    (*cleanup_.function)(cleanup_.arg1, cleanup_.arg2);
    for (Cleanup* c = cleanup_.next; c != nullptr;) {
      (*c->function)(c->arg1, c->arg2);
      Cleanup* next = c->next;
      delete c;
      c = next;
    }
  }
}

void Iterator::RegisterCleanup(CleanupFunction func, void* arg1, void* arg2) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/lib/io/iterator.cc", "Iterator::RegisterCleanup");

  assert(func != nullptr);
  Cleanup* c;
  if (cleanup_.function == nullptr) {
    c = &cleanup_;
  } else {
    c = new Cleanup;
    c->next = cleanup_.next;
    cleanup_.next = c;
  }
  c->function = func;
  c->arg1 = arg1;
  c->arg2 = arg2;
}

namespace {
class EmptyIterator : public Iterator {
 public:
  explicit EmptyIterator(const Status& s) : status_(s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/lib/io/iterator.cc", "EmptyIterator");
}
  bool Valid() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/lib/io/iterator.cc", "Valid");
 return false; }
  void Seek(const StringPiece& target) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_5(mht_5_v, 242, "", "./tensorflow/core/lib/io/iterator.cc", "Seek");
}
  void SeekToFirst() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_6(mht_6_v, 246, "", "./tensorflow/core/lib/io/iterator.cc", "SeekToFirst");
}
  void Next() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_7(mht_7_v, 250, "", "./tensorflow/core/lib/io/iterator.cc", "Next");
 assert(false); }
  StringPiece key() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_8(mht_8_v, 254, "", "./tensorflow/core/lib/io/iterator.cc", "key");

    assert(false);
    return StringPiece();
  }
  StringPiece value() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_9(mht_9_v, 261, "", "./tensorflow/core/lib/io/iterator.cc", "value");

    assert(false);
    return StringPiece();
  }
  Status status() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_10(mht_10_v, 268, "", "./tensorflow/core/lib/io/iterator.cc", "status");
 return status_; }

 private:
  Status status_;
};
}  // namespace

Iterator* NewEmptyIterator() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_11(mht_11_v, 278, "", "./tensorflow/core/lib/io/iterator.cc", "NewEmptyIterator");
 return new EmptyIterator(Status::OK()); }

Iterator* NewErrorIterator(const Status& status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSiteratorDTcc mht_12(mht_12_v, 283, "", "./tensorflow/core/lib/io/iterator.cc", "NewErrorIterator");

  return new EmptyIterator(status);
}

}  // namespace table
}  // namespace tensorflow
