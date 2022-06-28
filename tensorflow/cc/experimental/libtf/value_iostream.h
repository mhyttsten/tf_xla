/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_VALUE_IOSTREAM_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_VALUE_IOSTREAM_H_
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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh() {
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


#include <iostream>

#include "tensorflow/cc/experimental/libtf/value.h"

namespace tf {
namespace libtf {
namespace impl {

inline std::ostream& operator<<(std::ostream& o, const Dict& v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh mht_0(mht_0_v, 195, "", "./tensorflow/cc/experimental/libtf/value_iostream.h", "operator<<");

  o << "{";
  for (auto& x : v) {
    o << x.first;
    o << ": ";
    o << x.second;
    o << ", ";
  }
  o << "}";
  return o;
}
template <class IT>
inline std::ostream& OutList(std::ostream& o, IT v_start, IT const v_end,
                             char start, char end) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("start: '" + std::string(1, start) + "'");
   mht_1_v.push_back("end: '" + std::string(1, end) + "'");
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh mht_1(mht_1_v, 213, "", "./tensorflow/cc/experimental/libtf/value_iostream.h", "OutList");

  o << start;
  for (IT p = v_start; p != v_end; ++p) {
    o << *p;
    o << ", ";
  }
  o << end;
  return o;
}

class TaggedValueIOStreamVisitor {
  std::ostream& o_;

 public:
  explicit TaggedValueIOStreamVisitor(std::ostream& o) : o_(o) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh mht_2(mht_2_v, 230, "", "./tensorflow/cc/experimental/libtf/value_iostream.h", "TaggedValueIOStreamVisitor");
}

  std::ostream& operator()(const ListPtr& x) {
    OutList(o_, x->begin(), x->end(), '[', ']');
    return o_;
  }
  std::ostream& operator()(const TuplePtr& x) {
    OutList(o_, x->begin(), x->end(), '(', ')');
    return o_;
  }
  std::ostream& operator()(const DictPtr& x) {
    o_ << *x;
    return o_;
  }
  std::ostream& operator()(const Capsule& x) {
    o_ << "Capsule(" << x.get() << ")";
    return o_;
  }
  std::ostream& operator()(const Func& x) {
    o_ << "Func";
    return o_;
  }
  std::ostream& operator()(const TaggedValueTensor& x) {
    o_ << "Tensor";
    return o_;
  }

  template <class T>
  std::ostream& operator()(const T& x) {
    o_ << x;
    return o_;
  }
};

inline std::ostream& operator<<(std::ostream& o, const TaggedValue& v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSvalue_iostreamDTh mht_3(mht_3_v, 267, "", "./tensorflow/cc/experimental/libtf/value_iostream.h", "operator<<");

  return v.visit<std::ostream&>(TaggedValueIOStreamVisitor(o));
}
}  // namespace impl
}  // namespace libtf
}  // namespace tf
#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_VALUE_IOSTREAM_H_
