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
// LINT.IfChange

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_ENTRY_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_ENTRY_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh() {
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


#include <algorithm>
#include <memory>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/ctc/ctc_loss_util.h"

namespace tensorflow {
namespace ctc {

// The ctc_beam_search namespace holds several classes meant to be accessed only
// in case of extending the CTCBeamSearch decoder to allow custom scoring
// functions.
//
// BeamEntry is exposed through template arguments BeamScorer and BeamComparer
// of CTCBeamSearch (ctc_beam_search.h).
namespace ctc_beam_search {

struct EmptyBeamState {};

template <typename T>
struct BeamProbability {
  BeamProbability()
      : total(kLogZero<T>()), blank(kLogZero<T>()), label(kLogZero<T>()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/util/ctc/ctc_beam_entry.h", "BeamProbability");
}
  void Reset() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh mht_1(mht_1_v, 220, "", "./tensorflow/core/util/ctc/ctc_beam_entry.h", "Reset");

    total = kLogZero<T>();
    blank = kLogZero<T>();
    label = kLogZero<T>();
  }
  T total;
  T blank;
  T label;
};

template <class T, class CTCBeamState>
class BeamRoot;

template <class T, class CTCBeamState = EmptyBeamState>
struct BeamEntry {
  // BeamRoot<CTCBeamState>::AddEntry() serves as the factory method.
  friend BeamEntry<T, CTCBeamState>* BeamRoot<T, CTCBeamState>::AddEntry(
      BeamEntry<T, CTCBeamState>* p, int l);
  inline bool Active() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh mht_2(mht_2_v, 241, "", "./tensorflow/core/util/ctc/ctc_beam_entry.h", "Active");
 return newp.total != kLogZero<T>(); }
  // Return the child at the given index, or construct a new one in-place if
  // none was found.
  BeamEntry<T, CTCBeamState>& GetChild(int ind) {
    auto entry = children.emplace(ind, nullptr);
    auto& child_entry = entry.first->second;
    // If this is a new child, populate the BeamEntry<CTCBeamState>*.
    if (entry.second) {
      child_entry = beam_root->AddEntry(this, ind);
    }
    return *child_entry;
  }
  std::vector<int> LabelSeq(bool merge_repeated) const {
    std::vector<int> labels;
    int prev_label = -1;
    const BeamEntry<T, CTCBeamState>* c = this;
    while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
      if (!merge_repeated || c->label != prev_label) {
        labels.push_back(c->label);
      }
      prev_label = c->label;
      c = c->parent;
    }
    std::reverse(labels.begin(), labels.end());
    return labels;
  }

  BeamEntry<T, CTCBeamState>* parent;
  int label;
  // All instances of child BeamEntry are owned by *beam_root.
  gtl::FlatMap<int, BeamEntry<T, CTCBeamState>*> children;
  BeamProbability<T> oldp;
  BeamProbability<T> newp;
  CTCBeamState state;

 private:
  // Constructor giving parent, label, and the beam_root.
  // The object pointed to by p cannot be copied and should not be moved,
  // otherwise parent will become invalid.
  // This private constructor is only called through the factory method
  // BeamRoot<CTCBeamState>::AddEntry().
  BeamEntry(BeamEntry* p, int l, BeamRoot<T, CTCBeamState>* beam_root)
      : parent(p), label(l), beam_root(beam_root) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh mht_3(mht_3_v, 286, "", "./tensorflow/core/util/ctc/ctc_beam_entry.h", "BeamEntry");
}
  BeamRoot<T, CTCBeamState>* beam_root;
  TF_DISALLOW_COPY_AND_ASSIGN(BeamEntry);
};

// This class owns all instances of BeamEntry.  This is used to avoid recursive
// destructor call during destruction.
template <class T, class CTCBeamState = EmptyBeamState>
class BeamRoot {
 public:
  BeamRoot(BeamEntry<T, CTCBeamState>* p, int l) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh mht_4(mht_4_v, 299, "", "./tensorflow/core/util/ctc/ctc_beam_entry.h", "BeamRoot");

    root_entry_ = AddEntry(p, l);
  }
  BeamRoot(const BeamRoot&) = delete;
  BeamRoot& operator=(const BeamRoot&) = delete;

  BeamEntry<T, CTCBeamState>* AddEntry(BeamEntry<T, CTCBeamState>* p, int l) {
    auto* new_entry = new BeamEntry<T, CTCBeamState>(p, l, this);
    beam_entries_.emplace_back(new_entry);
    return new_entry;
  }
  BeamEntry<T, CTCBeamState>* RootEntry() const { return root_entry_; }

 private:
  BeamEntry<T, CTCBeamState>* root_entry_ = nullptr;
  std::vector<std::unique_ptr<BeamEntry<T, CTCBeamState>>> beam_entries_;
};

// BeamComparer is the default beam comparer provided in CTCBeamSearch.
template <class T, class CTCBeamState = EmptyBeamState>
class BeamComparer {
 public:
  virtual ~BeamComparer() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSctcPSctc_beam_entryDTh mht_5(mht_5_v, 324, "", "./tensorflow/core/util/ctc/ctc_beam_entry.h", "~BeamComparer");
}
  virtual bool inline operator()(const BeamEntry<T, CTCBeamState>* a,
                                 const BeamEntry<T, CTCBeamState>* b) const {
    return a->newp.total > b->newp.total;
  }
};

}  // namespace ctc_beam_search

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_ENTRY_H_
// LINT.ThenChange(//tensorflow/lite/kernels/ctc/ctc_beam_entry.h)
