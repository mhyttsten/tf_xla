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

#ifndef TENSORFLOW_CORE_FRAMEWORK_READER_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_READER_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh() {
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
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

class ReaderBaseState;

// Default implementation of ReaderInterface.
class ReaderBase : public ReaderInterface {
 public:
  // name: For use in error messages, should mention both the name of
  // the op and the node.
  explicit ReaderBase(const string& name);

  // Note that methods with names ending in "Locked" are called while
  // the ReaderBase's mutex is held.

  // Implement this function in descendants -----------------------------------

  // Produce the next key/value pair from the current work item.
  // This is called "Locked" since it is executed under a mutex
  // that serializes all Reader calls.
  // Usage:
  //  a) If a record was successfully produced, set *produced = true,
  //  and fill in *key and *value.
  //  b) If no more records will be produced for this work item, set
  //  *at_end = true.
  //  c) If a record was produced, but no more will be produced, you
  //     may either do both (a) and (b), or do (a) in this call and do (b) in
  //     the next call to ReadLocked().
  //  d) If there was an error producing (e.g. an error reading the file,
  //     data corruption), return a non-OK() status.  ReadLocked may be
  //     called again if the user reruns this part of the graph.
  virtual Status ReadLocked(tstring* key, tstring* value, bool* produced,
                            bool* at_end) = 0;

  // Descendants may optionally implement these -------------------------------

  // Produce up to num_records next key/value pairs from the current
  // work item, in the same manner of ReadLocked.
  virtual Status ReadUpToLocked(int64_t num_records, std::vector<tstring>* keys,
                                std::vector<tstring>* values, int64_t* num_read,
                                bool* at_end);

  // Called when work starts / finishes.
  virtual Status OnWorkStartedLocked() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh mht_0(mht_0_v, 236, "", "./tensorflow/core/framework/reader_base.h", "OnWorkStartedLocked");
 return Status::OK(); }
  virtual Status OnWorkFinishedLocked() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/framework/reader_base.h", "OnWorkFinishedLocked");
 return Status::OK(); }

  // Called to reset the Reader to a newly constructed state.
  virtual Status ResetLocked();

  // Default implementation generates an Unimplemented error.
  // See the protected helper methods below.
  virtual Status SerializeStateLocked(tstring* state);
  virtual Status RestoreStateLocked(const tstring& state);

  // Accessors ----------------------------------------------------------------

  // Always true during a call to ReadLocked().
  bool work_in_progress() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh mht_2(mht_2_v, 256, "", "./tensorflow/core/framework/reader_base.h", "work_in_progress");
 return work_finished_ < work_started_; }

  // Returns the name of the current work item (valid if
  // work_in_progress() returns true).  May change between calls to
  // ReadLocked().
  const tstring& current_work() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh mht_3(mht_3_v, 264, "", "./tensorflow/core/framework/reader_base.h", "current_work");
 return work_; }

  // What was passed to the constructor.
  const string& name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTh mht_4(mht_4_v, 270, "", "./tensorflow/core/framework/reader_base.h", "name");
 return name_; }

  // Produce the key name (from current_work and the actual key).
  tstring KeyName(const tstring& key) const;

 protected:
  // For descendants wishing to implement serialize & restore state.

  // Writes ReaderBase state to *state.
  void SaveBaseState(ReaderBaseState* state) const;

  // Restores ReaderBase state from state. Assumes state was filled
  // using SaveBaseState() above.
  Status RestoreBaseState(const ReaderBaseState& state);

 private:
  // For descendants that wish to obtain the next work item in a different way.
  // For implementing Read().  Dequeues the next work item from
  // *queue, and if successful returns "work" (a string). May block.
  virtual string GetNextWorkLocked(QueueInterface* queue,
                                   OpKernelContext* context) const;

  // Implementations of ReaderInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Read(QueueInterface* queue, tstring* key, tstring* value,
            OpKernelContext* context) override;

  // Produces up to num_records.
  // In this implementation all the records come from the same work unit.
  int64_t ReadUpTo(const int64_t num_records, QueueInterface* queue,
                   std::vector<tstring>* keys, std::vector<tstring>* value,
                   OpKernelContext* context) override;

  Status Reset() override;
  int64_t NumRecordsProduced() override;
  int64_t NumWorkUnitsCompleted() override;
  Status SerializeState(tstring* state) override;
  Status RestoreState(const tstring& state) override;

  mutable mutex mu_;
  const string name_;
  int64_t work_started_ = 0;
  int64_t work_finished_ = 0;
  int64_t num_records_produced_ = 0;
  tstring work_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_READER_BASE_H_
