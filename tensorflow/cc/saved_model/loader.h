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

/// SavedModel loading functions and SavedModelBundle struct.

#ifndef TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
#define TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh() {
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
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

/// Represents a SavedModel that is loaded from storage.
class SavedModelBundleInterface {
 public:
  virtual ~SavedModelBundleInterface();

  /// Returns the TensorFlow Session that can be used to interact with the
  /// SavedModel.
  virtual Session* GetSession() const = 0;

  /// Returns a map from signature name to SignatureDef for all signatures in
  /// in the SavedModel.
  virtual const protobuf::Map<string, SignatureDef>& GetSignatures() const = 0;
};

/// SavedModel representation once the SavedModel is loaded from storage.
///
/// NOTE: Prefer to use SavedModelBundleLite in new code, as it consumes less
/// RAM.
struct SavedModelBundle : public SavedModelBundleInterface {
  /// A TensorFlow Session does not Close itself on destruction. To avoid
  /// resource leaks, we explicitly call Close on Sessions that we create.
  ~SavedModelBundle() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh mht_0(mht_0_v, 221, "", "./tensorflow/cc/saved_model/loader.h", "~SavedModelBundle");

    if (session) {
      session->Close().IgnoreError();
    }
  }

  SavedModelBundle() = default;

  Session* GetSession() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh mht_1(mht_1_v, 232, "", "./tensorflow/cc/saved_model/loader.h", "GetSession");
 return session.get(); }
  const protobuf::Map<string, SignatureDef>& GetSignatures() const override {
    return meta_graph_def.signature_def();
  }

  std::unique_ptr<Session> session;
  MetaGraphDef meta_graph_def;
  std::unique_ptr<GraphDebugInfo> debug_info;
};

// A version of SavedModelBundle that avoids storing a potentially large
// MetaGraphDef. Prefer to use SavedModelBundleLite in new code.
class SavedModelBundleLite : public SavedModelBundleInterface {
 public:
  SavedModelBundleLite() = default;
  SavedModelBundleLite(SavedModelBundleLite&& other) = default;
  SavedModelBundleLite& operator=(SavedModelBundleLite&& other) = default;

  SavedModelBundleLite(std::unique_ptr<Session> session,
                       protobuf::Map<string, SignatureDef> signatures)
      : session_(std::move(session)), signatures_(std::move(signatures)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh mht_2(mht_2_v, 255, "", "./tensorflow/cc/saved_model/loader.h", "SavedModelBundleLite");
}

  /// A TensorFlow Session does not Close itself on destruction. To avoid
  /// resource leaks, we explicitly call Close on Sessions that we create.
  ~SavedModelBundleLite() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh mht_3(mht_3_v, 262, "", "./tensorflow/cc/saved_model/loader.h", "~SavedModelBundleLite");

    if (session_) {
      session_->Close().IgnoreError();
    }
  }

  Session* GetSession() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSloaderDTh mht_4(mht_4_v, 271, "", "./tensorflow/cc/saved_model/loader.h", "GetSession");
 return session_.get(); }
  const protobuf::Map<string, SignatureDef>& GetSignatures() const override {
    return signatures_;
  }

 private:
  std::unique_ptr<Session> session_;
  protobuf::Map<string, SignatureDef> signatures_;
};

// Restore variable and resources in the SavedModel export dir for the
// indicated metagraph.
// The recommended way to load a saved model is to call LoadSavedModel,
// which provides an already initialized Metagraph, Session, and DebugInfo.
Status RestoreSession(const RunOptions& run_options,
                      const MetaGraphDef& meta_graph, const string& export_dir,
                      std::unique_ptr<Session>* session);

// Initialize a session which wraps this metagraph.
// The recommended way to load a saved model is to call LoadSavedModel,
// which provides an already initialized Metagraph, Session, and DebugInfo.
Status LoadMetagraphIntoSession(const SessionOptions& session_options,
                                const MetaGraphDef& meta_graph,
                                std::unique_ptr<Session>* session);

/// Loads a SavedModel from the specified export directory. The MetaGraphDef
/// to be loaded is identified by the supplied tags, corresponding exactly to
/// the set of tags used at SavedModel build time. Stores a SavedModel bundle in
/// *bundle with a session and the requested MetaGraphDef, if found.
///
/// NOTE: Prefer the overload that takes a SavedModelBundleLite* in new code.
Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundle* const bundle);

/// Loads a SavedModel from the specified export directory. The MetaGraphDef
/// to be loaded is identified by the supplied tags, corresponding exactly to
/// the set of tags used at SavedModel build time. Stores a SavedModel bundle
/// in *bundle with a session created from the requested MetaGraphDef if found.
///
/// This overload creates a SavedModelBundleLite, which consumes less RAM than
/// an equivalent SavedModelBundle.
Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const string& export_dir,
                      const std::unordered_set<string>& tags,
                      SavedModelBundleLite* const bundle);

/// Checks whether the provided directory could contain a SavedModel. Note that
/// the method does not load any data by itself. If the method returns `false`,
/// the export directory definitely does not contain a SavedModel. If the method
/// returns `true`, the export directory may contain a SavedModel but provides
/// no guarantee that it can be loaded.
bool MaybeSavedModelDirectory(const std::string& export_dir);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_LOADER_H_
