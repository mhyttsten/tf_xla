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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_tracker_jniDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_tracker_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_tracker_jniDTcc() {
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

#include <android/log.h>
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/jni_utils.h"
#include "tensorflow/tools/android/test/jni/object_tracking/object_tracker.h"
#include "tensorflow/tools/android/test/jni/object_tracking/time_log.h"

namespace tf_tracking {

#define OBJECT_TRACKER_METHOD(METHOD_NAME) \
  Java_org_tensorflow_demo_tracking_ObjectTracker_##METHOD_NAME  // NOLINT

JniLongField object_tracker_field("nativeObjectTracker");

ObjectTracker* get_object_tracker(JNIEnv* env, jobject thiz) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_tracker_jniDTcc mht_0(mht_0_v, 207, "", "./tensorflow/tools/android/test/jni/object_tracking/object_tracker_jni.cc", "get_object_tracker");

  ObjectTracker* const object_tracker =
      reinterpret_cast<ObjectTracker*>(object_tracker_field.get(env, thiz));
  CHECK_ALWAYS(object_tracker != NULL, "null object tracker!");
  return object_tracker;
}

void set_object_tracker(JNIEnv* env, jobject thiz,
                        const ObjectTracker* object_tracker) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_tracker_jniDTcc mht_1(mht_1_v, 218, "", "./tensorflow/tools/android/test/jni/object_tracking/object_tracker_jni.cc", "set_object_tracker");

  object_tracker_field.set(env, thiz,
                           reinterpret_cast<intptr_t>(object_tracker));
}

#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(initNative)(JNIEnv* env, jobject thiz,
                                               jint width, jint height,
                                               jboolean always_track);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(releaseMemoryNative)(JNIEnv* env,
                                                        jobject thiz);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(registerNewObjectWithAppearanceNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jbyteArray frame_data);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setPreviousPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jlong timestamp);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2);

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(haveObject)(JNIEnv* env, jobject thiz,
                                                   jstring object_id);

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(isObjectVisible)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id);

JNIEXPORT
jstring JNICALL OBJECT_TRACKER_METHOD(getModelIdNative)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id);

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getCurrentCorrelation)(JNIEnv* env,
                                                            jobject thiz,
                                                            jstring object_id);

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getMatchScore)(JNIEnv* env, jobject thiz,
                                                    jstring object_id);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getTrackedPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloatArray rect_array);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(nextFrameNative)(JNIEnv* env, jobject thiz,
                                                    jbyteArray y_data,
                                                    jbyteArray uv_data,
                                                    jlong timestamp,
                                                    jfloatArray vg_matrix_2x3);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(forgetNative)(JNIEnv* env, jobject thiz,
                                                 jstring object_id);

JNIEXPORT
jbyteArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsPacked)(
    JNIEnv* env, jobject thiz, jfloat scale_factor);

JNIEXPORT
jfloatArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsNative)(
    JNIEnv* env, jobject thiz, jboolean only_found_);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jlong timestamp, jfloat position_x1,
    jfloat position_y1, jfloat position_x2, jfloat position_y2,
    jfloatArray delta);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(drawNative)(JNIEnv* env, jobject obj,
                                               jint view_width,
                                               jint view_height,
                                               jfloatArray delta);

JNIEXPORT void JNICALL OBJECT_TRACKER_METHOD(downsampleImageNative)(
    JNIEnv* env, jobject thiz, jint width, jint height, jint row_stride,
    jbyteArray input, jint factor, jbyteArray output);

#ifdef __cplusplus
}
#endif

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(initNative)(JNIEnv* env, jobject thiz,
                                               jint width, jint height,
                                               jboolean always_track) {
  LOGI("Initializing object tracker. %dx%d @%p", width, height, thiz);
  const Size image_size(width, height);
  TrackerConfig* const tracker_config = new TrackerConfig(image_size);
  tracker_config->always_track = always_track;

  // XXX detector
  ObjectTracker* const tracker = new ObjectTracker(tracker_config, NULL);
  set_object_tracker(env, thiz, tracker);
  LOGI("Initialized!");

  CHECK_ALWAYS(get_object_tracker(env, thiz) == tracker,
               "Failure to set hand tracker!");
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(releaseMemoryNative)(JNIEnv* env,
                                                        jobject thiz) {
  delete get_object_tracker(env, thiz);
  set_object_tracker(env, thiz, NULL);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(registerNewObjectWithAppearanceNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jbyteArray frame_data) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  LOGI("Registering the position of %s at %.2f,%.2f,%.2f,%.2f", id_str, x1, y1,
       x2, y2);

  jboolean iCopied = JNI_FALSE;

  // Copy image into currFrame.
  jbyte* pixels = env->GetByteArrayElements(frame_data, &iCopied);

  BoundingBox bounding_box(x1, y1, x2, y2);
  get_object_tracker(env, thiz)->RegisterNewObjectWithAppearance(
      id_str, reinterpret_cast<const uint8_t*>(pixels), bounding_box);

  env->ReleaseByteArrayElements(frame_data, pixels, JNI_ABORT);

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setPreviousPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jlong timestamp) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  LOGI(
      "Registering the position of %s at %.2f,%.2f,%.2f,%.2f"
      " at time %lld",
      id_str, x1, y1, x2, y2, static_cast<long long>(timestamp));

  get_object_tracker(env, thiz)->SetPreviousPositionOfObject(
      id_str, BoundingBox(x1, y1, x2, y2), timestamp);

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  LOGI("Registering the position of %s at %.2f,%.2f,%.2f,%.2f", id_str, x1, y1,
       x2, y2);

  get_object_tracker(env, thiz)->SetCurrentPositionOfObject(
      id_str, BoundingBox(x1, y1, x2, y2));

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(haveObject)(JNIEnv* env, jobject thiz,
                                                   jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const bool haveObject = get_object_tracker(env, thiz)->HaveObject(id_str);
  env->ReleaseStringUTFChars(object_id, id_str);
  return haveObject;
}

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(isObjectVisible)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const bool visible = get_object_tracker(env, thiz)->IsObjectVisible(id_str);
  env->ReleaseStringUTFChars(object_id, id_str);
  return visible;
}

JNIEXPORT
jstring JNICALL OBJECT_TRACKER_METHOD(getModelIdNative)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);
  const TrackedObject* const object =
      get_object_tracker(env, thiz)->GetObject(id_str);
  env->ReleaseStringUTFChars(object_id, id_str);
  jstring model_name = env->NewStringUTF(object->GetModel()->GetName().c_str());
  return model_name;
}

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getCurrentCorrelation)(JNIEnv* env,
                                                            jobject thiz,
                                                            jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const float correlation =
      get_object_tracker(env, thiz)->GetObject(id_str)->GetCorrelation();
  env->ReleaseStringUTFChars(object_id, id_str);
  return correlation;
}

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getMatchScore)(JNIEnv* env, jobject thiz,
                                                    jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const float match_score =
      get_object_tracker(env, thiz)->GetObject(id_str)->GetMatchScore().value;
  env->ReleaseStringUTFChars(object_id, id_str);
  return match_score;
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getTrackedPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloatArray rect_array) {
  jboolean iCopied = JNI_FALSE;
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const BoundingBox bounding_box =
      get_object_tracker(env, thiz)->GetObject(id_str)->GetPosition();
  env->ReleaseStringUTFChars(object_id, id_str);

  jfloat* rect = env->GetFloatArrayElements(rect_array, &iCopied);
  bounding_box.CopyToArray(reinterpret_cast<float*>(rect));
  env->ReleaseFloatArrayElements(rect_array, rect, 0);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(nextFrameNative)(JNIEnv* env, jobject thiz,
                                                    jbyteArray y_data,
                                                    jbyteArray uv_data,
                                                    jlong timestamp,
                                                    jfloatArray vg_matrix_2x3) {
  TimeLog("Starting object tracker");

  jboolean iCopied = JNI_FALSE;

  float vision_gyro_matrix_array[6];
  jfloat* jmat = NULL;

  if (vg_matrix_2x3 != NULL) {
    // Copy the alignment matrix into a float array.
    jmat = env->GetFloatArrayElements(vg_matrix_2x3, &iCopied);
    for (int i = 0; i < 6; ++i) {
      vision_gyro_matrix_array[i] = static_cast<float>(jmat[i]);
    }
  }
  // Copy image into currFrame.
  jbyte* pixels = env->GetByteArrayElements(y_data, &iCopied);
  jbyte* uv_pixels =
      uv_data != NULL ? env->GetByteArrayElements(uv_data, &iCopied) : NULL;

  TimeLog("Got elements");

  // Add the frame to the object tracker object.
  get_object_tracker(env, thiz)->NextFrame(
      reinterpret_cast<uint8_t*>(pixels), reinterpret_cast<uint8_t*>(uv_pixels),
      timestamp, vg_matrix_2x3 != NULL ? vision_gyro_matrix_array : NULL);

  env->ReleaseByteArrayElements(y_data, pixels, JNI_ABORT);

  if (uv_data != NULL) {
    env->ReleaseByteArrayElements(uv_data, uv_pixels, JNI_ABORT);
  }

  if (vg_matrix_2x3 != NULL) {
    env->ReleaseFloatArrayElements(vg_matrix_2x3, jmat, JNI_ABORT);
  }

  TimeLog("Released elements");

  PrintTimeLog();
  ResetTimeLog();
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(forgetNative)(JNIEnv* env, jobject thiz,
                                                 jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  get_object_tracker(env, thiz)->ForgetTarget(id_str);

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
jfloatArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsNative)(
    JNIEnv* env, jobject thiz, jboolean only_found) {
  jfloat keypoint_arr[kMaxKeypoints * kKeypointStep];

  const int number_of_keypoints =
      get_object_tracker(env, thiz)->GetKeypoints(only_found, keypoint_arr);

  // Create and return the array that will be passed back to Java.
  jfloatArray keypoints =
      env->NewFloatArray(number_of_keypoints * kKeypointStep);
  if (keypoints == NULL) {
    LOGE("null array!");
    return NULL;
  }
  env->SetFloatArrayRegion(keypoints, 0, number_of_keypoints * kKeypointStep,
                           keypoint_arr);

  return keypoints;
}

JNIEXPORT
jbyteArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsPacked)(
    JNIEnv* env, jobject thiz, jfloat scale_factor) {
  // 2 bytes to a uint16_t and two pairs of xy coordinates per keypoint.
  const int bytes_per_keypoint = sizeof(uint16_t) * 2 * 2;
  jbyte keypoint_arr[kMaxKeypoints * bytes_per_keypoint];

  const int number_of_keypoints =
      get_object_tracker(env, thiz)->GetKeypointsPacked(
          reinterpret_cast<uint16_t*>(keypoint_arr), scale_factor);

  // Create and return the array that will be passed back to Java.
  jbyteArray keypoints =
      env->NewByteArray(number_of_keypoints * bytes_per_keypoint);

  if (keypoints == NULL) {
    LOGE("null array!");
    return NULL;
  }

  env->SetByteArrayRegion(
      keypoints, 0, number_of_keypoints * bytes_per_keypoint, keypoint_arr);

  return keypoints;
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jlong timestamp, jfloat position_x1,
    jfloat position_y1, jfloat position_x2, jfloat position_y2,
    jfloatArray delta) {
  jfloat point_arr[4];

  const BoundingBox new_position = get_object_tracker(env, thiz)->TrackBox(
      BoundingBox(position_x1, position_y1, position_x2, position_y2),
      timestamp);

  new_position.CopyToArray(point_arr);
  env->SetFloatArrayRegion(delta, 0, 4, point_arr);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(drawNative)(
    JNIEnv* env, jobject thiz, jint view_width, jint view_height,
    jfloatArray frame_to_canvas_arr) {
  ObjectTracker* object_tracker = get_object_tracker(env, thiz);
  if (object_tracker != NULL) {
    jfloat* frame_to_canvas =
        env->GetFloatArrayElements(frame_to_canvas_arr, NULL);

    object_tracker->Draw(view_width, view_height, frame_to_canvas);
    env->ReleaseFloatArrayElements(frame_to_canvas_arr, frame_to_canvas,
                                   JNI_ABORT);
  }
}

JNIEXPORT void JNICALL OBJECT_TRACKER_METHOD(downsampleImageNative)(
    JNIEnv* env, jobject thiz, jint width, jint height, jint row_stride,
    jbyteArray input, jint factor, jbyteArray output) {
  if (input == NULL || output == NULL) {
    LOGW("Received null arrays, hopefully this is a test!");
    return;
  }

  jbyte* const input_array = env->GetByteArrayElements(input, 0);
  jbyte* const output_array = env->GetByteArrayElements(output, 0);

  {
    tf_tracking::Image<uint8_t> full_image(
        width, height, reinterpret_cast<uint8_t*>(input_array), false);

    const int new_width = (width + factor - 1) / factor;
    const int new_height = (height + factor - 1) / factor;

    tf_tracking::Image<uint8_t> downsampled_image(
        new_width, new_height, reinterpret_cast<uint8_t*>(output_array), false);

    downsampled_image.DownsampleAveraged(
        reinterpret_cast<uint8_t*>(input_array), row_stride, factor);
  }

  env->ReleaseByteArrayElements(input, input_array, JNI_ABORT);
  env->ReleaseByteArrayElements(output, output_array, 0);
}

}  // namespace tf_tracking
