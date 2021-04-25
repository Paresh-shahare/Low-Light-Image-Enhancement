import ctypes
import imp
import sys

def main():
  try:
    import tensorflow as tf
    print("TensorFlow successfully installed.")
    if tf.test.is_built_with_cuda():
      print("The installed version of TensorFlow includes GPU support.")
    else:
      print("The installed version of TensorFlow does not include GPU support.")
    sys.exit(0)
  except ImportError:
    print("ERROR: Failed to import the TensorFlow module.")

  candidate_explanation = False

  python_version = sys.version_info.major, sys.version_info.minor

  print("\n- Python version is %d.%d." % python_version)

  if python_version != (3, 5):

    candidate_explanation = True

    print("- The official distribution of TensorFlow for Windows requires "

          "Python version 3.5.")

  

  try:

    _, pathname, _ = imp.find_module("tensorflow")

    print("\n- TensorFlow is installed at: %s" % pathname)

  except ImportError:

    candidate_explanation = False

    print("""
- No module named TensorFlow is installed in this Python environment. You may
  install it using the command `pip install tensorflow`.""")

    

  try:

    msvcp140 = ctypes.WinDLL("msvcp140.dll")

  except OSError:

    candidate_explanation = True

    print("""
- Could not load 'msvcp140.dll'. TensorFlow requires that this DLL be
  installed in a directory that is named in your %PATH% environment
  variable. You may install this DLL by downloading Microsoft Visual
  C++ 2015 Redistributable Update 3 from this URL:
  https://www.microsoft.com/en-us/download/details.aspx?id=53587""")



  try:

    cudart64_80 = ctypes.WinDLL("cudart64_80.dll")

  except OSError:

    candidate_explanation = True

    print("""
- Could not load 'cudart64_80.dll'. The GPU version of TensorFlow
  requires that this DLL be installed in a directory that is named in
  your %PATH% environment variable. Download and install CUDA 8.0 from
  this URL: https://developer.nvidia.com/cuda-toolkit""")



  try:

    cudnn = ctypes.WinDLL("cudnn64_5.dll")

  except OSError:

    candidate_explanation = True

    print("""
- Could not load 'cudnn64_5.dll'. The GPU version of TensorFlow
  requires that this DLL be installed in a directory that is named in
  your %PATH% environment variable. Note that installing cuDNN is a
  separate step from installing CUDA, and it is often found in a
  different directory from the CUDA DLLs. You may install the
  necessary DLL by downloading cuDNN 5.1 from this URL:
  https://developer.nvidia.com/cudnn""")



  if not candidate_explanation:

    print("""
- All required DLLs are present. Please open an issue on the
  TensorFlow GitHub page: https://github.com/tensorflow/tensorflow/issues""")



  sys.exit(-1)



if __name__ == "__main__":

  main()