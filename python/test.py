import ctypes
import scipy.linalg.cython_blas as blas

# PyCapsule オブジェクトを取得
caps = blas.__pyx_capi__.get("dgemm")  # または blas.__pyx_capi__["isamax"]
if caps is None:
    raise RuntimeError("No PyCapsule for isamax")

# ctypes.pythonapi を使って PyCapsule_GetName / PyCapsule_GetPointer を設定
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
name = ctypes.pythonapi.PyCapsule_GetName(caps)

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ptr = ctypes.pythonapi.PyCapsule_GetPointer(caps, name)

print("Pointer for isamax (C function) is:", hex(ptr))

if ptr:
    class Dl_info(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]
    libc = ctypes.CDLL(None)
    dladdr = libc.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Dl_info)]
    dladdr.restype = ctypes.c_int
    info = Dl_info()
    res = dladdr(ctypes.c_void_p(ptr), ctypes.byref(info))
    if res:
        print("Shared lib for isamax:", info.dli_fname.decode())
    else:
        print("dladdr failed.")
else:
    print("Pointer is NULL or could not get from capsule.")
