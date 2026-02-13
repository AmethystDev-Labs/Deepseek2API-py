import ctypes
import os
from importlib import resources
import wasmtime

ASSET_WASM_NAME = "sha3_wasm_bg.7b9ca65ddd.wasm"

def resolve_wasm_path() -> str:
    try:
        pkg = resources.files("deepseek_client") / "assets" / ASSET_WASM_NAME
        # If packaged as a file, return its filesystem path
        if pkg.exists():
            with resources.as_file(pkg) as p:
                return str(p)
    except Exception:
        pass
    # Fallback to local assets directory
    base_dir = os.path.dirname(__file__)
    assets_path = os.path.join(base_dir, "assets", ASSET_WASM_NAME)
    if os.path.exists(assets_path):
        return assets_path
    
    # Fallback to project root (deprecated)
    root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ASSET_WASM_NAME)
    return root_path

class DeepSeekPoW:
    def __init__(self, wasm_path: str | None = None):
        self.store = wasmtime.Store()
        path = wasm_path or resolve_wasm_path()
        module = wasmtime.Module.from_file(self.store.engine, path)
        self.instance = wasmtime.Instance(self.store, module, [])
        self.memory = self.instance.exports(self.store)["memory"]
        self.malloc = self.instance.exports(self.store)["__wbindgen_export_0"]
        self.free = self.instance.exports(self.store)["__wbindgen_export_2"]
        self.solve_func = self.instance.exports(self.store)["wasm_solve"]
        self.hash_func = self.instance.exports(self.store)["wasm_deepseek_hash_v1"]

    def write_bytes(self, b: bytes):
        size = len(b)
        ptr = self.malloc(self.store, size, 1)
        mem_ptr = self.memory.data_ptr(self.store)
        mem_view = (ctypes.c_ubyte * self.memory.data_len(self.store)).from_address(ctypes.addressof(mem_ptr.contents))
        for i, val in enumerate(b):
            mem_view[ptr + i] = val
        return ptr, size

    def read_bytes(self, ptr: int, size: int):
        mem_ptr = self.memory.data_ptr(self.store)
        mem_view = (ctypes.c_ubyte * self.memory.data_len(self.store)).from_address(ctypes.addressof(mem_ptr.contents))
        data = bytearray(size)
        for i in range(size):
            data[i] = mem_view[ptr + i]
        return data

    def solve(self, c_bytes: bytes, s_bytes: bytes, num_arg: float):
        ret_ptr_loc = self.malloc(self.store, 16, 8)
        mem_ptr = self.memory.data_ptr(self.store)
        mem_view = (ctypes.c_ubyte * self.memory.data_len(self.store)).from_address(ctypes.addressof(mem_ptr.contents))
        for i in range(16):
            mem_view[ret_ptr_loc + i] = 0

        c_ptr, c_len = self.write_bytes(c_bytes)
        s_ptr, s_len = self.write_bytes(s_bytes)
        self.solve_func(self.store, ret_ptr_loc, c_ptr, c_len, s_ptr, s_len, float(num_arg))

        mem_ptr = self.memory.data_ptr(self.store)
        mem_view = (ctypes.c_ubyte * self.memory.data_len(self.store)).from_address(ctypes.addressof(mem_ptr.contents))
        status = int.from_bytes(bytearray(mem_view[ret_ptr_loc : ret_ptr_loc+4]), "little", signed=True)
        if status == 0:
            return None
        raw = bytes(mem_view[ret_ptr_loc+8 : ret_ptr_loc+16])
        value = ctypes.c_double.from_buffer_copy(raw).value
        try:
            return int(value)
        except Exception:
            return None

    def hash(self, data_bytes: bytes):
        ret_ptr_loc = self.malloc(self.store, 8, 4)
        mem_ptr = self.memory.data_ptr(self.store)
        mem_view = (ctypes.c_ubyte * self.memory.data_len(self.store)).from_address(ctypes.addressof(mem_ptr.contents))
        for i in range(8):
            mem_view[ret_ptr_loc + i] = 0
        d_ptr, d_len = self.write_bytes(data_bytes)
        try:
            self.hash_func(self.store, ret_ptr_loc, d_ptr, d_len)
        except Exception as e:
            print(f"Error calling hash_func: {e}")
            return None
        mem_ptr = self.memory.data_ptr(self.store)
        mem_view = (ctypes.c_ubyte * self.memory.data_len(self.store)).from_address(ctypes.addressof(mem_ptr.contents))
        res_ptr = int.from_bytes(bytearray(mem_view[ret_ptr_loc : ret_ptr_loc+4]), "little")
        res_len = int.from_bytes(bytearray(mem_view[ret_ptr_loc+4 : ret_ptr_loc+8]), "little")
        if res_len > 0:
            return self.read_bytes(res_ptr, res_len)
        return None

    def solve_challenge(self, algorithm: str, challenge_str: str, salt: str, difficulty: int, expire_at: int):
        if algorithm != "DeepSeekHashV1":
            raise ValueError("Unsupported algorithm")
        prefix = f"{salt}_{expire_at}_"
        c_bytes = challenge_str.encode("utf-8")
        p_bytes = prefix.encode("utf-8")
        res = self.solve(c_bytes, p_bytes, float(difficulty))
        if isinstance(res, int):
            return res
        return None

