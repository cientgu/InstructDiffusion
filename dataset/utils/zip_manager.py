import zipfile
import os.path as osp
# import lmdb
import logging
from PIL import Image
import pickle
import io
import glob
import os
from pathlib import Path
import time
from threading import Thread
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

home = str(Path.home())
abs_blob_path=os.path.realpath("/mnt/blob/")
CACHE_FOLDER=os.path.join(home,"caching")
USE_CACHE=True

def norm(path):
    assert "*" not in path
    return os.path.realpath(os.path.abspath(path))

def in_blob(file):
    if abs_blob_path in file:
        return True
    else:
        return False

def map_name(file):
    path=norm(file)
    path=path.lstrip(abs_blob_path+"/")
    path=path.replace("/","_")
    assert len(path)<250
    return path


def preload(db,sync=False):
    if sync:
        db.initialize()
    else:
        p = Thread(target=db.initialize)
        p.start()

def get_keys_from_lmdb(db):
    with db.begin(write=False) as txn:
        return list(txn.cursor().iternext(values=False))

def decode_img(byteflow):
    try:
        img=Image.open(io.BytesIO(byteflow)).convert("RGB")
        img.load()
    except:
        img = Image.open("white.jpeg").convert("RGB")
        img.load()
    return img

def decode_text(byteflow):
    return pickle.loads(byteflow)
    
decode_funcs={
    "image": decode_img,
    "text": decode_text
}


class ZipManager:
    def __init__(self, zip_path,data_type,prefix=None) -> None:
        self.decode_func=decode_funcs[data_type]
        self.zip_path=zip_path
        self._init=False
        preload(self)
        
    def deinitialze(self):
        self.zip_fd.close()
        del self.zip_fd
        self._init = False

    def initialize(self,close=True):
        self.zip_fd = zipfile.ZipFile(self.zip_path, mode="r")
        if not hasattr(self,"_keys"):
            self._keys = self.zip_fd.namelist()
        self._init = True
        if close:
            self.deinitialze()
        
    @property
    def keys(self):
        while not hasattr(self,"_keys"):
            time.sleep(0.1)
        return self._keys

    def get(self, name):
        if not self._init:
            self.initialize(close=False)  
        byteflow = self.zip_fd.read(name)
        return self.decode_func(byteflow)


class MultipleZipManager:
    def __init__(self, files: list, data_type, sync=True):
        self.files = files
        self._is_init = False
        self.data_type=data_type
        if sync:
            print("sync",files)
            self.initialize()
        else:
            print("async",files)
            preload(self)
        print("initialize over")
        

    def initialize(self):
        self.mapping={}
        self.managers={}
        for file in self.files:
            manager = ZipManager(file, self.data_type)
            self.managers[file]=manager

        for file,manager in self.managers.items():
            print(file)
            # print("loading")
            logging.info(f"{file} loading")
            keys=manager.keys
            for key in keys:
                self.mapping[key]=file
            logging.info(f"{file} loaded, size = {len(keys)}")
            print("loaded")

        self._keys=list(self.mapping.keys())
        self._is_init=True

    @property
    def keys(self):
        while not self._is_init:
            time.sleep(0.1)
        return self._keys
        
    def get(self, name):
        data = self.managers[self.mapping[name]].get(name)
        return data

