# MIT License
#
# Copyright (c) 2021 Cyclip
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modifed based on https://github.com/Cyclip/SimpleCache

import hashlib  # Hash filenames
from pathlib import Path  # Cache dir
from functools import wraps  # Decorators
import zlib  # Compression for file sizes
import pickle  # Store objects
import shutil  # Clear cache faster
import logging
import os
from secrets import token_hex


class Cache:
    """
    Class to implement a simple pure-python caching system with files, without
    using up too much memory.

    Parameters:
        cacheDir (str)                      :   Specify where the cache will be stored.
                                                Default: "cache"
                                                This attribute can be accessed, but should not be changed.

        algorithm (function)                :   Hashing algorithm from hashlib
                                                Default: hashlib.sha256
                                                This attribute can be accessed, but should not be changed.

        startEmpty (bool)                   :   Start with an empty cache.
                                                WILL ERASE ALL PREVIOUS CACHE obviously
                                                Default: False

    Functions:
        clear()                             :   Clear cache

    Decorators:
        @cache_function
            Cache results of function based on arguments
    """

    def __init__(
        self,
        cacheDir="cache",
        algorithm=hashlib.sha256,
        startEmpty=False,
        log=False,
    ):
        """
        Constructs system with necessary attributes.

        Parameters:
            cacheDir (str)                      :   Specify where the cache will be stored.
                                                    Default: "cache"
                                                    This attribute can be accessed, but should not be changed.

            algorithm (function)                :   Hashing algorithm from hashlib
                                                    Default: hashlib.sha256
                                                    This attribute can be accessed, but should not be changed.

            startEmpty (bool)                   :   Start with an empty cache.
                                                    WILL ERASE ALL PREVIOUS CACHE obviously
                                                    Default: False
        """
        self.cacheDir = cacheDir
        self.algorithm = algorithm
        Path(self.cacheDir).mkdir(exist_ok=True)

        if startEmpty:
            self.clear()

    def cache_function(self, func):
        """
        Implements the cache system onto a function.

        No parameters, nor arguments as all settings are set during construction.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            fileName = self._build_file_name(func, args, kwargs)

            try:
                return self._read_cache(fileName)
            except FileNotFoundError:
                pass

            returnVal = func(*args, **kwargs)
            self._write_cache(fileName, returnVal)
            return returnVal

        return wrapper

    def clear(self):
        """
        Clear cache for this Cache() instance.
        If 2 instances share the same directory, it will
        affect both instances.
        """
        shutil.rmtree(self.cacheDir)  # Remoeve the cache directory
        os.mkdir(self.cacheDir)  # Create cache dir again

    def _build_file_name(self, func, args, kwargs):
        h = self.algorithm()
        def update(obj):
            pickled = pickle.dumps(obj)
            h.update(len(pickled).to_bytes(8, 'big'))
            h.update(pickled)

        h.update((func.__module__ + "." + func.__qualname__).encode() + b'\0')
        update(func.__defaults__ if hasattr(func, '__defaults__') else b'')
        update(func.__kwdefaults__ if hasattr(func, '__kwdefaults__') else b'')
        update(func.__self__ if hasattr(func, '__self__') else b'')
        update(args)
        update(kwargs)

        fname = func.__name__ + "_" + h.hexdigest()

        pathToFile = os.path.join(self.cacheDir, fname)
        return pathToFile

    def _read_cache(self, fileName):
        """
        Retrieve a file contents, decompress and extract python objects and
        return them.

        Arguments:
            fileName (str)                          Path to the cache file which is being read

        Returns:
            variables (mixed)                       Variable name is literally 'variables'. Returns python
                                                    objects of an unknown type.
        """
        # Cache hit

        with open(fileName, "rb") as f:
            return pickle.load(f)

    def _write_cache(self, fileName, returnVal):
        """
        Dump python objects into an encoded string, compress and write to
        cache.

        Parameters:
            fileName (str)                          Path to the cache file which will be written in

            returnVal (mixed)                       The function's return value to write into cache
        """

        tmpfile = token_hex() + '.tmp'
        with open(tmpfile, 'wb') as f:
            packed = pickle.dump(returnVal, f)
        os.rename(tmpfile, fileName)
