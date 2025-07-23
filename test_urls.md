# Python `json` Module: JSON Encoder and Decoder

The `json` module in Python provides an API for working with JSON (JavaScript Object Notation), a lightweight data interchange format. JSON is specified by [RFC 7159](https://datatracker.ietf.org/doc/html/rfc7159.html) (which obsoletes RFC 4627) and [ECMA-404](https://ecma-international.org/publications-and-standards/standards/ecma-404/).

**Note on "Object" Terminology:** In the context of JSON processing in Python, the term "object" can be ambiguous. While all values in Python are objects, in JSON, an "object" specifically refers to data wrapped in curly braces, similar to a Python dictionary.

**Security Warning:** Be cautious when parsing JSON data from untrusted sources. Malicious JSON strings can consume significant CPU and memory resources. Limiting the size of data to be parsed is highly recommended.

The `json` module's API is similar to Python's standard library `marshal` and `pickle` modules for serialization.

**JSON and YAML:** JSON is a subset of YAML 1.2. The JSON produced by this module's default settings (especially the default `separators` value) is also a subset of YAML 1.0 and 1.1. This means the `json` module can also be used as a YAML serializer.

**Order Preservation:** This module's encoders and decoders preserve input and output order by default. Order is only lost if the underlying Python containers (e.g., `dict` before Python 3.7, or `set`) are unordered.

## Basic Usage Examples

### Encoding Python Objects to JSON

```python
import json

# Basic encoding
print(json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}]))
# Output: '["foo", {"bar": ["baz", null, 1.0, 2]}]'

# Escaping special characters
print(json.dumps("\"foo\bar"))
# Output: "\"foo\\bar"
print(json.dumps('\u1234'))
# Output: "\u1234"
print(json.dumps('\\'))
# Output: "\\"

# Sorting keys for consistent output
print(json.dumps({"c": 0, "b": 0, "a": 0}, sort_keys=True))
# Output: '{"a": 0, "b": 0, "c": 0}'

# Streaming API with json.dump
from io import StringIO
io = StringIO()
json.dump(['streaming API'], io)
print(io.getvalue())
# Output: '["streaming API"]'
```

### Compact Encoding

```python
import json
print(json.dumps([1, 2, 3, {'4': 5, '6': 7}], separators=(',', ':')))
# Output: '[1,2,3,{"4":5,"6":7}]'
```

### Pretty Printing

```python
import json
print(json.dumps({'6': 7, '4': 5}, sort_keys=True, indent=4))
# Output:
# {
#     "4": 5,
#     "6": 7
# }
```

### Customizing JSON Object Encoding

```python
import json

def custom_json(obj):
    if isinstance(obj, complex):
        return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
    raise TypeError(f'Cannot serialize object of {type(obj)}')

print(json.dumps(1 + 2j, default=custom_json))
# Output: '{"__complex__": true, "real": 1.0, "imag": 2.0}'
```

### Decoding JSON to Python Objects

```python
import json

# Basic decoding
print(json.loads('["foo", {"bar":["baz", null, 1.0, 2]}]'))
# Output: ['foo', {'bar': ['baz', None, 1.0, 2]}]

# Decoding escaped characters
print(json.loads('"\\"foo\\bar"'))
# Output: '"foo\x08ar'

# Streaming API with json.load
from io import StringIO
io = StringIO('["streaming API"]')
print(json.load(io))
# Output: ['streaming API']
```

### Customizing JSON Object Decoding

```python
import json
import decimal

def as_complex(dct):
    if '__complex__' in dct:
        return complex(dct['real'], dct['imag'])
    return dct

print(json.loads('{"__complex__": true, "real": 1, "imag": 2}',
                 object_hook=as_complex))
# Output: (1+2j)

print(json.loads('1.1', parse_float=decimal.Decimal))
# Output: Decimal('1.1')
```

### Extending `JSONEncoder`

```python
import json

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        # Let the base class default method raise the TypeError
        return super().default(obj)

print(json.dumps(2 + 1j, cls=ComplexEncoder))
# Output: '[2.0, 1.0]'

print(ComplexEncoder().encode(2 + 1j))
# Output: '[2.0, 1.0]'

print(list(ComplexEncoder().iterencode(2 + 1j)))
# Output: ['[2.0', ', 1.0', ']']
```

## Core Functions

### `json.dump(obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)`

Serializes `obj` as a JSON formatted stream to `fp` (a `.write()`-supporting file-like object).

**Parameters:**
*   **`obj`** (`object`): The Python object to be serialized.
*   **`fp`** (`file-like object`): The file-like object to which `obj` will be serialized. `fp.write()` must support `str` input, as the `json` module always produces `str` objects.
*   **`skipkeys`** (`bool`): If `True`, keys that are not of a basic type (`str`, `int`, `float`, `bool`, `None`) will be skipped instead of raising a `TypeError`. Default is `False`.
*   **`ensure_ascii`** (`bool`): If `True` (default), non-ASCII characters are escaped in the output. If `False`, they are outputted as-is.
*   **`check_circular`** (`bool`): If `False`, circular reference checks for container types are skipped. A circular reference will then result in a `RecursionError`. Default is `True`.
*   **`allow_nan`** (`bool`): If `False`, serialization of out-of-range `float` values (`nan`, `inf`, `-inf`) will raise a `ValueError` for strict JSON compliance. If `True` (default), their JavaScript equivalents (`NaN`, `Infinity`, `-Infinity`) are used.
*   **`cls`** (`JSONEncoder` subclass): A custom JSON encoder subclass with an overridden `default()` method for serializing custom datatypes. If `None` (default), `JSONEncoder` is used.
*   **`indent`** (`int` | `str` | `None`): If a positive integer or string, JSON array elements and object members will be pretty-printed with that indent level. An integer indents that many spaces; a string (e.g., `"\t"`) is used for each level. If zero, negative, or `""`, only newlines are inserted. If `None` (default), the most compact representation is used. (Changed in 3.2: allows strings for `indent`).
*   **`separators`** (`tuple` | `None`): A two-tuple: `(item_separator, key_separator)`. If `None` (default), it defaults to `(', ', ': ')` if `indent` is `None`, and `(',', ': ')` otherwise. For the most compact JSON, specify `(',', ':')`. (Changed in 3.4: default for `indent` not `None`).
*   **`default`** (`callable` | `None`): A function called for objects that cannot otherwise be serialized. It should return a JSON encodable version of the object or raise a `TypeError`. If `None` (default), `TypeError` is raised.
*   **`sort_keys`** (`bool`): If `True`, dictionaries will be outputted sorted by key. Default is `False`.

**Note:** Unlike `pickle` and `marshal`, JSON is not a framed protocol. Serializing multiple objects with repeated calls to `dump()` using the same `fp` will result in an invalid JSON file.

### `json.dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)`

Serializes `obj` to a JSON formatted `str`. The arguments have the same meaning as in `json.dump()`.

**Note:** Keys in JSON key/value pairs are always `str`. When a dictionary is converted to JSON, all its keys are coerced to strings. Consequently, `loads(dumps(x)) != x` if `x` has non-string keys.

### `json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)`

Deserializes `fp` to a Python object.

**Parameters:**
*   **`fp`** (`file-like object`): A `.read()`-supporting text file or binary file containing the JSON document. For binary files, the input encoding should be UTF-8, UTF-16, or UTF-32. (Changed in 3.6: `fp` can be a binary file).
*   **`cls`** (`JSONDecoder` subclass): A custom JSON decoder. Additional keyword arguments to `load()` are passed to the constructor of `cls`. If `None` (default), `JSONDecoder` is used.
*   **`object_hook`** (`callable` | `None`): A function called with the result of any JSON object literal decoded (a `dict`). Its return value replaces the `dict`. Useful for custom decoders (e.g., JSON-RPC class hinting). Default is `None`.
*   **`object_pairs_hook`** (`callable` | `None`): A function called with the result of any JSON object literal decoded as an ordered list of pairs. Its return value replaces the `dict`. If `object_hook` is also set, `object_pairs_hook` takes priority. Default is `None`. (Added in 3.1).
*   **`parse_float`** (`callable` | `None`): A function called with the string of every JSON float to be decoded. Default is `float(num_str)`. Can be used to parse floats into custom datatypes (e.g., `decimal.Decimal`).
*   **`parse_int`** (`callable` | `None`): A function called with the string of every JSON int to be decoded. Default is `int(num_str)`. Can be used to parse integers into custom datatypes (e.g., `float`). (Changed in 3.11: `int()` now limits max length of integer string to avoid DoS attacks).
*   **`parse_constant`** (`callable` | `None`): A function called with one of `'-Infinity'`, `'Infinity'`, or `'NaN'`. Can be used to raise an exception for invalid JSON numbers. Default is `None`. (Changed in 3.1: not called on 'null', 'true', 'false' anymore).

**Raises:**
*   `json.JSONDecodeError`: If the data is not a valid JSON document.
*   `UnicodeDecodeError`: If the data is not UTF-8, UTF-16, or UTF-32 encoded.

### `json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)`

Identical to `json.load()`, but deserializes `s` (a `str`, `bytes`, or `bytearray` instance containing a JSON document) instead of a file-like object.

(Changed in 3.6: `s` can be `bytes` or `bytearray`. Input encoding should be UTF-8, UTF-16, or UTF-32. Changed in 3.9: `encoding` keyword argument removed).

## Encoders and Decoders (Classes)

### `class json.JSONDecoder(*, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, strict=True, object_pairs_hook=None)`

A simple JSON decoder.

**Default Python-to-JSON Translations:**

| JSON          | Python  |
| :------------ | :------ |
| `object`      | `dict`  |
| `array`       | `list`  |
| `string`      | `str`   |
| `number (int)`| `int`   |
| `number (real)`| `float` |
| `true`        | `True`  |
| `false`       | `False` |
| `null`        | `None`  |

It also understands `NaN`, `Infinity`, and `-Infinity` as their corresponding `float` values, which is an extension beyond the strict JSON specification.

**Constructor Parameters:**
*   **`object_hook`**: An optional function called with the result of every JSON object decoded (a `dict`). Its return value replaces the `dict`.
*   **`object_pairs_hook`**: An optional function called with the result of every JSON object decoded as an ordered list of pairs. Its return value replaces the `dict`. If `object_hook` is also defined, `object_pairs_hook` takes priority. (Added in 3.1).
*   **`parse_float`**: An optional function called with the string of every JSON float to be decoded. Default is `float(num_str)`.
*   **`parse_int`**: An optional function called with the string of every JSON int to be decoded. Default is `int(num_str)`.
*   **`parse_constant`**: An optional function called with `'-Infinity'`, `'Infinity'`, or `'NaN'`.
*   **`strict`**: If `False` (default is `True`), control characters (0-31 range, including `\t`, `\n`, `\r`, `\0`) are allowed inside strings.

If the data is not a valid JSON document, a `JSONDecodeError` will be raised.

#### `decode(s)`

Returns the Python representation of `s` (a `str` instance containing a JSON document). Raises `JSONDecodeError` if the JSON document is invalid.

#### `raw_decode(s)`

Decodes a JSON document from `s` (a `str` beginning with a JSON document) and returns a 2-tuple: `(Python_representation, index_where_document_ended)`. Useful for decoding JSON from a string with extraneous data at the end.

### `class json.JSONEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)`

An extensible JSON encoder for Python data structures.

**Default Python-to-JSON Translations:**

| Python          | JSON     |
| :-------------- | :------- |
| `dict`          | `object` |
| `list`, `tuple` | `array`  |
| `str`           | `string` |
| `int`, `float`, `int- & float-derived Enums` | `number` |
| `True`          | `true`   |
| `False`         | `false`  |
| `None`          | `null`   |

(Changed in 3.4: Added support for int- and float-derived Enum classes).

To extend this encoder, subclass `JSONEncoder` and implement a `default(self, o)` method. This method should return a serializable object for `o` if possible, otherwise it should call the superclass implementation (to raise `TypeError`).

**Constructor Parameters:**
*   **`skipkeys`**: If `False` (default), a `TypeError` is raised for non-basic keys (`str`, `int`, `float`, `bool`, `None`). If `True`, such items are skipped.
*   **`ensure_ascii`**: If `True` (default), non-ASCII characters are escaped. If `False`, they are output as-is.
*   **`check_circular`**: If `True` (default), lists, dicts, and custom encoded objects are checked for circular references to prevent `RecursionError`.
*   **`allow_nan`**: If `True` (default), `NaN`, `Infinity`, and `-Infinity` are encoded as such (not strictly JSON compliant but common in JavaScript). If `False`, encoding such floats raises a `ValueError`.
*   **`sort_keys`**: If `True` (default: `False`), dictionary output is sorted by key. Useful for regression tests.
*   **`indent`**: If a non-negative integer or string, JSON array elements and object members are pretty-printed. `0`, negative, or `""` inserts only newlines. `None` (default) selects the most compact representation. (Changed in 3.2: allows strings for `indent`).
*   **`separators`**: An `(item_separator, key_separator)` tuple. Default is `(', ', ': ')` if `indent` is `None` and `(',', ': ')` otherwise. For compact JSON, use `(',', ':')`. (Changed in 3.4: default for `indent` not `None`).
*   **`default`**: A function called for objects that cannot otherwise be serialized. It should return a JSON encodable version of the object or raise a `TypeError`. If not specified, `TypeError` is raised.

#### `default(o)`

Implement this method in a subclass to return a serializable object for `o`, or call the base implementation to raise a `TypeError`.

**Example:** To support arbitrary iterators:
```python
def default(self, o):
    try:
        iterable = iter(o)
    except TypeError:
        pass
    else:
        return list(iterable)
    # Let the base class default method raise the TypeError
    return super().default(o)
```

#### `encode(o)`

Returns a JSON string representation of a Python data structure, `o`.

**Example:**
```python
import json
json.JSONEncoder().encode({"foo": ["bar", "baz"]})
# Output: '{"foo": ["bar", "baz"]}'
```

#### `iterencode(o)`

Encodes the given object, `o`, and yields each string representation as available. Useful for streaming large objects.

**Example:**
```python
import json
# Assuming 'bigobject' is a large Python object
# for chunk in json.JSONEncoder().iterencode(bigobject):
#     mysocket.write(chunk)
```

## Exceptions

### `exception json.JSONDecodeError(msg, doc, pos)`

A subclass of `ValueError` raised when the data being deserialized is not a valid JSON document. It includes the following additional attributes:

*   **`msg`**: The unformatted error message.
*   **`doc`**: The JSON document being parsed.
*   **`pos`**: The start index in `doc` where parsing failed.
*   **`lineno`**: The line number corresponding to `pos`. (Added in 3.5).
*   **`colno`**: The column number corresponding to `pos`. (Added in 3.5).

## Standard Compliance and Interoperability

The JSON format is specified by [RFC 7159](https://datatracker.ietf.org/doc/html/rfc7159.html) and [ECMA-404](https://ecma-international.org/publications-and-standards/standards/ecma-404/). This module implements some extensions that are valid JavaScript but not strictly valid JSON:

*   Infinite and NaN number values are accepted and output.
*   Repeated names within an object are accepted, with only the value of the last name-value pair being used.

Since the RFC permits compliant parsers to accept non-compliant input, this module's deserializer is technically RFC-compliant under default settings.

### Character Encodings

The RFC requires JSON to be represented using UTF-8, UTF-16, or UTF-32, with UTF-8 recommended.
By default, this module's serializer sets `ensure_ascii=True`, escaping non-ASCII characters to produce ASCII-only output.
The module primarily handles conversion between Python objects and Unicode strings, not directly character encodings.
The serializer does not add a Byte Order Mark (BOM). The deserializer raises a `ValueError` if an initial BOM is present, as prohibited by the RFC.
This module accepts and outputs (when present in the original `str`) code points for byte sequences that don't correspond to valid Unicode characters (e.g., unpaired UTF-16 surrogates), which may cause interoperability problems.

### Infinite and NaN Number Values

The RFC does not permit infinite or NaN number values. However, by default, this module accepts and outputs `Infinity`, `-Infinity`, and `NaN` as if they were valid JSON number literals:

```python
import json
# Neither of these calls raises an exception, but the results are not valid JSON
print(json.dumps(float('-inf'))) # Output: '-Infinity'
print(json.dumps(float('nan')))  # Output: 'NaN'

# Same when deserializing
print(json.loads('-Infinity')) # Output: -inf
print(json.loads('NaN'))      # Output: nan
```

This behavior can be altered using the `allow_nan` parameter in the serializer and the `parse_constant` parameter in the deserializer.

### Repeated Names Within an Object

The RFC specifies that names within a JSON object should be unique but doesn't mandate handling of repeated names. By default, this module ignores all but the last name-value pair for a given name:

```python
import json
weird_json = '{"x": 1, "x": 2, "x": 3}'
print(json.loads(weird_json)) # Output: {'x': 3}
```

The `object_pairs_hook` parameter can be used to alter this behavior.

### Top-level Non-Object, Non-Array Values

The obsolete RFC 4627 required the top-level JSON value to be an object or array. RFC 7159 removed this restriction. This module does not and has never implemented this restriction in its serializer or deserializer. For maximum interoperability, you may still choose to adhere to the older restriction.

### Implementation Limitations

Some JSON deserializer implementations may impose limits on:
*   The size of accepted JSON texts.
*   The maximum level of nesting of JSON objects and arrays.
*   The range and precision of JSON numbers.
*   The content and maximum length of JSON strings.

This module does not impose such limits beyond those of Python datatypes or the interpreter itself. When serializing, be aware of these potential limitations in consuming applications, especially regarding large integer values or "exotic" numerical types like `decimal.Decimal`, which might be deserialized into IEEE 754 double-precision numbers.

## Command Line Interface (`json.tool`)

The `json.tool` module provides a simple command-line interface to validate and pretty-print JSON objects.

If `infile` and `outfile` arguments are not specified, `sys.stdin` and `sys.stdout` are used respectively.

```bash
$ echo '{"json": "obj"}' | python -m json.tool
{
    "json": "obj"
}
$ echo '{1.2:3.4}' | python -m json.tool
Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

(Changed in 3.5: Output order matches input order by default. Use `--sort-keys` for alphabetical sorting).

### Command Line Options

*   **`infile`**: The JSON file to be validated or pretty-printed. If not specified, reads from `sys.stdin`.
    ```bash
    $ python -m json.tool my_data.json
    # Example content of my_data.json:
    # [
    #     {
    #         "title": "And Now for Something Completely Different",
    #         "year": 1971
    #     },
    #     {
    #         "title": "Monty Python and the Holy Grail",
    #         "year": 1975
    #     }
    # ]
    ```
*   **`outfile`**: Writes the output to the given `outfile`. Otherwise, writes to `sys.stdout`.
*   **`--sort-keys`**: Sorts the output of dictionaries alphabetically by key. (Added in 3.5).
*   **`--no-ensure-ascii`**: Disables escaping of non-ASCII characters. See `json.dumps()` for more information. (Added in 3.9).
*   **`--json-lines`**: Parses every input line as a separate JSON object. (Added in 3.8).
*   **`--indent`, `--tab`, `--no-indent`, `--compact`**: Mutually exclusive options for whitespace control. (Added in 3.9).
*   **`-h`, `--help`**: Shows the help message.

---

# Python `pathlib` Module: Object-Oriented Filesystem Paths

The `pathlib` module provides an object-oriented approach to filesystem paths, offering classes that represent paths with semantics appropriate for different operating systems. Introduced in Python 3.4, it aims to simplify common path manipulation and I/O operations compared to the traditional `os` and `os.path` modules.

The module divides path classes into two main categories:
*   **Pure paths**: Provide purely computational operations without accessing the filesystem. Useful for manipulating paths on a different OS (e.g., Windows paths on Unix) or for ensuring no accidental I/O.
*   **Concrete paths**: Inherit from pure paths and extend functionality with I/O operations that interact with the filesystem.

For most common use cases, the `Path` class is recommended, as it instantiates a concrete path suitable for the platform the code is running on.

## Basic Use

Here are some fundamental examples demonstrating `pathlib`'s capabilities:

```python
from pathlib import Path

# Get the current directory
p = Path('.')

# Listing subdirectories
print([x for x in p.iterdir() if x.is_dir()])

# Listing Python source files recursively
print(list(p.glob('**/*.py')))

# Navigating inside a directory tree using the slash operator
p = Path('/etc')
q = p / 'init.d' / 'reboot'
print(q)
print(q.resolve()) # Resolve symlinks and '..' components

# Querying path properties
print(q.exists())
print(q.is_dir())

# Opening a file
with q.open() as f:
    print(f.readline())
```

## Exceptions

### `pathlib.UnsupportedOperation`

This exception inherits from `NotImplementedError` and is raised when an unsupported operation is called on a path object. This typically occurs when attempting to perform a concrete path operation (which involves system calls) on a pure path object, or when trying to instantiate a platform-specific concrete path (e.g., `WindowsPath`) on an incompatible operating system.

## Pure Paths

Pure path objects handle path manipulation without interacting with the filesystem. They are useful for abstract path operations or cross-platform path handling.

### Classes

*   **`pathlib.PurePath(*pathsegments)`**:
    A generic class representing the system's path flavor. Instantiating it creates either a `PurePosixPath` or a `PureWindowsPath` based on the current OS.
    ```python
    >>> from pathlib import PurePath
    >>> PurePath('setup.py') # Running on a Unix machine
    PurePosixPath('setup.py')
    ```
    Path segments can be strings or other path-like objects. If a segment is an absolute path, all previous segments are ignored. Spurious slashes and single dots (`.`) are collapsed, but double dots (`..`) and leading double slashes (`//`) are preserved to maintain path meaning (e.g., for symbolic links or UNC paths).
    Pure path objects implement the `os.PathLike` interface.

*   **`pathlib.PurePosixPath(*pathsegments)`**:
    A subclass of `PurePath` representing non-Windows filesystem paths.
    ```python
    >>> from pathlib import PurePosixPath
    >>> PurePosixPath('/etc/hosts')
    PurePosixPath('/etc/hosts')
    ```

*   **`pathlib.PureWindowsPath(*pathsegments)`**:
    A subclass of `PurePath` representing Windows filesystem paths, including UNC paths.
    ```python
    >>> from pathlib import PureWindowsPath
    >>> PureWindowsPath('c:/', 'Users', 'Ximénez')
    PureWindowsPath('c:/Users/Ximénez')
    >>> PureWindowsPath('//server/share/file')
    PureWindowsPath('//server/share/file')
    ```

### General Properties

Pure paths are immutable and hashable. Paths of the same flavor are comparable and orderable, respecting the flavor's case-folding semantics (e.g., Windows paths are case-insensitive for comparison). Paths of different flavors compare unequal and cannot be ordered.

```python
>>> PurePosixPath('foo') == PurePosixPath('FOO')
False
>>> PureWindowsPath('foo') == PureWindowsPath('FOO')
True
>>> PureWindowsPath('FOO') in { PureWindowsPath('foo') }
True
```

### Operators

*   **Slash Operator (`/`)**: Used to create child paths, similar to `os.path.join()`. If the argument is an absolute path, the previous path is ignored.
    ```python
    >>> p = PurePath('/etc')
    >>> p / 'init.d' / 'apache2'
    PurePosixPath('/etc/init.d/apache2')
    >>> '/usr' / PurePath('bin')
    PurePosixPath('/usr/bin')
    >>> p / '/an_absolute_path'
    PurePosixPath('/an_absolute_path')
    ```
*   **`os.PathLike` Interface**: Path objects can be used anywhere an object implementing `os.PathLike` is accepted.
    ```python
    >>> import os
    >>> p = PurePath('/etc')
    >>> os.fspath(p)
    '/etc'
    ```
*   **String Representation (`str(path)`)**: Returns the raw filesystem path in native form (e.g., with backslashes on Windows).
    ```python
    >>> p = PurePath('/etc')
    >>> str(p)
    '/etc'
    >>> p = PureWindowsPath('c:/Program Files')
    >>> str(p)
    'c:\\Program Files'
    ```
*   **Bytes Representation (`bytes(path)`)**: Returns the raw filesystem path as a bytes object, encoded by `os.fsencode()`. This is primarily recommended for Unix systems, as the Unicode form is canonical on Windows.
    ```python
    >>> bytes(PurePosixPath('/etc'))
    b'/etc'
    ```

### Accessing Individual Parts

*   **`PurePath.parts`**: A tuple providing access to the path's individual components.
    ```python
    >>> p = PurePath('/usr/bin/python3')
    >>> p.parts
    ('/', 'usr', 'bin', 'python3')
    >>> p = PureWindowsPath('c:/Program Files/PSF')
    >>> p.parts
    ('c:\\', 'Program Files', 'PSF')
    ```

### Methods and Properties

*   **`PurePath.parser`**: The `os.path` module implementation used for low-level path parsing and joining (either `posixpath` or `ntpath`).
*   **`PurePath.drive`**: A string representing the drive letter or name, if any. UNC shares are considered drives on Windows.
    ```python
    >>> PureWindowsPath('c:/Program Files/').drive
    'c:'
    >>> PureWindowsPath('//host/share/foo.txt').drive
    '\\\\host\\share'
    ```
*   **`PurePath.root`**: A string representing the (local or global) root, if any.
    ```python
    >>> PureWindowsPath('c:/Program Files/').root
    '\\'
    >>> PurePosixPath('/etc').root
    '/'
    ```
    `PurePosixPath` collapses more than two leading slashes to a single slash (e.g., `///etc` becomes `/etc`).
*   **`PurePath.anchor`**: The concatenation of the drive and root.
    ```python
    >>> PureWindowsPath('c:/Program Files/').anchor
    'c:\\'
    >>> PurePosixPath('/etc').anchor
    '/'
    ```
*   **`PurePath.parents`**: An immutable sequence providing access to the logical ancestors of the path. Supports slicing and negative indexing since Python 3.10.
    ```python
    >>> p = PureWindowsPath('c:/foo/bar/setup.py')
    >>> p.parents[0]
    PureWindowsPath('c:/foo/bar')
    >>> p.parents[2]
    PureWindowsPath('c:/')
    ```
*   **`PurePath.parent`**: The logical parent of the path. This is a purely lexical operation; it does not resolve `..` components or symlinks.
    ```python
    >>> p = PurePosixPath('/a/b/c/d')
    >>> p.parent
    PurePosixPath('/a/b/c')
    >>> PurePosixPath('/').parent
    PurePosixPath('/')
    ```
*   **`PurePath.name`**: A string representing the final path component, excluding the drive and root.
    ```python
    >>> PurePosixPath('my/library/setup.py').name
    'setup.py'
    ```
*   **`PurePath.suffix`**: The last dot-separated portion of the final component (file extension), if any.
    ```python
    >>> PurePosixPath('my/library/setup.py').suffix
    '.py'
    >>> PurePosixPath('my/library').suffix
    ''
    ```
*   **`PurePath.suffixes`**: A list of the path's suffixes (file extensions).
    ```python
    >>> PurePosixPath('my/library.tar.gz').suffixes
    ['.tar', '.gz']
    ```
*   **`PurePath.stem`**: The final path component, without its suffix.
    ```python
    >>> PurePosixPath('my/library.tar.gz').stem
    'library.tar'
    ```
*   **`PurePath.as_posix()`**: Returns a string representation of the path with forward slashes (`/`), even on Windows.
    ```python
    >>> p = PureWindowsPath('c:\\windows')
    >>> p.as_posix()
    'c:/windows'
    ```
*   **`PurePath.is_absolute()`**: Returns `True` if the path is absolute (has both a root and, if applicable, a drive).
    ```python
    >>> PurePosixPath('/a/b').is_absolute()
    True
    >>> PureWindowsPath('/a/b').is_absolute()
    False # Windows paths need a drive letter to be absolute
    >>> PureWindowsPath('c:/a/b').is_absolute()
    True
    ```
*   **`PurePath.is_relative_to(other)`**: Returns `True` if this path is relative to `other`. This is a string-based operation and does not access the filesystem or treat `..` specially.
    ```python
    >>> p = PurePath('/etc/passwd')
    >>> p.is_relative_to('/etc')
    True
    ```
    *Deprecated since 3.12, will be removed in 3.14: Passing additional arguments is deprecated.*
*   **`PurePath.is_reserved()`**: (Windows only) Returns `True` if the path is considered reserved under Windows. Always `False` for `PurePosixPath`.
    *Deprecated since 3.13, will be removed in 3.15: Use `os.path.isreserved()` instead.*
*   **`PurePath.joinpath(*pathsegments)`**: Equivalent to combining the path with each of the given `pathsegments` in turn using the slash operator.
    ```python
    >>> PurePosixPath('/etc').joinpath('init.d', 'apache2')
    PurePosixPath('/etc/init.d/apache2')
    ```
*   **`PurePath.full_match(pattern, *, case_sensitive=None)`**: Matches the path against a glob-style pattern. Returns `True` if the entire path matches. Case-sensitivity follows platform defaults unless overridden.
    ```python
    >>> PurePath('a/b.py').full_match('a/*.py')
    True
    >>> PurePath('/a/b/c.py').full_match('**/*.py')
    True
    ```
    *Added in version 3.13.*
*   **`PurePath.match(pattern, *, case_sensitive=None)`**: Matches the path against a non-recursive glob-style pattern. Similar to `full_match()`, but `**` acts like `*`, and relative patterns match from the right. Empty patterns are not allowed.
    ```python
    >>> PurePath('a/b.py').match('*.py')
    True
    >>> PurePath('/a/b/c.py').match('b/*.py')
    True
    ```
*   **`PurePath.relative_to(other, walk_up=False)`**: Computes a version of this path relative to `other`. Raises `ValueError` if impossible.
    If `walk_up` is `True`, `..` entries may be added to form the relative path. This is a lexical operation and does not check the filesystem or resolve symlinks.
    ```python
    >>> p = PurePosixPath('/etc/passwd')
    >>> p.relative_to('/')
    PurePosixPath('etc/passwd')
    >>> p.relative_to('/usr', walk_up=True)
    PurePosixPath('../etc/passwd')
    ```
    *Changed in version 3.12: The `walk_up` parameter was added.*
    *Deprecated since 3.12, will be removed in 3.14: Passing additional positional arguments is deprecated.*
*   **`PurePath.with_name(name)`**: Returns a new path with the `name` changed. Raises `ValueError` if the original path has no name (e.g., `PureWindowsPath('c:/')`).
    ```python
    >>> p = PureWindowsPath('c:/Downloads/pathlib.tar.gz')
    >>> p.with_name('setup.py')
    PureWindowsPath('c:/Downloads/setup.py')
    ```
*   **`PurePath.with_stem(stem)`**: Returns a new path with the `stem` changed. Raises `ValueError` if the original path has no name.
    ```python
    >>> p = PureWindowsPath('c:/Downloads/draft.txt')
    >>> p.with_stem('final')
    PureWindowsPath('c:/Downloads/final.txt')
    ```
    *Added in version 3.9.*
*   **`PurePath.with_suffix(suffix)`**: Returns a new path with the `suffix` changed. If the original path has no suffix, the new suffix is appended. An empty string removes the suffix.
    ```python
    >>> p = PureWindowsPath('c:/Downloads/pathlib.tar.gz')
    >>> p.with_suffix('.bz2')
    PureWindowsPath('c:/Downloads/pathlib.tar.bz2')
    ```
*   **`PurePath.with_segments(*pathsegments)`**: Creates a new path object of the same type by combining the given `pathsegments`. Subclasses can override this to pass information to derivative paths.
    ```python
    from pathlib import PurePosixPath

    class MyPath(PurePosixPath):
        def __init__(self, *pathsegments, session_id):
            super().__init__(*pathsegments)
            self.session_id = session_id

        def with_segments(self, *pathsegments):
            return type(self)(*pathsegments, session_id=self.session_id)

    etc = MyPath('/etc', session_id=42)
    hosts = etc / 'hosts'
    print(hosts.session_id) # 42
    ```
    *Added in version 3.12.*

## Concrete Paths

Concrete path objects are subclasses of pure path classes and provide methods for performing system calls that interact with the filesystem.

### Classes

*   **`pathlib.Path(*pathsegments)`**:
    A subclass of `PurePath` representing concrete paths of the system’s path flavor. Instantiating it creates either a `PosixPath` or a `WindowsPath`.
    ```python
    >>> from pathlib import Path
    >>> Path('setup.py')
    PosixPath('setup.py')
    ```

*   **`pathlib.PosixPath(*pathsegments)`**:
    A subclass of `Path` and `PurePosixPath`, representing concrete non-Windows filesystem paths.
    ```python
    >>> from pathlib import PosixPath
    >>> PosixPath('/etc/hosts')
    PosixPath('/etc/hosts')
    ```
    Raises `UnsupportedOperation` on Windows.

*   **`pathlib.WindowsPath(*pathsegments)`**:
    A subclass of `Path` and `PureWindowsPath`, representing concrete Windows filesystem paths.
    ```python
    >>> from pathlib import WindowsPath
    >>> WindowsPath('c:/', 'Users', 'Ximénez')
    WindowsPath('c:/Users/Ximénez')
    ```
    Raises `UnsupportedOperation` on non-Windows platforms.

You can only instantiate the concrete path class that corresponds to your system. Concrete path methods can raise an `OSError` if a system call fails (e.g., path doesn't exist).

### Parsing and Generating URIs

Concrete path objects can be created from and represented as 'file' URIs conforming to [RFC 8089](https://datatracker.ietf.org/doc/html/rfc8089.html).

*   **`classmethod Path.from_uri(uri)`**: Returns a new path object by parsing a 'file' URI. Raises `ValueError` if the URI does not start with `file:` or the parsed path isn't absolute.
    ```python
    >>> Path.from_uri('file:///etc/hosts')
    PosixPath('/etc/hosts')
    >>> Path.from_uri('file:///c:/windows')
    WindowsPath('c:/windows')
    ```
    *Added in version 3.13.*
*   **`Path.as_uri()`**: Represents the path as a 'file' URI. Raises `ValueError` if the path isn't absolute.
    ```python
    >>> p = PosixPath('/etc/passwd')
    >>> p.as_uri()
    'file:///etc/passwd'
    ```

### Expanding and Resolving Paths

*   **`classmethod Path.home()`**: Returns a new path object representing the user's home directory. Raises `RuntimeError` if the home directory cannot be resolved.
    ```python
    >>> Path.home()
    PosixPath('/home/antoine')
    ```
    *Added in version 3.5.*
*   **`Path.expanduser()`**: Returns a new path with expanded `~` and `~user` constructs. Raises `RuntimeError` if a home directory cannot be resolved.
    ```python
    >>> p = PosixPath('~/films/Monty Python')
    >>> p.expanduser()
    PosixPath('/home/eric/films/Monty Python')
    ```
    *Added in version 3.5.*
*   **`classmethod Path.cwd()`**: Returns a new path object representing the current working directory.
    ```python
    >>> Path.cwd()
    PosixPath('/home/antoine/pathlib')
    ```
*   **`Path.absolute()`**: Makes the path absolute without normalization or resolving symlinks. Returns a new path object.
    ```python
    >>> p = Path('tests')
    >>> p.absolute()
    PosixPath('/home/antoine/pathlib/tests')
    ```
*   **`Path.resolve(strict=False)`**: Makes the path absolute, resolving any symlinks and eliminating `..` components. Returns a new path object.
    If `strict` is `True` (default pre-3.6), `OSError` is raised if the path doesn't exist or a symlink loop is encountered. If `strict` is `False`, the path is resolved as far as possible.
    ```python
    >>> p = Path('docs/../setup.py')
    >>> p.resolve()
    PosixPath('/home/antoine/pathlib/setup.py')
    ```
    *Changed in version 3.13: Symlink loops are treated like other errors: `OSError` in strict mode, no exception in non-strict mode.*
*   **`Path.readlink()`**: Returns the path to which the symbolic link points.
    ```python
    >>> p = Path('mylink')
    >>> p.symlink_to('setup.py')
    >>> p.readlink()
    PosixPath('setup.py')
    ```
    *Added in version 3.9.* Raises `UnsupportedOperation` if `os.readlink()` is not available.

### Querying File Type and Status

These methods return `False` instead of raising an exception for paths that contain characters unrepresentable at the OS level (since 3.8).

*   **`Path.stat(*, follow_symlinks=True)`**: Returns an `os.stat_result` object containing information about the path. By default, follows symlinks.
    ```python
    >>> p = Path('setup.py')
    >>> p.stat().st_size
    956
    ```
    *Changed in version 3.10: The `follow_symlinks` parameter was added.*
*   **`Path.lstat()`**: Like `Path.stat()` but returns information about the symbolic link itself, not its target.
*   **`Path.exists(*, follow_symlinks=True)`**: Returns `True` if the path points to an existing file or directory. By default, follows symlinks.
    ```python
    >>> Path('.').exists()
    True
    >>> Path('nonexistentfile').exists()
    False
    ```
    *Changed in version 3.12: The `follow_symlinks` parameter was added.*
*   **`Path.is_file(*, follow_symlinks=True)`**: Returns `True` if the path points to a regular file. `False` if it's another type of file, doesn't exist, or is a broken symlink. By default, follows symlinks.
    *Changed in version 3.13: The `follow_symlinks` parameter was added.*
*   **`Path.is_dir(*, follow_symlinks=True)`**: Returns `True` if the path points to a directory. `False` if it's another type of file, doesn't exist, or is a broken symlink. By default, follows symlinks.
    *Changed in version 3.13: The `follow_symlinks` parameter was added.*
*   **`Path.is_symlink()`**: Returns `True` if the path points to a symbolic link. `False` if it doesn't exist.
*   **`Path.is_junction()`**: Returns `True` if the path points to a junction (Windows only).
    *Added in version 3.12.*
*   **`Path.is_mount()`**: Returns `True` if the path is a mount point.
    *Added in version 3.7. Windows support added in 3.12.*
*   **`Path.is_socket()`**: Returns `True` if the path points to a Unix socket.
*   **`Path.is_fifo()`**: Returns `True` if the path points to a FIFO (named pipe).
*   **`Path.is_block_device()`**: Returns `True` if the path points to a block device.
*   **`Path.is_char_device()`**: Returns `True` if the path points to a character device.
*   **`Path.samefile(other_path)`**: Returns `True` if this path points to the same file as `other_path`. Raises `OSError` if files cannot be accessed.
    ```python
    >>> p = Path('spam')
    >>> q = Path('eggs')
    >>> p.samefile(q)
    False
    ```
    *Added in version 3.5.*

### Reading and Writing Files

*   **`Path.open(mode='r', buffering=-1, encoding=None, errors=None, newline=None)`**: Opens the file pointed to by the path, like the built-in `open()` function.
    ```python
    >>> p = Path('setup.py')
    >>> with p.open() as f:
    ...     f.readline()
    ...
    '#!/usr/bin/env python3\n'
    ```
*   **`Path.read_text(encoding=None, errors=None, newline=None)`**: Returns the decoded contents of the file as a string. The file is opened and closed.
    ```python
    >>> p = Path('my_text_file')
    >>> p.write_text('Text file contents')
    18
    >>> p.read_text()
    'Text file contents'
    ```
    *Added in version 3.5.*
*   **`Path.read_bytes()`**: Returns the binary contents of the file as a bytes object.
    ```python
    >>> p = Path('my_binary_file')
    >>> p.write_bytes(b'Binary file contents')
    20
    >>> p.read_bytes()
    b'Binary file contents'
    ```
    *Added in version 3.5.*
*   **`Path.write_text(data, encoding=None, errors=None, newline=None)`**: Opens the file in text mode, writes `data` to it, and closes the file. Overwrites existing files.
    ```python
    >>> p = Path('my_text_file')
    >>> p.write_text('Text file contents')
    18
    ```
    *Added in version 3.5.*
*   **`Path.write_bytes(data)`**: Opens the file in bytes mode, writes `data` to it, and closes the file. Overwrites existing files.
    ```python
    >>> p = Path('my_binary_file')
    >>> p.write_bytes(b'Binary file contents')
    20
    ```
    *Added in version 3.5.*

### Reading Directories

*   **`Path.iterdir()`**: When the path points to a directory, yields path objects for its contents. Special entries `.` and `..` are not included. Raises `OSError` if the path is not a directory or inaccessible.
    ```python
    >>> p = Path('docs')
    >>> for child in p.iterdir(): print(child)
    PosixPath('docs/conf.py')
    PosixPath('docs/_templates')
    # ...
    ```
*   **`Path.glob(pattern, *, case_sensitive=None, recurse_symlinks=False)`**: Globs the given relative `pattern` in the directory represented by this path, yielding all matching files.
    ```python
    >>> sorted(Path('.').glob('**/*.py'))
    [PosixPath('build/lib/pathlib.py'), PosixPath('docs/conf.py'), ...]
    ```
    Case-sensitivity follows platform defaults unless overridden. By default, `**` does not follow symlinks; set `recurse_symlinks=True` to always follow them.
    *Changed in version 3.13: `recurse_symlinks` parameter added. Any `OSError` exceptions from scanning the filesystem are suppressed.*
*   **`Path.rglob(pattern, *, case_sensitive=None, recurse_symlinks=False)`**: Globs the given relative `pattern` recursively. Equivalent to `Path.glob()` with `**/` added in front of the pattern.
    *Changed in version 3.13: `recurse_symlinks` parameter added.*
*   **`Path.walk(top_down=True, on_error=None, follow_symlinks=False)`**: Generates filenames in a directory tree by walking it top-down or bottom-up. For each directory, yields a 3-tuple: `(dirpath, dirnames, filenames)`.
    *   `dirpath`: A `Path` object for the current directory.
    *   `dirnames`: List of strings for subdirectory names.
    *   `filenames`: List of strings for non-directory file names.
    If `top_