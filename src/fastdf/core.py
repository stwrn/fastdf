import pandas as pd
import numpy as np
import time
from typing import List, Dict, Union, Any

class FastRow:
    def __init__(self, data: np.ndarray, column_indices: Dict[str, int]):
        self.data = data
        self.column_indices = column_indices

    def __getitem__(self, key: str) -> Any:
        return self.data[self.column_indices[key]]

class LocIndexer:
    def __init__(self, obj: 'FastDataFrameView'):
        self.obj = obj

    def __getitem__(self, key: Union[int, slice, str]) -> Union['FastRow', 'FastDataFrameView', np.ndarray]:
        if isinstance(key, int):
            absolute_key = self.obj.start + key if key >= 0 else self.obj.stop + key
            if self.obj.start <= absolute_key < self.obj.stop:
                return FastRow(self.obj.data[absolute_key - self.obj.start], self.obj.column_indices)
            raise IndexError("FastDataFrame index out of range")
        elif isinstance(key, slice):
            start = self.obj.start if key.start is None else (self.obj.start + key.start if key.start >= 0 else self.obj.stop + key.start)
            stop = self.obj.stop if key.stop is None else (self.obj.start + key.stop if key.stop >= 0 else self.obj.stop + key.stop)
            start = max(self.obj.start, min(start, self.obj.stop))
            stop = max(start, min(stop, self.obj.stop))
            return FastDataFrameView(self.obj.data, self.obj.column_names, self.obj.column_indices, start, stop)
        elif isinstance(key, str):
            return self.obj.data[self.obj.start:self.obj.stop, self.obj.column_indices[key]]
        return self.obj.data[self.obj.start:self.obj.stop][key]

class FastDataFrameView:
    def __init__(self, data: np.ndarray, column_names: List[str], column_indices: Dict[str, int], start: int, stop: int):
        self.data = data
        self.column_names = column_names
        self.column_indices = column_indices
        self.start = start
        self.stop = stop

    def __getitem__(self, key: Union[str, int]) -> Union[np.ndarray, 'FastRow']:
        if isinstance(key, str):
            return self.data[self.start:self.stop, self.column_indices[key]]
        elif isinstance(key, int):
            if 0 <= key < len(self):
                return FastRow(self.data[self.start + key], self.column_indices)
            raise IndexError("FastDataFrame index out of range")
        return self.data[self.start:self.stop][key]

    def __len__(self) -> int:
        return self.stop - self.start

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        rows = min(5, len(self))
        col_width = max(max(len(str(x)) for x in col[self.start:self.stop]) for col in self.data.T)
        result = []
        for i, name in enumerate(self.column_names):
            col_data = self.data[self.start:self.stop, i]
            col_str = f"{name:>{col_width}}"
            for j in range(rows):
                col_str += f"\n{col_data[j]:>{col_width}.6f}"
            if len(col_data) > rows:
                col_str += f"\n{'...':>{col_width}}"
            result.append(col_str)
        return "  ".join(result)

    def shift(self, periods: int = 1) -> 'FastDataFrameView':
        shifted_data = np.roll(self.data[self.start:self.stop], periods, axis=0)
        if periods > 0:
            shifted_data[:periods] = 0
        elif periods < 0:
            shifted_data[periods:] = 0
        return FastDataFrameView(shifted_data, self.column_names, self.column_indices, 0, len(shifted_data))

    def any(self, axis: int = 0) -> np.ndarray:
        return np.any(self.data[self.start:self.stop], axis=axis)

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

class FastDataFrame(FastDataFrameView):
    def __init__(self, data: np.ndarray, column_names: List[str]):
        super().__init__(np.ascontiguousarray(data), list(column_names),
                         {name: index for index, name in enumerate(column_names)}, 0, len(data))

    @staticmethod
    def from_pandas(df: 'pd.DataFrame') -> 'FastDataFrame':
        return FastDataFrame(df.values, df.columns)
