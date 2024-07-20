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
        self._row_cache = {}
        self._col_cache = {}

    def __getitem__(self, key: Union[int, slice, str]) -> Union[np.ndarray, 'FastRow', 'FastDataFrameView']:
        if isinstance(key, int):
            return self._get_row(key)
        elif isinstance(key, slice):
            return self._get_slice(key)
        elif isinstance(key, str):
            return self._get_column(key)
        else:
            return self.obj.data[self.obj.start:self.obj.stop][key]

    def _get_row(self, key: int) -> 'FastRow':
        absolute_key = self.obj.start + key if key >= 0 else self.obj.stop + key
        if self.obj.start <= absolute_key < self.obj.stop:
            if absolute_key not in self._row_cache:
                self._row_cache[absolute_key] = FastRow(self.obj.data[absolute_key], self.obj.column_indices)
            return self._row_cache[absolute_key]
        raise IndexError("FastDataFrame index out of range")

    def _get_slice(self, key: slice) -> 'FastDataFrameView':
        start = self.obj.start if key.start is None else (self.obj.start + key.start if key.start >= 0 else self.obj.stop + key.start)
        stop = self.obj.stop if key.stop is None else (self.obj.start + key.stop + 1 if key.stop >= 0 else self.obj.stop + key.stop + 1)
        start = max(self.obj.start, min(start, self.obj.stop))
        stop = max(start, min(stop, self.obj.stop))
        return FastDataFrameView(self.obj.data, self.obj.column_names, self.obj.column_indices, start, stop)

    def _get_column(self, key: str) -> np.ndarray:
        if key not in self._col_cache:
            self._col_cache[key] = self.obj.column_indices[key]
        col_index = self._col_cache[key]
        return self.obj.data[self.obj.start:self.obj.stop, col_index]

class FastDataFrameView:
    def __init__(self, data: np.ndarray, column_names: List[str], column_indices: Dict[str, int], start: int, stop: int):
        self.data = data
        self.column_names = column_names
        self.column_indices = column_indices
        self.start = start
        self.stop = stop
        self._loc = None

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[self.start:self.stop, self.column_indices[key]]
        elif isinstance(key, int):
            return self.data[self.start + key]
        return FastDataFrameView(self.data[self.start:self.stop][key], self.column_names, self.column_indices, 0, len(self.data[self.start:self.stop][key]))

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

    def shift(self, periods: int = 1, fill_value=None) -> 'FastDataFrameView':
        shifted_data = np.empty_like(self.data[self.start:self.stop])
        if periods > 0:
            shifted_data[:periods] = fill_value if fill_value is not None else np.nan
            shifted_data[periods:] = self.data[self.start:self.stop-periods]
        elif periods < 0:
            shifted_data[periods:] = fill_value if fill_value is not None else np.nan
            shifted_data[:periods] = self.data[self.start-periods:self.stop]
        else:
            shifted_data[:] = self.data[self.start:self.stop]
        return FastDataFrameView(shifted_data, self.column_names, self.column_indices, 0, len(shifted_data))

    def mean(self, axis=None):
        if axis is None:
            return np.nanmean(self.data[self.start:self.stop], axis=0)
        elif axis == 0 or axis == 1:
            return np.nanmean(self.data[self.start:self.stop], axis=axis)
        else:
            raise ValueError("Axis must be 0, 1 or None")

    def sum(self, axis=None):
        if axis is None:
            return np.nansum(self.data[self.start:self.stop], axis=0)
        elif axis == 0 or axis == 1:
            return np.nansum(self.data[self.start:self.stop], axis=axis)
        else:
            raise ValueError("Axis must be 0, 1 or None")

    def min(self, axis=None):
        if axis is None:
            return np.nanmin(self.data[self.start:self.stop], axis=0)
        elif axis == 0 or axis == 1:
            return np.nanmin(self.data[self.start:self.stop], axis=axis)
        else:
            raise ValueError("Axis must be 0, 1 or None")

    def max(self, axis=None):
        if axis is None:
            return np.nanmax(self.data[self.start:self.stop], axis=0)
        elif axis == 0 or axis == 1:
            return np.nanmax(self.data[self.start:self.stop], axis=axis)
        else:
            raise ValueError("Axis must be 0, 1 or None")

    def isna(self):
        return np.isnan(self.data[self.start:self.stop])

    def fillna(self, value):
        filled_data = np.where(np.isnan(self.data[self.start:self.stop]), value, self.data[self.start:self.stop])
        return FastDataFrameView(filled_data, self.column_names, self.column_indices, 0, len(filled_data))

    def dropna(self, axis=0):
        if axis == 0:
            mask = ~np.isnan(self.data[self.start:self.stop]).any(axis=1)
            return FastDataFrameView(self.data[self.start:self.stop][mask], self.column_names, self.column_indices, 0, np.sum(mask))
        elif axis == 1:
            mask = ~np.isnan(self.data[self.start:self.stop]).any(axis=0)
            new_column_names = [name for name, keep in zip(self.column_names, mask) if keep]
            new_column_indices = {name: i for i, name in enumerate(new_column_names)}
            return FastDataFrameView(self.data[self.start:self.stop][:, mask], new_column_names, new_column_indices, 0, self.stop - self.start)

    def any(self, axis: int = 0) -> np.ndarray:
        return np.any(self.data[self.start:self.stop], axis=axis)

    @property
    def loc(self) -> LocIndexer:
        if self._loc is None:
            self._loc = LocIndexer(self)
        return self._loc

class FastDataFrame(FastDataFrameView):
    def __init__(self, data: np.ndarray, column_names: List[str]):
        super().__init__(np.ascontiguousarray(data), list(column_names),
                         {name: index for index, name in enumerate(column_names)}, 0, len(data))

    @staticmethod
    def from_pandas(df: 'pd.DataFrame') -> 'FastDataFrame':
        return FastDataFrame(df.values, df.columns)
    
    def __setitem__(self, key: str, value: Union[np.ndarray, List, Any]) -> None:
        if isinstance(key, str):
            if key in self.column_indices:
                col_index = self.column_indices[key]
                if isinstance(value, (np.ndarray, list)) and len(value) == len(self):
                    self.data[:, col_index] = value
                else:
                    self.data[:, col_index] = np.full(len(self), value)
            else:
                new_col_index = self.data.shape[1]
                if isinstance(value, (np.ndarray, list)) and len(value) == len(self):
                    new_column = np.array(value).reshape(-1, 1)
                else:
                    new_column = np.full((len(self), 1), value)
                self.data = np.column_stack((self.data, new_column))
                self.column_indices[key] = new_col_index
                self.column_names.append(key)
        else:
            raise ValueError("Only string column names are supported for assignment")

