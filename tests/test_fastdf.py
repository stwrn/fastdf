import pandas as pd
import numpy as np
import time
from fastdf import fdf 
import unittest

print("UNITTEST")

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper

def time_access(df, iterations=1000000):
    df_len = len(df)
    start = time.perf_counter()
    for i in range(iterations):
        _ = df.loc[i % df_len]['col_1']
    return time.perf_counter() - start

class TestFastDataFrame(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100000, 100)
        self.columns = [f'col_{i}' for i in range(100)]
        self.pdf = pd.DataFrame(self.data, columns=self.columns)
        self.fdf = fdf(self.data, self.columns)

    def test_indexing(self):
        np.testing.assert_array_equal(self.fdf['col_0'], self.pdf['col_0'])

    def test_slicing(self):
        np.testing.assert_array_equal(self.fdf.loc[:2]['col_0'], self.pdf.loc[:2]['col_0'])

    def test_shift(self):
        np.testing.assert_array_equal(self.fdf.shift(1)['col_0'], self.pdf.shift(1)['col_0'])
        np.testing.assert_array_equal(self.fdf.shift(1, fill_value=0)['col_0'], self.pdf.shift(1, fill_value=0)['col_0'])

    def test_any(self):
        np.testing.assert_array_equal(self.fdf.any(axis=0), self.pdf.any(axis=0))

    def test_mean(self):
        np.testing.assert_array_almost_equal(self.fdf.mean(), self.pdf.mean())
        np.testing.assert_array_almost_equal(self.fdf.mean(axis=0), self.pdf.mean(axis=0))
        np.testing.assert_array_almost_equal(self.fdf.mean(axis=1), self.pdf.mean(axis=1))

    def test_sum(self):
        np.testing.assert_array_almost_equal(self.fdf.sum(), self.pdf.sum())
        np.testing.assert_array_almost_equal(self.fdf.sum(axis=0), self.pdf.sum(axis=0))
        np.testing.assert_array_almost_equal(self.fdf.sum(axis=1), self.pdf.sum(axis=1))

    def test_min(self):
        np.testing.assert_array_almost_equal(self.fdf.min(), self.pdf.min())
        np.testing.assert_array_almost_equal(self.fdf.min(axis=0), self.pdf.min(axis=0))
        np.testing.assert_array_almost_equal(self.fdf.min(axis=1), self.pdf.min(axis=1))

    def test_max(self):
        np.testing.assert_array_almost_equal(self.fdf.max(), self.pdf.max())
        np.testing.assert_array_almost_equal(self.fdf.max(axis=0), self.pdf.max(axis=0))
        np.testing.assert_array_almost_equal(self.fdf.max(axis=1), self.pdf.max(axis=1))

    def test_isna(self):
        np.testing.assert_array_equal(self.fdf.isna(), self.pdf.isna())

    def test_fillna(self):
        np.testing.assert_array_equal(self.fdf.fillna(0).data, self.pdf.fillna(0).values)

    def test_dropna(self):
        np.testing.assert_array_equal(self.fdf.dropna().data, self.pdf.dropna().values)
        np.testing.assert_array_equal(self.fdf.dropna(axis=1).data, self.pdf.dropna(axis=1).values)

    def test_overwrite(self):
        new_values = np.random.rand(100000)
        self.pdf['col_0'] = new_values
        self.fdf['col_0'] = new_values
        np.testing.assert_array_equal(self.fdf['col_0'], self.pdf['col_0'])

    def test_performance(self):
        operations = [
            ('Indexing', lambda df: df['col_0']),
            ('Slicing', lambda df: df.loc[:100]),
            ('Shift', lambda df: df.shift(1)),
            ('Mean', lambda df: df.mean()),
            ('Sum', lambda df: df.sum()),
            ('Min', lambda df: df.min()),
            ('Max', lambda df: df.max()),
            ('IsNA', lambda df: df.isna()),
            ('FillNA', lambda df: df.fillna(0)),
            ('DropNA', lambda df: df.dropna()),
            ('Overwrite', lambda df: df.__setitem__('col_0', np.random.rand(len(df)))),
        ]

        print("\nPerformance comparison:")
        for name, op in operations:
            fdf_op = measure_time(op)
            pdf_op = measure_time(op)
            
            _, fdf_time = fdf_op(self.fdf)
            _, pdf_time = pdf_op(self.pdf)
            
            speedup = pdf_time / fdf_time
            print(f"{name:10}: fdf {fdf_time:.6f}s, pandas {pdf_time:.6f}s, Speedup: {speedup:.2f}x")

        fdf_access_time = time_access(self.fdf)
        pdf_access_time = time_access(self.pdf)
        access_speedup = pdf_access_time / fdf_access_time
        print(f"{'Access':10}: fdf {fdf_access_time:.6f}s, pandas {pdf_access_time:.6f}s, Speedup: {access_speedup:.2f}x")

if __name__ == '__main__':
    unittest.main(verbosity=2)
