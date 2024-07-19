import pandas as pd
import numpy as np
import time
from fastdf import fdf 

def create_test_data(size=1000000):
    return pd.DataFrame({
        'A': np.random.rand(size),
        'B': np.random.rand(size),
        'C': np.random.rand(size)
    })

def time_operation(func, *args):
    start = time.time()
    func(*args)
    return time.time() - start

def test_element_access(pandas_df, fast_df, iterations=1000000):
    def pandas_access(df):
        for i in range(iterations):
            _ = df.loc[i % len(df)]['B']

    def fast_access(df):
        for i in range(iterations):
            _ = df.loc[i % len(df)]['B']

    pandas_time = time_operation(pandas_access, pandas_df)
    fast_time = time_operation(fast_access, fast_df)
    
    return pandas_time, fast_time

def test_slice_creation(pandas_df, fast_df, iterations=10000):
    def pandas_slice(df):
        for i in range(iterations):
            slice_start = i % (len(df) - 10)
            _ = df.loc[slice_start:slice_start+10]

    def fast_slice(df):
        for i in range(iterations):
            slice_start = i % (len(df) - 10)
            _ = df.loc[slice_start:slice_start+10]

    pandas_time = time_operation(pandas_slice, pandas_df)
    fast_time = time_operation(fast_slice, fast_df)
    
    return pandas_time, fast_time

def test_nested_slice(pandas_df, fast_df, iterations=10000):
    def pandas_nested_slice(df):
        for i in range(iterations):
            slice_start = i % (len(df) - 20)
            slice1 = df.loc[slice_start:slice_start+20]
            _ = slice1.loc[5:15]

    def fast_nested_slice(df):
        for i in range(iterations):
            slice_start = i % (len(df) - 20)
            slice1 = df.loc[slice_start:slice_start+20]
            _ = slice1.loc[5:15]

    pandas_time = time_operation(pandas_nested_slice, pandas_df)
    fast_time = time_operation(fast_nested_slice, fast_df)
    
    return pandas_time, fast_time

def run_performance_tests():
    pandas_df = create_test_data()
    fast_df =  fdf.from_pandas(pandas_df)

    print("Performance Tests:")
    print("-----------------")

    # Element access test
    pandas_time, fast_time = test_element_access(pandas_df, fast_df)
    print(f"\nElement Access:")
    print(f"Pandas time: {pandas_time:.6f} seconds")
    print(f"FastDF time: {fast_time:.6f} seconds")
    print(f"Speedup: {pandas_time / fast_time:.2f}x")

    # Slice creation test
    pandas_time, fast_time = test_slice_creation(pandas_df, fast_df)
    print(f"\nSlice Creation:")
    print(f"Pandas time: {pandas_time:.6f} seconds")
    print(f"FastDF time: {fast_time:.6f} seconds")
    print(f"Speedup: {pandas_time / fast_time:.2f}x")

    # Nested slice test
    pandas_time, fast_time = test_nested_slice(pandas_df, fast_df)
    print(f"\nNested Slice:")
    print(f"Pandas time: {pandas_time:.6f} seconds")
    print(f"FastDF time: {fast_time:.6f} seconds")
    print(f"Speedup: {pandas_time / fast_time:.2f}x")

if __name__ == "__main__":
    run_performance_tests()
