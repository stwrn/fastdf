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

#import pytest
#import numpy as np
#import pandas as pd
#import time
#from fastdf import fdf 
#
#def generate_test_data(size=10000000):
#    return pd.DataFrame({
#        'A': np.random.rand(size),
#        'B': np.random.rand(size),
#        'C': np.random.rand(size)
#    })
#
#@pytest.fixture
#def test_data():
#    return generate_test_data()
#
#def test_from_pandas(test_data):
#    fast_df = fdf.from_pandas(test_data) 
#    assert fast_df.data.shape == test_data.shape
#    assert all(fast_df.column_names == test_data.columns)
#
#def test_getitem_performance(test_data):
#    fast_df = fdf.from_pandas(test_data)  
#    
#    start = time.time()
#    for _ in range(10000):
#        _ = test_data.loc[500000, 'B']
#    pandas_time = time.time() - start
#    
#    start = time.time()
#    for _ in range(10000):
#        _ = fast_df.loc[500000, 'B']
#    fastdf_time = time.time() - start
#    
#    assert fastdf_time < pandas_time
#    speedup = pandas_time / fastdf_time
#    print(f"\nGetitem speedup: {speedup:.2f}x")
#
#def test_slice_performance(test_data):
#    fast_df = fdf.from_pandas(test_data) 
#    
#    start = time.time()
#    for _ in range(1000):
#        _ = test_data.loc[250000:750000, 'A':'C']
#    pandas_time = time.time() - start
#    
#    start = time.time()
#    for _ in range(1000):
#        _ = fast_df.loc[250000:750000, 'A':'C']
#    fastdf_time = time.time() - start
#    
#    assert fastdf_time < pandas_time
#    speedup = pandas_time / fastdf_time
#    print(f"\nSlice speedup: {speedup:.2f}x")
#
#def test_shift_performance(test_data):
#    fast_df = fdf.from_pandas(test_data)
#    
#    start = time.time()
#    _ = test_data.shift(1)
#    pandas_time = time.time() - start
#    
#    start = time.time()
#    _ = fast_df.shift(1)
#    fastdf_time = time.time() - start
#    
#    assert fastdf_time < pandas_time
#    speedup = pandas_time / fastdf_time
#    print(f"\nShift speedup: {speedup:.2f}x")
#
#def test_any_performance(test_data):
#    fast_df = fdf.from_pandas(test_data)
#    
#    start = time.time()
#    _ = test_data.any()
#    pandas_time = time.time() - start
#    
#    start = time.time()
#    _ = fast_df.any()
#    fastdf_time = time.time() - start
#    
#    assert fastdf_time < pandas_time
#    speedup = pandas_time / fastdf_time
#    print(f"\nAny speedup: {speedup:.2f}x")
#
#def test_correctness(test_data):
#    fast_df = fdf.from_pandas(test_data)
#    
#    # Test getitem
#    assert np.allclose(fast_df.loc[500000, 'B'], test_data.loc[500000, 'B'])
#    
#    # Test slice
#    assert np.allclose(fast_df.loc[250000:750000, 'A':'C'], test_data.loc[250000:750000, 'A':'C'])
#    
#    # Test shift
#    assert np.allclose(fast_df.shift(1).data, test_data.shift(1).values)
#    
#    # Test any
#    assert np.allclose(fast_df.any(), test_data.any())
#
#if __name__ == "__main__":
#    pytest.main([__file__])