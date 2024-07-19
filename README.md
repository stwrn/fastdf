# FastDF: High-Performance DataFrame for Python

FastDF is a lightning-fast, memory-efficient DataFrame implementation built on top of NumPy, designed to overcome the performance limitations of pandas for basic data operations.

## 🚀 Key Features

- **Blazing Fast**: Up to 126x faster data access compared to pandas
- **Memory Efficient**: Optimized memory usage with NumPy 2D arrays
- **Pandas-Compatible**: Seamless integration with existing pandas-based projects
- **Minimalist**: Focuses on core functionality for maximum performance

## 🎯 Motivation

FastDF was born out of frustration with the sluggish performance of pandas, especially when dealing with large datasets. After exploring various alternatives that either didn't work as expected or introduced complex syntax changes, we realized that for many data analysis tasks, we only need a handful of core features:

- Named columns
- Efficient slicing
- Basic operations like `shift` and `any`

By leveraging the power of NumPy's 2D arrays and implementing only the essential features, FastDF achieves remarkable performance improvements without sacrificing ease of use.

## ⚡ Performance

In our benchmarks, FastDF has shown:

- 126x faster data access compared to pandas
- Significantly faster slicing operations
- Reduced memory footprint

## 🛠 Installation

```bash
pip install git+https://github.com/stwrn/fastdf.git
```

## 🚦 Quick Start

```python
from fastdf import fdf
import pandas as pd
import numpy as np

# Create a pandas DataFrame
pdf = pd.DataFrame({'A': np.random.rand(1000000), 'B': np.random.rand(1000000)})

# Convert to FastDF
fast_df = fdf.from_pandas(pdf)

# Use FastDF with familiar pandas-like syntax
print(fast_df.loc[0:5, 'A'])
print(fast_df['B'].shift(1))
print(fast_df.any())
```

## 🔄 Compatibility

FastDF is designed to be a drop-in replacement for basic pandas operations. You can easily convert your pandas DataFrame to FastDF and continue using the familiar syntax:

```python
# Your existing pandas code
result = df.loc[1000:2000, 'column_name']

# With FastDF
fast_df = fdf.from_pandas(df)
result = fast_df.loc[1000:2000, 'column_name']
```

## 🤝 Contributing

We welcome contributions to FastDF! Whether it's bug reports, feature requests, or code contributions, please feel free to make a pull request or open an issue.

## 📜 License

FastDF is released under the MIT License. See the LICENSE file for more details.

## 🙏 Acknowledgements

Special thanks to the NumPy and pandas teams for their incredible work, which laid the foundation for this project.

---

FastDF is still in active development. We're excited to see how it can help accelerate your data analysis workflows!
