# 💻whl安装包使用方法
1. 进入GitHub的[Releases](https://github.com/SeanNg997/XrKit-Package/releases)中下载最新的whl安装包；
2. 使用命令行进入whl安装包所在路径；
3. 使用`pip install xrkit-1.0-py3-none-any.whl`进行安装；
4. 在自己的代码中使用`import xrkit`使用。



# 💡XrKit使用指南

1. `User Guide.ipynb`文件提供了详细的教程；
2. 使用`xrkit.show_functions()`可查看xrkit库中的所有函数名称及简介；
3. 安装XrKit不会附带安装运行代码所必须的库，请检查以下库是否可以正常导入：

```python
import math
import chardet
import numpy
import pandas
import geopandas
import rasterio
import rioxarray
import xarray
import multiprocessing
import tqdm
import affine
import pyproj
import geocube
```

---

***更新日期：2025.3.11***