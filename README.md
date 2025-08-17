## 简介

- Image of the Vaidya Black Hole, 简称`IVBH`
- 用来计算被薄吸积盘环绕的Vaidya黑洞图像

## 用法

- 下载所有程序文件后，在根目录创建`Results/`和`imgs/`文件夹
- 首先使用`pip install -r requirements.txt`安装依赖库
- 然后运行`save_brcd_flux.py`程序，生成所有的数据文件
    - 一组`mu`和`theta_0`大约需要30分钟
    - 示例文件全部计算完成大约需要3小时
- 之后依次运行`Fig01.ipynb`至`Fig08.ipynb`文件即可，图片编号顺序与参考文献中一致

> 建议使用`VSCode`编辑器运行代码
> `VSCode`需要安装`python`以及`Jupyter`插件
> `VSCode`详细安装教程可使用使用Google或者询问GPT查询得知

## 参考文献

- Liang, Yu, Sen Guo, Kai Lin, et al. 《Image of the Time-Dependent Black Hole》. Physics Letters B 139745. doi:10.1016/j.physletb.2025.139745.