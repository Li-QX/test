vision\setup.py
# torchvision源码编译命令
python setup.py bdist_wheel # 该方式会在torchvision目录下生成一个dist目录，里面有编译好的whl包，然后 pip install xxx.whl
python setup.py develop # 该方式会以开发者模式安装，修改torchvision的python源码后，不需要编译版本包可直接生效，修改了cpp+cu源码需要再次执行该命令即可
python setup.py clean # 该命令会clean掉已经编译的内容，后面再次执行上面两个命令即可

# 导入setup
from setuptools import find_packages, setup
# 导入 torch中的CppExtension和CUDAExtension，用于后面编译torchvision的cpp+cuda源码
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension

# vision\version.txt : 0.16.0a0
version_txt = os.path.join(cwd, "version.txt")
with open(version_txt) as f:
    version = f.readline().strip()

package_name = "torchvision" # 设置版本包的名称

pytorch_dep = "torch"
if os.getenv("PYTORCH_VERSION"):
    pytorch_dep += "==" + os.getenv("PYTORCH_VERSION")

requirements = [
    "numpy",
    "requests",
    pytorch_dep,
]

if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    write_version_file()

    with open("README.md") as f:
        readme = f.read()

    setup(
        # Metadata
        name=package_name, # 传递包名字
        version=version, # 版本号
        author="PyTorch Core Team",
        author_email="soumith@pytorch.org",
        url="https://github.com/pytorch/vision",
        description="image and video datasets and models for torch deep learning",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="BSD",
        # Package info
        packages=find_packages(exclude=("test",)),
        package_data={package_name: ["*.dll", "*.dylib", "*.so", "prototype/datasets/_builtin/*.categories"]},
        zip_safe=False,
        install_requires=requirements,# 编译依赖，如torch，即需要先安装torch
        extras_require={
            "scipy": ["scipy"],
        },
        ext_modules=get_extensions(), # 最关键的地方，后面详细分析
        python_requires=">=3.8",# 依赖的python版本
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchvision", "csrc") # 获取ccsrc的路径

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp")) + glob.glob(
        os.path.join(extensions_dir, "ops", "*.cpp")
    )
    source_cpu = (
        glob.glob(os.path.join(extensions_dir, "ops", "autograd", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "ops", "cpu", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "ops", "quantized", "cpu", "*.cpp"))
    )
    
    debug_mode = os.getenv("DEBUG", "0") == "1"
    print(f"  DEBUG: {debug_mode}") # release还是debug模式
    
    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    print(f"  NVCC_FLAGS: {nvcc_flags}") # nvcc编译选项
    
    is_rocm_pytorch = False

    if torch.__version__ >= "1.5":
        from torch.utils.cpp_extension import ROCM_HOME

        is_rocm_pytorch = (torch.version.hip is not None) and (ROCM_HOME is not None)

    if is_rocm_pytorch:
        from torch.utils.hipify import hipify_python
        # hip的工具会将cuda代码转换为HIP代码，然后传得给HIPCC编译器编译
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="torchvision/csrc/ops/cuda/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )
        source_cuda = glob.glob(os.path.join(extensions_dir, "ops", "hip", "*.hip"))
        # Copy over additional files
        for file in glob.glob(r"torchvision/csrc/ops/cuda/*.h"):
            shutil.copy(file, "torchvision/csrc/ops/hip")
    else:
        source_cuda = glob.glob(os.path.join(extensions_dir, "ops", "cuda", "*.cu"))

    source_cuda += glob.glob(os.path.join(extensions_dir, "ops", "autocast", "*.cpp"))
    
    extension = CppExtension # 默认设置为cpp

    define_macros = []

    extra_compile_args = {"cxx": []}
    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or force_cuda:
        extension = CUDAExtension # 修改为cuda
        sources += source_cuda
        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            if nvcc_flags == "":
                nvcc_flags = []
            else:
                nvcc_flags = nvcc_flags.split(" ")
        else:
            define_macros += [("WITH_HIP", None)]
            nvcc_flags = []
        extra_compile_args["nvcc"] = nvcc_flags
        
        sources = [os.path.join(extensions_dir, s) for s in sources] # 收集到ccsrc下面的所有需要编译的源码：cpp or cpp+cuda

    include_dirs = [extensions_dir]
    
    ext_modules = [ # ext_modules 会被get_extensions返回后，传递给setup函数
        extension( # extension is CppExtension  or CUDAExtension 
            "torchvision._C",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    
    
