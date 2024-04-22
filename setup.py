from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == "__main__":
    setup(
        name="spherical_mask",
        version="1.0",
        description="spherical_mask",
        author="sangyun shin",
        packages=["spherical_mask"],
        package_data={"spherical_mask.ops": ["*/*.so"]},
        ext_modules=[
            CUDAExtension(
                name="spherical_mask.ops.ops",
                sources=[
                    "spherical_mask/ops/src/isbnet_api.cpp",
                    "spherical_mask/ops/src/isbnet_ops.cpp",
                    "spherical_mask/ops/src/cuda.cu",
                ],
                extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
