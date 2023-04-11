ScaleConvertion
-----------------------------------------------------------------

ResTransformer-based surface reflectance scale conversion tool

As a critical component of many remote sensing satellites and remote sensing model validation, image scale surface quantitative parameters are often affected by scale effects in the acquisition process, resulting in deviations in the accuracy of image scale parameters. We propose ResTransformer, a deep learning model for scale conversion of surface reflectance combined with UAV images, which can adapt to surface reflectance scale conversion scenarios with different sizes, heterogeneous sample areas and arbitrary sampling methods. The method provides a promising, highly accurate, robust approach for image element-scale surface reflectance scale conversion.

1. Processing hyperspectral images

.. code:: python

    from ScaleConvertion.SC import read_tiff

    if __name__ == '__main__':
        tiff_path=r"D:\Transformer\test\dat1.tif"
        read_tiff(feature_bands=[[0,8],[8,16],[16,24]],save_dir=r"D:\Transformer\test",tiff_path=tiff_path,sen_alt=[9,9,9,9,9,9])


2. Processing uav images

.. code:: python

    from ScaleConvertion.SC import read_tiff, read_png

    if __name__ == '__main__':
        tiff_path = r"D:\Transformer\test\dat1.tif"
        img_path = r"G:\UAVPICTURRE\13裸土-上东下西-H975m-6m-116.JPG"
        save_dir = "tmp"
        read_png(img_path=img_path, save_dir=save_dir, pixel=[2, 4, 8, 10, 20, 30])
