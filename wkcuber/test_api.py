from PIL import Image, ImageFilter

from wkcuber.api.Dataset import WKDataset, TiffDataset
from wkcuber.api.Slice import TiffSlice
import numpy as np
from skimage import io

if __name__ == "__main__":

    ds = WKDataset.create("/home/robert/Desktop/temp/simple_wk_dataset", [1])
    mag = ds.add_layer("color", "color", num_channels=3).add_mag(
        "1-1-1", block_len=8, file_len=4
    )
    write_data = np.random.rand(3, 24, 24, 24).astype(np.uint8)
    mag.write(write_data)

    # pixels = np.ones((10, 10, 3), dtype=np.uint32) * 255 * 100
    # print(pixels[0,0,0])

    # img = Image.fromarray(pixels, "RGB")
    # print(np.array(img, dtype=np.uint32)[0,0,0])

    """
    ds_tiff = TiffDataset.create("/home/robert/Desktop/temp/tiff_dataset_2", [1])
    ds_tiff.add_layer("color", "color")
    ds_tiff.get_layer("color").add_mag("1")
    mag = ds_tiff.get_layer("color").get_mag("1")

    data = np.random.rand(250, 250, 20) * 256

    # mag.write(data)
    mag.write(data, offset=(50, 50, 50))

    # data2 = mag.read(size=(300, 300, 10), offset=(0, 0, 0)) # TODO: fails
    data2 = mag.read(size=(300, 300, 10), offset=(0, 0, 50))
    transformed_data = np.moveaxis(data2, 0, -1)
    img = Image.fromarray(data[:, :, 1])
    img.show(data)
    """
    # data2 = mag.read(size=(300, 300, 10), offset=(0, 0, 50))
    # transformed_data = np.moveaxis(data2, 0, -1)
    # img = Image.fromarray(transformed_data[:, :, 1, :], "RGB")
    # img.show(data)

    # ds_tiff = TiffDataset.create("/home/robert/Desktop/temp/tiff_dataset_3", [1])
    # ds_tiff.add_layer("color", "color")
    # ds_tiff.get_layer("color").add_mag("1")
    # mag = ds_tiff.get_layer("color").get_mag("1")
    """
    data = np.zeros((3, 250, 250, 10), dtype=np.uint16)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256

    # mag.write(data)
    # mag.write(data, offset=(50, 50, 50))

    data = np.moveaxis(data, 0, -1)
    print(data.shape)

    io.imsave("/home/robert/Desktop/temp/16_bit_test.tiff", data[:,:,1,:])
    img = Image.open("/home/robert/Desktop/temp/16_bit_test.tiff")
    print(np.array(img).dtype)
    img.show()
    """

    # data2 = mag.read(size=(300, 300, 10), offset=(0, 0, 50)) # TODO: fails because the shape is different
    # transformed_data = np.moveaxis(data2, 0, -1)
    # img = Image.fromarray(data[:, :, 1])
    # img.show(data)

    # data2 = mag.read(size=(300, 300, 10), offset=(0, 0, 50))
    # transformed_data = np.moveaxis(data2, 0, -1)
    # img = Image.fromarray(transformed_data[:, :, 1, :], "RGB")
    # img.show(data)

    """
    img = Image.new('RGB', (250, 250), "black")  # create a new black image
    pixels = img.load()  # create the pixel map

    for i in range(img.size[0]):  # for every col:
        for j in range(img.size[1]):  # For every row
            pixels[i, j] = (i, j, 100)  # set the colour accordingly

    print(np.array(img).shape)  # (250, 250, 3)
    print(np.array(img))

    #img.show()

    img2 = Image.fromarray(np.array(img))
    #img2.show()

    data = np.array(img)
    print((-1,) + data.shape[-2:])

    test_data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    print(test_data)
    print()
    print(np.moveaxis(test_data, -1, 0))
    data = data.reshape((-1,) + data.shape[-2:])

    print(np.moveaxis(np.zeros((100, 200, 3)), -1, 0).shape)
    print(np.moveaxis(np.zeros((3, 100, 200)), 0, -1).shape)

    # write:    (100, 200)      ->  (100, 200)
    # write:    (3, 100, 200)   ->  (100, 200, 3)
    # read:     (100, 200)      ->  (100, 200)
    # read:     (100, 200, 3)   ->  (3, 100, 200)

    dat1 = np.zeros([100, 200])
    dat2 = np.zeros([3, 100, 200])
    dat3 = np.zeros([100, 200])
    dat4 = np.zeros([100, 200, 3])

    print("write:    (100, 200)      ->  (100, 200)")
    print("write:    (3, 100, 200)   ->  (100, 200, 3)")
    print("read:     (100, 200)      ->  (100, 200)")
    print("read:     (100, 200, 3)   ->  (3, 100, 200)")        
        
    print(dat1[-2:].shape)


    # Load an image from the hard drive
    #original = Image.open("/home/robert/Desktop/temp/tiff_dataset_1/color/1/test.0000.tiff")

    # Blur the image
    #blurred = original.filter(ImageFilter.BLUR)

    # Display both images
    #original.show()
    #blurred.show()

    # save the new image
    #blurred.save("/home/robert/Desktop/temp/tiff_dataset_1/color/1/2test.0000.tiff")

    #ds_tiff = TiffDataset.create("/home/robert/Desktop/temp/tiff_dataset_1", [1])
    #ds_tiff.add_layer("color", "color")

    #ds_tiff.get_layer("color").add_mag("1")
    """

    """
    ds_tiff = TiffDataset.open("/home/robert/Desktop/temp/tiff_dataset_1")

    tiff_slice = TiffSlice("/home/robert/Desktop/temp/tiff_dataset_1/color/1")
    tiff_slice.open()

    print(tiff_slice.read(size=(10, 10, 9), offset=(10, 10, 0)))
    tiff_slice.write(np.zeros((10, 10, 9)), offset=(10, 10, 0))
    print(tiff_slice.read(size=(10, 10, 9), offset=(10, 10, 0)))

    #print(tiff_slice.read(size=(100, 100, 9)))
    #tiff_slice.write(np.zeros((100, 100, 9)), offset=(10, 10, 0))
    #print(tiff_slice.read(size=(100, 100, 9)))
    """
    """
    ds = WKDataset.create("/home/robert/Desktop/temp/third_dataset", [1])

    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    #ds.get_layer("color").add_mag("2-2-1")

    
    ds = WKDataset.open("/home/robert/Desktop/temp/first_dataset")
    mag1 = ds.get_layer("color").get_mag("1")
    #mag2 = ds.get_layer("color").get_mag("2-2-1")

    #data = mag2.read(size=(10, 10, 10), offset=(5, 5, 5))
    #print("zeroed data", data)
    #data[0][0][0] = 1
    
    # mag2.write(data, offset=(5, 5, 5))  # Error: When writing compressed files, each file has to be written as a whole. Please pad your data so that all cubes are complete and the write position is block-aligned.
    # new_data = mag2.read(size=(10, 10, 10), offset=(5, 5, 5))
    # print("new data", new_data)

    data = mag1.read(size=(10, 10, 10), offset=(5, 5, 5))
    print("zeroed data", data.shape)
    print("zeroed data", data)
    data[0][0][0] = data[0][0][0] + 1
    mag1.write(data, offset=(5, 5, 5))
    new_data = mag1.read(size=(10, 10, 10), offset=(5, 5, 5))
    print("new data", new_data)

    """
    # ds.get_layer("color").delete_mag("2-2-1")

    # ds.delete_layer("color")

    # ds2 = WKDataset.open("/home/robert/Desktop/temp/second_dataset")
    # ds2.properties.export_as_json()
