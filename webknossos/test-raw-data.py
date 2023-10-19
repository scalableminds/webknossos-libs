import numpy as np
import requests
from PIL import Image


def main():
    local_dataset = "e75_zarr_local"
    # streamed_dataset = "l4_v2_sample_streamed"
    # fetch(local_dataset)
    fetch(local_dataset)


def fetch(ds_name):
    width = 100
    height = 100
    uri = f"http://localhost:9000/data/datasets/sample_organization/{ds_name}/layers/color/data?token=secretSampleUserToken&x=0&y=0&z=78&width={width}&height={height}&depth=1&mag=1-1-1"
    response = requests.get(uri)
    array = np.frombuffer(response.content, dtype=np.uint16)
    array = array.reshape((width, height))

    print(array.shape)
    array = array * 30

    im = Image.fromarray(array)
    im.save(f"{ds_name}.png")


if __name__ == "__main__":
    main()
