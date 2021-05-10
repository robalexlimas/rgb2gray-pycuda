import numpy as np

from image import array2image, array2vector, load_image, save_image
from rgb2gray import rgb2gray


def main():
    input_name = 'shingeki.jpeg'
    output_name = 'shingeki_gray.jpeg'

    input_image = load_image(input_name)
    image = np.array(
        input_image.getdata()).reshape(input_image.size[1], input_image.size[0], 3
    )
    
    image_vector = array2vector(image)
    height, width = input_image.size[1], input_image.size[0]

    output_image_array = rgb2gray(image_vector, height, width)
    output_image = array2image(output_image_array)
    save_image(output_image, output_name)


if __name__=='__main__':
    main()
