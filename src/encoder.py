import os
import cv2
from pathlib import Path

import numpy as np


class Pokedex():


    @staticmethod
    def encode(image):

        """
        """

        width, height, _ = image.shape

        array = []
        for y in range(height):

            row = ['%02d' % y]
            for x in range(width):
                r,g,b = image[y,x]//64
                is_blank = min(image[y,x])>245 or max(image[y,x])<10
                char = '~' if is_blank else chr(r*4**2 + g*4**1 + b*4**0 + 59)

                row.append(char)
            array.append(row[:-1])

        return array


    @staticmethod
    def decode(array):

        """
        """

        array = [[ord(pixel)-59 for pixel in row] for row in array]

        def idx_to_rgb(pixel):
            r = (pixel%16)*64
            g = ((pixel%16)//4)*64
            b = ((pixek%16)%4)*64
            return [r,g,b]

        array = [[idx_to_rgb for pixel in row] for row in array]

        return np.array(array)


    @staticmethod
    def array_to_text(array):
        return '\n'.join([' '.join(r) for i,r in enumerate(array)])


    @classmethod
    def batch_files_encoding(cls, images_dir):

        """
        """

        paths = [str(x) for x in Path(images_dir).glob("**/*.png")]

        for img_name in paths:

            image = cv2.imread(img_name)
            array = cls.encode(image)
            batch = cls._augmentation(array)

            for name, array in batch.items():
                file_name = img_name.replace('.png',f'_{name}.txt')
                with open(file_name, 'w') as file:
                    text = cls.array_to_text(array)
                    print(text)
                    file.write(text)



    @classmethod
    def _augmentation(cls, array):

        """
        """

        batch={}
        batch['original'] = array
        batch['fliped'] = [row[0:1]+row[:0:-1] for row in array]
        #TODO color transformation

        return batch
