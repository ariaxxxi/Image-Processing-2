#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# CODE FROM LAB 1 (replace with your code)
def get_pixel(image, x, y): 

    index = x + y*image['width']
    return image['pixels'][index] 

def set_pixel(result, x, y, newcolor):
    # for y in range(result['height']):
    #     for x in range(result['width']):
    result['pixels'].append(newcolor)
    return result['pixels']  


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }
    for y in range(image['height']):
        for x in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result  

def inverted(image):
    return apply_per_pixel(image, lambda c: 255 - c)

def correlate(image, kernel):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }

    def convolute(img, x, y, kernel):
        val = 0
        mid = math.floor(len(kernel) / 2)
        #edge effect
        for i in range(len(kernel)):
            for j in range(len(kernel)):
                x_offset, y_offset = x + i - mid, y + j - mid
                if x_offset < 0: 
                    x_offset = 0
                elif x_offset >= img['width']:
                    x_offset = img['width']-1

                if y_offset < 0:
                    y_offset = 0
                elif y_offset >= img['height']:
                    y_offset = img['height']-1
                #correlated color     
                val += kernel[j][i] * get_pixel(img, x_offset, y_offset)
        # print(val)
        return val

    for y in range(image['height']):
        for x in range(image['width']):
            set_pixel(result, x, y, convolute(image, x, y, kernel))
    # print(result)        
    return result


def round_and_clip_image(image):

    
    def round_and_clip_pixel(pixel):
        if pixel < 0:
            return 0
        elif pixel > 255:
            return 255
        else:
            return round(pixel)

    pixels = [round_and_clip_pixel(p) for p in image['pixels']]

    return {'height': image['height'], 'width': image['width'], 'pixels': pixels}


def blurred(image,n):
 
    k = []
    num = 1/(n*n)
    for i in range(n):
        k.append(num)
    kernel = [k for i in range(n)]
    # print(kernel)
    return round_and_clip_image(correlate(image,kernel))
    # return image


def sharpened(image, n):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }
    blurred_image = blurred(image,n)
    for y in range(image['height']):
        for x in range(image['width']):
             blurred_color = get_pixel(blurred_image,x,y)
             original_color = get_pixel(image,x,y)
             sharpened_color = 2*original_color - blurred_color
             set_pixel(result, x, y, sharpened_color)
    return round_and_clip_image(result)

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES
def edges(image):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }
    k1 = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
    k2 = [[-1, -2, -1],[0,  0,  0],[1,  2,  1]]
    im1 = correlate(image,k1)
    im2 = correlate(image,k2)

    for y in range(image['height']):
        for x in range(image['width']):
            pix1 = get_pixel(im1,x,y)
            pix2 = get_pixel(im2,x,y)
            edge_color = round(math.sqrt(pix1**2 + pix2**2))
            set_pixel(result, x, y, edge_color)

    return round_and_clip_image(result)


# LAB 2 FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def function_f(image):
        result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
        }   
        def divide_red(image):
            red_image = {
            'height': image['height'],
            'width': image['width'],
            'pixels': [], }

            for y in range(image['height']):
                for x in range(image['width']):
                    index = x + y * image['width']
                    r = image['pixels'][index][0]#rgb of original image
                    red_image['pixels'].append(r)
            # return round_and_clip_image(red_image) 
            return red_image

        def divide_green(image):
            green_image = {
            'height': image['height'],
            'width': image['width'],
            'pixels': [], }
            
            for y in range(image['height']):
                for x in range(image['width']):
                    index = x + y * image['width']
                    g = image['pixels'][index][1]
                    green_image['pixels'].append(g)
            # return round_and_clip_image(green_image) 
            return green_image 

        def divide_blue(image):
            blue_image = {
            'height': image['height'],
            'width': image['width'],
            'pixels': [], }
            
            for y in range(image['height']):
                for x in range(image['width']):
                    index = x + y * image['width']
                    b = image['pixels'][index][2]
                    blue_image['pixels'].append(b)
            # return round_and_clip_image(blue_image) 
            return blue_image

        red_filt = filt(divide_red(image))  
        green_filt = filt(divide_green(image))
        blue_filt = filt(divide_blue(image))

        for y in range(image['height']):
            for x in range(image['width']):
                index = x + y * image['width']
                r = red_filt['pixels'][index]#rgb of original image
                g = green_filt['pixels'][index]
                b = blue_filt['pixels'][index]
                newcolor = (r,g,b)
                result['pixels'].append(newcolor)
        return result

    return function_f


def color_inverted(image):      
    
    return color_filter_from_greyscale_filter(inverted)(image)
         

def make_blur_filter(n):
    def blurred_function(image):
        return blurred(image,n)

    return blurred_function    


def make_sharpen_filter(n):
    def sharpen_image(image):
        return sharpened(image,n)

    return sharpen_image

def make_edge_filter(image):
    return color_filter_from_greyscale_filter(edges)(image)
   

def filter_cascade(filters):
    
    def f(image):
        for fil in filters:
            image = fil(image)
        return image    
    return f
# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }

    for i in range(ncols):
        def f (image):
            image = image_without_seam(image, seam)
        return f
    return result
# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [],
    }
    for y in range(image['height']):
        for x in range(image['width']): 
            index = x + y * image['width']
            # print(image['pixels'][index])
            r = image['pixels'][index][0]#rgb of original image
            g = image['pixels'][index][1]
            b = image['pixels'][index][2]

            v = round(0.299*r + 0.587*g + 0.114*b)
            # newcolor = (v,v,v)

            result['pixels'].append(v)
    # print(result)
    return result

def compute_energy(image):
    
    return edges(image)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    result = {
        'height': energy['height'],
        'width': energy['width'],
        'pixels': [],
    }

    w = energy['width']
    h = energy['height']

    for y in range(0,h):
        for x in range(0,w):
            index = x + y * w
            color = energy['pixels'][index]

            index1 = x-1 + (y-1)*w
            index2 = x + (y-1)*w
            index3 = x+1 + (y-1)*w

            if y == 0:
                v = color
                result['pixels'].append(v)
            else:    
                if x==0:
                    adjacent = [ result['pixels'][index2],result['pixels'][index3] ]

                elif x == w-1:
                    adjacent = [ result['pixels'][index1],result['pixels'][index2] ]

                else:
                    adjacent = [ result['pixels'][index1],result['pixels'][index2],result['pixels'][index3] ]
               
                v = color + min(adjacent)
                result['pixels'].append(v)
    return result       

def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    w = cem['width']
    h = cem['height']
    p = cem['pixels']
    seam = []
    x_location = []

    lastrow = p[w*h-w : ]
    min_value = min(lastrow)  #min pixel of last row    
    min_index = lastrow.index(min_value) + w*(h-1) # index of the min pixel in last row
    xPos = min_index - w*(h-1) #xPos of min pixel in last row
    
    seam.append(min_index)
    x_location.append(xPos)
    # print(xPos)

    def min_pixel (x,y,image): #decide the min value pixel of (x,y)'s previous row
        
        w = image['width']
        h = image['height']
        p = image['pixels']

        index1 = x-1 + (y-1)*w
        index2 = x + (y-1)*w
        index3 = x+1 + (y-1)*w

        

        if x == 0:
            raw = [image['pixels'][index2],image['pixels'][index3]]
            minimum_px = min(raw)
            if minimum_px == p[index2]:
                 min_index = index2
            elif minimum_px == p[index3]:
                min_index = index3

        elif x == image['width'] :
            raw = [image['pixels'][index1], image['pixels'][index2]]
            minimum_px = min(raw)
            if minimum_px == p[index1]:
                min_index = index1
            elif minimum_px == p[index2]:
                min_index = index2

        else:
            raw = [image['pixels'][index1],image['pixels'][index2], image['pixels'][index3]]
            minimum_px = min(raw)
            if minimum_px == p[index1]:
                min_index = index1
            elif minimum_px == p[index2]:
                min_index = index2
            elif minimum_px == p[index3]:
                min_index = index3

        min_xPos = min_index - w*(y-1)
        seam.append(min_index)

        return min_xPos #min pixel above (x,y)
    

    def get_x_up(x): #get the seam start from x,y
        x_up = x
        for y in range (h-1,0, -1):
            x_up = min_pixel(x_up,y,cem)
            x_location.append(x_up)
               
        return x_up
    
    
    get_x_up(xPos)   

    # seam.reverse()    
    print(x_location)
    print(seam)
    

    return seam
    


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    w = image['width']
    h = image['height']
    p = image['pixels']

    result = {
        'height': image['height'],
        'width': image['width']-1,
        'pixels': [],
        }

    seam.reverse()

    for y in range(h):  
        for x in range(w):
            index = x + y * w     
            color = image['pixels'][index]
        # result['pixels'].pop(seam[y])
            if index not in seam:
                result['pixels'].append(color)

    print(result)
    return result

# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # im1 = load_color_image('test_images/cat.png')
    # inverted_color_cat = color_inverted(im1)  
    # save_color_image(inverted_color_cat, 'cat_inverted.png', mode='PNG')

    # im2 = load_color_image('test_images/python.png')
    # blurred_color = make_blur_filter(9)(im2)
    # save_color_image(blurred_color, 'python_blur.png', mode='PNG')

    # im3 = load_color_image('test_images/sparrowchick.png')
    # sharpen_color = make_sharpen_filter(7)(im3)
    # save_color_image(sharpen_color, 'sparrowchick_sharpen.png', mode='PNG')

    # im4 = load_color_image('test_images/sparrowchick.png')
    # sharpen_color = make_edge_filter(im4)
    # save_color_image(sharpen_color, 'sparrowchick_edge.png', mode='PNG')
  
    im5 = load_color_image('test_images/frog.png')
    # filter1 = make_edge_filter
    # filter2 = make_blur_filter(5)
    # filt = filter_cascade([filter1,filter2,filter1,filter2])(im5)
    # save_color_image(filt, 'frog_cascade.png', mode='PNG')

    # sharpen_color = make_edge_filter(im5)
    # save_color_image(sharpen_color, 'frog_edge.png', mode='PNG')

    # im6 = load_color_image('test_images/pattern.png')
    # save_color_image(cumulative_energy_map(im6),'grey.png',mode='PNG')
    
    # im_cumulative_map = {'width': 9, 'height': 5, 'pixels': [473,0,  40,   10, 28, 3, 40,  265, 473,
    #                                                         160, 160, 0,   1,  10,  28, 0,  160, 160, 
    #                                                         415, 3,   10,  22, 14, 22, 10, 218, 415, 
    #                                                         0,   2,   0,   10,  3, 10, 0,  265, 473, 
    #                                                         520, 295, 20,  32, 10, 32, 41, 295, 520]}
    #                                                        # 0   1    2    3   4  5    6   7     8
    # im_list = minimum_energy_seam(im_cumulative_map)
    # expected = [2, 11, 21, 31]
    # print(im_list)


    im_cumulative_map = {'width': 9, 'height': 4, 'pixels': [160, 160, 0, 28, 0, 28, 0, 160, 160, 415, 218, 10, 22, 14, 22, 10, 218, 415, 473, 265, 40, 10, 28, 10, 40, 265, 473, 520, 295, 41, 32, 10, 32, 41, 295, 520]}
    im_list = minimum_energy_seam(im_cumulative_map)
    expected = [31, 21, 11, 2]

    # print(im_list)
    print(expected)


    save_color_image(seam_carving(im5, 100),'seam.png',mode='PNG')

