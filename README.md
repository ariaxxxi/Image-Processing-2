# Image-Processing-2

#Filters on Color Images

Separating our color image into three separate greyscale images (one for each channel: red, green, and blue), applying the same 'greyscale' filter to each one, and then recombining the results together into a color image.

Inverted, Blurred, Sharpen, Edges detection

#Cascade of Filters

#Seam Carving

The idea behind seam carving is that, when we remove a pixel from each row to decrease the size of an image, instead of removing a straight vertical line, we instead find and remove connected 'paths' of pixels from the top to the bottom of the image, with one pixel in each row. Each time we want to decrease the horizontal size of the image by one pixel, we start by finding the connected path from top to bottom that has the minimum total "energy," removing the pixels contained therein. To shrink the image further, we can apply this process repeatedly.





