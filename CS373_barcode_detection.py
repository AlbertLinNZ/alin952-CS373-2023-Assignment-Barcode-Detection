# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def combineRGB(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    pixel_array_rgb = []
    for x in range(image_height):
        pixel_row_rgb = []
        for y in range (image_width):
            pixel_row_rgb.append([pixel_array_r[x][y], pixel_array_g[x][y] , pixel_array_b[x][y]])
        pixel_array_rgb.append(pixel_row_rgb)
    return pixel_array_rgb;

def padZero(pixel_array, image_width, image_height, amount):
    for row in pixel_array:
        for i in range(amount):
            row.insert(0, 0)
            row.append(0)
    for i in range(amount):
        pixel_array.insert(0, [0] * (image_width + amount * 2))
        pixel_array.append([0] * (image_width + amount * 2))
    return pixel_array

def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_width):
        for j in range(image_height):
            greyscale_pixel_array[j][i] = int(round(0.299 * pixel_array_r[j][i] + 0.587 * pixel_array_g[j][i] + 0.114 * pixel_array_b[j][i])) 
    return greyscale_pixel_array

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    result_array = [[float(0) for pixel in row] for row in result_array]
    for i in range(2, image_width - 2):
        for j in range(2, image_height - 2):
            slice_5x5 = (pixel_array[j + 2][i-2:i+3] +
                         pixel_array[j + 1][i-2:i+3] +
                         pixel_array[j][i-2:i+3] +
                         pixel_array[j - 1][i-2:i+3] +
                         pixel_array[j - 2][i-2:i+3] )
            mean = sum(slice_5x5) / len(slice_5x5)  
            var  = sum(pow(x-mean,2) for x in slice_5x5) / len(slice_5x5)  
            std  = math.sqrt(var) 
            result_array[j][i] += std
    return result_array

def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    boundary_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for row in pixel_array:
        row.insert(0, row[0])
        row.append(row[len(row) - 1])
    pixel_array.insert(0, pixel_array[0])
    pixel_array.insert(len(pixel_array) - 1, pixel_array[len(pixel_array) - 1])
    for i in range(image_width):
        for j in range(image_height):
            gauss_array = (pixel_array[j][i: i + 3] + 
                            pixel_array[j + 1][i: i + 3] +
                            pixel_array[j + 2][i: i + 3])
            for x in range(len(gauss_array)):
                if x % 2 == 1:
                    gauss_array[x] *= 2
            gauss_array[4] *= 4
            boundary_array[j][i] = sum(gauss_array) / 16
    return boundary_array

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    for i in range(image_width):
        for j in range(image_height):
            if pixel_array[j][i] >= threshold_value:
                pixel_array[j][i] = 255
            elif  pixel_array[j][i] < threshold_value:
                pixel_array[j][i] = 0
    return pixel_array

def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    result_array = [[float(0) for pixel in row] for row in result_array]

    padZero(pixel_array, image_width, image_height, 2)
    
    for i in range(image_width):
        for j in range(image_height):
            slice_5x5 = (pixel_array[j + 4][i:i + 5] + pixel_array[j + 3][i:i + 5] +
                        pixel_array[j + 2][i:i + 5] + pixel_array[j + 1][i:i + 5] + 
                        pixel_array[j][i:i+5])
            if all(pixel > 0 for pixel in slice_5x5):
                result_array[j][i] = 1
            else: 
                result_array[j][i] = 0
    return result_array

def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    result_array = createInitializedGreyscalePixelArray(image_width, image_height)
    result_array = [[float(0) for pixel in row] for row in result_array]

    padZero(pixel_array, image_width, image_height, 2)
    
    for i in range(image_width):
        for j in range(image_height):
            slice_5x5 = (pixel_array[j + 4][i:i + 5] + pixel_array[j + 3][i:i + 5] +
                        pixel_array[j + 2][i:i + 5] + pixel_array[j + 1][i:i + 5] + 
                        pixel_array[j][i:i+5])
            if any(pixel > 0 for pixel in slice_5x5):
                result_array[j][i] = 1
            else: 
                result_array[j][i] = 0
    return result_array

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    label_dict = {}
    visited = set()
    current_label = 1

    for j in range(image_height):
        for i in range(image_width):
            if (pixel_array[j][i] != 0 and (i, j) not in visited):
                q = Queue()
                q.enqueue((i, j))
                label_dict[current_label] = set()
                while not q.isEmpty():
                    curr_i, curr_j = q.dequeue()
                    visited.add((curr_i, curr_j))
                    label_dict[current_label].add((curr_i, curr_j))
                    pixel_array[curr_j][curr_i] = current_label
                    if (curr_i - 1 >= 0 and pixel_array[curr_j][curr_i - 1] != 0 and (curr_i - 1, curr_j) not in visited):
                        q.enqueue((curr_i - 1, curr_j))
                        visited.add((curr_i - 1, curr_j))
                    if (curr_i + 1 < image_width and pixel_array[curr_j][curr_i + 1] != 0 and (curr_i + 1, curr_j) not in visited):
                        q.enqueue((curr_i + 1, curr_j))
                        visited.add((curr_i + 1, curr_j))
                    if (curr_j - 1 >= 0 and pixel_array[curr_j - 1][curr_i] != 0 and (curr_i, curr_j - 1) not in visited):
                        q.enqueue((curr_i, curr_j - 1))
                        visited.add((curr_i, curr_j - 1))
                    if (curr_j + 1 < image_height and pixel_array[curr_j + 1][curr_i] != 0 and (curr_i, curr_j + 1) not in visited):
                        q.enqueue((curr_i, curr_j + 1))
                        visited.add((curr_i, curr_j + 1))
                current_label += 1
    return pixel_array, label_dict



# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode1"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(4, 3) #change back to 2x2
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here
    rgb_px_array =  combineRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)
    greyscale_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    std_array = computeStandardDeviationImage5x5(greyscale_array, image_width, image_height)
    gauss_array = std_array
    
    for i in range(6):
        gauss_array = computeGaussianAveraging3x3RepeatBorder(gauss_array, image_width, image_height)
    threshold_array = computeThresholdGE([row[:] for row in gauss_array], 15, image_width, image_height)
    erosion_array = [row[:] for row in threshold_array]
    
    for i in range(5):
        erosion_array = computeErosion8Nbh5x5FlatSE(erosion_array, image_width, image_height)
    dilation_array = [row[:] for row in erosion_array]
    
    for i in range(4):
        dilation_array = computeDilation8Nbh5x5FlatSE(dilation_array, image_width, image_height)
    segment_array, segments = computeConnectedComponentLabeling([row[:] for row in dilation_array], image_width, image_height)
    largest_segment_size = 0
    largest_segment_index = 0
    
    for key in segments.keys():
        max_x = max(segments[key], key=lambda item: item[0])[0] + 0.5
        max_y = max(segments[key], key=lambda item: item[1])[1] + 0.5
        min_x = min(segments[key], key=lambda item: item[0])[0] - 0.5
        min_y = min(segments[key], key=lambda item: item[1])[1] - 0.5
        width = max_x - min_x
        height = max_y - min_y
        num_foreground_pixel = len(segments[key])
        density = num_foreground_pixel / (width * height)
        x_ratio = width / height
        y_ratio = height / width
        if len(segments[key]) > largest_segment_size and density > 0.5 and x_ratio < 1.8 and y_ratio < 1.8:
            largest_segment_size = len(segments[key])
            largest_segment_index = key

    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    
    if largest_segment_index != 0:
        max_x = max(segments[largest_segment_index], key=lambda item: item[0])[0] + 0.5
        max_y = max(segments[largest_segment_index], key=lambda item: item[1])[1] + 0.5
        min_x = min(segments[largest_segment_index], key=lambda item: item[0])[0] - 0.5
        min_y = min(segments[largest_segment_index], key=lambda item: item[1])[1] - 0.5
    else:
        print("No barcode found!")
        
    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    bbox_min_x = min_x
    bbox_max_x = max_x
    bbox_min_y = min_y 
    bbox_max_y = max_y 

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(rgb_px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)
    axs1[2, 0].imshow(std_array, cmap='gray')
    axs1[2, 1].imshow(gauss_array, cmap='gray')
    axs1[2, 2].imshow(threshold_array, cmap='gray')
    axs1[3, 0].imshow(erosion_array, cmap='gray')
    axs1[3, 1].imshow(dilation_array, cmap='gray')
    axs1[3, 2].imshow(segment_array, cmap='gray')
    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)
    
    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
