# Barcode Detector

A Python program that uses concepts of image processing that detects a barcode in an image and draws a bounding box. This is achieved by processing the image through standard deviation, gaussian averaging, thresholding, erosion, dilation and finially detected through connected component analysis.

# Demo

Original image:

![](/images/Barcode1.png)

Dectection:

![](/output_images/Barcode1_output.png)

# Usage

Ensure that the image is a png then direct your terminal to the location of the program.

Then add the appropriate arguments

```
CS373_barcode_detection.py input.png output.png
```

You can also just run the program and a default image will be used. This can be changed on line 255.

# Limitation

This is not accurate 100% of the time and does not detect multiple barcodes.
