This program counts coffee beans in an image using computer vision techniques. 
It first loads the image and resizes it to a fixed width so that detection parameters remain consistent regardless of the original resolution.
The image is then converted to grayscale, enhanced for contrast, and slightly blurred to reduce noise.
Next, the program separates the beans from the background using adaptive thresholding and cleans up small imperfections with morphological operations.
It applies a distance transform to highlight the centres of the beans, then detects local intensity peaks to identify likely bean centres, even when beans are touching.
Using these detected centres as markers, the program performs watershed segmentation to separate individual beans. 
It filters out very small regions to remove noise, outlines each valid bean, marks its centre, and counts them. 
The final result is rounded to the nearest ten and printed to the console, along with processing time. Intermediate images are saved at each major stage for inspection.

This program is open source and is open to be used and modified.
It can be used to count the number of objects in any image by renaming "coffe_beans.jpg" to the name of your image.
Note that this program has not been tested with other images and may produce incorrect results.
