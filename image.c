#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RED 0
#define GREEN 1
#define BLUE 2

#define PXL(y, x, w, c) (((y) * w + (x)) * 3 + (c))

typedef struct {
    int width;
    int height;
    int max_value;
    unsigned char *data;
} PPMImage;

PPMImage *read_ppm_file(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    
    char format[3];
    int width, height, max_value;
    
    fscanf(fp, "%s\n", format);
    if (strcmp(format, "P6") != 0) {
        printf("Invalid PPM format.\n");
        fclose(fp);
        return NULL;
    }
    
    fscanf(fp, "%d %d\n", &width, &height);
    fscanf(fp, "%d\n", &max_value);
    
    // Allocate memory for the image data
    unsigned char *data = (unsigned char *)malloc(width * height * 3);
    fread(data, 1, width * height * 3, fp);
    
    fclose(fp);
    
    PPMImage *image = (PPMImage *)malloc(sizeof(PPMImage));
    image->width = width;
    image->height = height;
    image->max_value = max_value;
    image->data = data;
    
    return image;
}

void write_ppm_file(const char *filename, PPMImage *image) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return;
    }
    
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "%d\n", image->max_value);
    
    fwrite(image->data, 1, image->width * image->height * 3, fp);
    
    fclose(fp);
}

PPMImage* invert_color(PPMImage* image) {
    for (int i = 0; i < image->height; ++i) {
        for (int j = 0; j < image->width; ++j) {
            image->data[PXL(i, j, image->width, RED)] = (unsigned char)255-image->data[PXL(i, j, image->width, RED)];
            image->data[PXL(i, j, image->width, GREEN)] = (unsigned char)255-image->data[PXL(i, j, image->width, GREEN)];
            image->data[PXL(i, j, image->width, BLUE)] = (unsigned char)255-image->data[PXL(i, j, image->width, BLUE)];
        }
    }
    return image;
}

void flip_pixel(PPMImage* image, unsigned char* new_data, int row, int column) {
    new_data[PXL(row, image->width-column, image->width, RED)] = image->data[PXL(row, column, image->width, RED)];
    new_data[PXL(row, image->width-column, image->width, GREEN)] = image->data[PXL(row, column, image->width, GREEN)];
    new_data[PXL(row, image->width-column, image->width, BLUE)] = image->data[PXL(row, column, image->width, BLUE)];
}

PPMImage* flip_image(PPMImage* image) {
    unsigned char* new_data = (unsigned char*)malloc(image->width*image->height*3);
    for (int i = 0; i < image->height; ++i) {
        for (int j = 0; j < image->width; ++j) {
            flip_pixel(image, new_data, i, j);
        }
    }
    image->data = new_data;
    return image;
}

void brighten_pixel(PPMImage* image, int i, int j, int factor) {
    int red = image->data[PXL(i, j, image->width, RED)]+factor;
    int green = image->data[PXL(i, j, image->width, GREEN)]+factor;
    int blue = image->data[PXL(i, j, image->width, BLUE)]+factor;
    if (red > 255) {
        red = 255;
    }
    if (green > 255) {
        green = 255;
    }
    if (blue > 255) {
        blue = 255;
    }
    image->data[PXL(i, j, image->width, RED)] = (unsigned char)red;
    image->data[PXL(i, j, image->width, GREEN)] = (unsigned char)green;
    image->data[PXL(i, j, image->width, BLUE)] = (unsigned char)blue;
}

PPMImage* brighten_image(PPMImage* image, int factor) {
    for (int i = 0; i < image->height; ++i) {
        for (int j = 0; j < image->width; ++j) {
            brighten_pixel(image, i, j, factor);
        }
    }
    return image;
}

void rotate_pixel(PPMImage* image, unsigned char* new_data, int i, int j) {
    new_data[PXL(j, image->height-i, image->height, RED)] = image->data[PXL(i, j, image->width, RED)];
    new_data[PXL(j, image->height-i, image->height, GREEN)] = image->data[PXL(i, j, image->width, GREEN)];
    new_data[PXL(j, image->height-i, image->height, BLUE)] = image->data[PXL(i, j, image->width, BLUE)];
}

PPMImage* rotate_image(PPMImage* image) {
    unsigned char* new_data = (unsigned char*)malloc(image->width*image->height*3);
    for (int i = 0; i < image->height; ++i) {
        for (int j = 0; j < image->width; ++j) {
            rotate_pixel(image, new_data, i, j);
        }
    }
    int tmp = image->height;
    image->height = image->width;
    image->width = tmp;
    image->data = new_data;
    return image;
}

void extract_pixel(PPMImage* image, unsigned char* new_data, int i, int j, int width, int min_height, int min_width) {
    new_data[PXL(i-min_height, j-min_width, width, RED)] = image->data[PXL(i, j, image->width, RED)];
    new_data[PXL(i-min_height, j-min_width, width, GREEN)] = image->data[PXL(i, j, image->width, GREEN)];
    new_data[PXL(i-min_height, j-min_width, width, BLUE)] = image->data[PXL(i, j, image->width, BLUE)];
}

PPMImage* extract_image_part(PPMImage* image, int min_height, int max_height, int min_width, int max_width) {
    if (max_height > image->height || max_width > image->width || min_height < 0 || min_width < 0 || min_width > max_width || min_height > max_height) {
        printf("invalid input\n");
        return NULL;
    }
    unsigned char* new_data = (unsigned char*) malloc((max_height-min_height)*(max_width-min_width)*3);
    for (int i = min_height; i < max_height; ++i) {
        for (int j = min_width; j < max_width; ++j) {
            extract_pixel(image, new_data, i, j, max_width-min_width, min_height, min_width);
        }
    }
    image->height = max_height-min_height;
    image->width = max_width-min_width;
    image->data = new_data;
    return image;
}


int main() {
    PPMImage* image = read_ppm_file((char*)"a.ppm");
    PPMImage* inverted_image = invert_color(image);
    image = read_ppm_file((char*)"a.ppm");
    PPMImage* brightened_image = brighten_image(image, 100);
    image = read_ppm_file((char*)"a.ppm");
    PPMImage* flipped_image = flip_image(image);
    image = read_ppm_file((char*)"a.ppm");
    PPMImage* rotated_image = rotate_image(image);
    image = read_ppm_file((char*)"a.ppm");
    PPMImage* extracted_image = extract_image_part(image, 50, 186, 300, 400);
    write_ppm_file((char*)"inverted_image.ppm", inverted_image);
    write_ppm_file((char*)"brightened_image.ppm", brightened_image);
    write_ppm_file((char*)"flipped_image.ppm", flipped_image);
    write_ppm_file((char*)"rotated_image.ppm", rotated_image);
    write_ppm_file((char*)"extracted_image.ppm", extracted_image);
    return 0;
}