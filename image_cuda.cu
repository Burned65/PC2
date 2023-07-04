#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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


__host__ __device__ void brighten_pixel(unsigned char *a, unsigned char *b, int i, int factor) {
    int red = a[i*3] + factor;
    int green = a[i*3+1] + factor;
    int blue = a[i*3+2] + factor;
    if (red > 255) {
        red = 255;
    }
    if (green > 255) {
        green = 255;
    }
    if (blue > 255) {
        blue = 255;
    }
    b[i*3] = (unsigned char) red;
    b[i*3+1] = (unsigned char) green;
    b[i*3+2] = (unsigned char) blue;
}

__global__ void brighten_image(unsigned char* a, unsigned char* b, int factor, int size) {
    int tid = threadIdx.x;    //lokaler Thread Index
    int bid = blockIdx.x;     //Index des Blockes
    int bdim = blockDim.x;     //Anzahl an Threads pro Block

    int i = tid+bid*bdim;     //Globale Adresse

    if (i<size) { //Fehlerbehandlung
        brighten_pixel(a, b, i, factor);
    }

}

__host__ __device__ void flip_pixel(unsigned char* a, unsigned char *b, int i, int width) {
    int index_in_row = i % width;
    int start_index = i - index_in_row;
    int end_index = start_index+width;
    b[3*(end_index-index_in_row)] = a[3*(start_index+index_in_row)];
    b[3*(end_index-index_in_row)+1] = a[3*(start_index+index_in_row)+1];
    b[3*(end_index-index_in_row)+2] = a[3*(start_index+index_in_row)+2];
}

__global__ void flip_image(unsigned char* a, unsigned char* b, int width, int size) {
    int tid = threadIdx.x;    //lokaler Thread Index
    int bid = blockIdx.x;     //Index des Blockes
    int bdim = blockDim.x;     //Anzahl an Threads pro Block

    int i = tid+bid*bdim;     //Globale Adresse

    if (i<size) { //Fehlerbehandlung
        flip_pixel(a, b, i, width);
    }
}

PPMImage * get_image(unsigned char*data, int width, int height, int max_value) {
    PPMImage* new_image = (PPMImage*) malloc(sizeof(PPMImage));
    new_image->width = width;
    new_image->height = height;
    new_image->max_value = max_value;
    new_image->data = data;
    return new_image;
}

int main(int argc, char**argv)
{
    PPMImage* image = read_ppm_file((char*)"a.ppm");
    //Problemgröße
    int size=image->width*image->height;
    //Pointer auf Host/Device Speicher
    unsigned char *a_host, *b_host, *a_dev, *b_dev;

    //Allokiere Host-Speicher
    b_host = (unsigned char*)malloc(size*3*sizeof(unsigned char));
    a_host = image->data;

    //Allokiere Device Speicher
    //Achtung: (void**)& sehr wichtig
    cudaMalloc((void**)&a_dev,size*3*sizeof(unsigned char));
    cudaMalloc((void**)&b_dev,size*3*sizeof(unsigned char));

    //Kopiere Host->Device
    cudaMemcpy(a_dev,a_host,size*3*sizeof(unsigned char),cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 grid(size/threads.x);

    //Starte Kernel mit Konfiguration <<<grid,threads>>> auf Device Speicher
    //Wichtig: Spitze Klammern <<<>>> nicht vergessen!
    //Kernel wird asynchron zu CPU ausgeführt, d.h. hier könnte die CPU noch Arbeit verrichten
    brighten_image<<<grid,threads>>>(a_dev,b_dev,100,size);

    //Kopiere Ergebnis zurück (implizite Synchronisierung)
    cudaMemcpy(b_host,b_dev,size*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    PPMImage* brightened_image = get_image(b_host, image->width, image->height, image->max_value);

    write_ppm_file((char*)"brightened_image.ppm", brightened_image);

    cudaMemcpy(a_dev,a_host,size*3*sizeof(unsigned char),cudaMemcpyHostToDevice);

    flip_image<<<grid,threads>>>(a_dev,b_dev,image->width,size);

    cudaMemcpy(b_host,b_dev,size*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    PPMImage* flipped_image = get_image(b_host, image->width, image->height, image->max_value);

    write_ppm_file((char*)"flipped_image.ppm", flipped_image);

    //Gib Speicher wieder frei
    cudaFree(a_dev);
    cudaFree(b_dev);
    free(a_host);
    free(b_host);
    return 0;
}
