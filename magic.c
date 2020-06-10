#include <stdio.h>
#include <math.h>

int main(){

    FILE* fp;
    char buffer[6800];
    int i=0;
    fp = fopen("/home/max/btl_workspace/guppi_58626_J0332+5434_0018.0000.raw","rb");         

    if(fp == NULL){
        printf("Error opening file");
    }

    fread(&buffer,sizeof(buffer),1,fp);
    fwrite(&buffer, 1, 6800, stdout);

    printf("\n");
    fclose(fp);

};